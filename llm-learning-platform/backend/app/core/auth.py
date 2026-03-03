"""
Authentication and authorization system.
Supports API key and JWT token authentication.
"""

import os
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import wraps
import logging

try:
    from jose import JWTError, jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.requests import Request

from app.core.exceptions import AuthenticationError, AuthorizationError, RateLimitExceededError


logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer(auto_error=False)

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
API_KEY_HEADER = "X-API-Key"

# In-memory API key storage (replace with database in production)
# Format: {api_key: {"user_id": str, "name": str, "permissions": List[str], "created_at": str}}
_api_keys: Dict[str, Dict[str, Any]] = {}

# Rate limiting storage
_rate_limit_store: Dict[str, List[datetime]] = {}
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds


class User:
    """User model."""
    
    def __init__(self, user_id: str, email: Optional[str] = None, 
                 name: Optional[str] = None, permissions: Optional[List[str]] = None):
        self.user_id = user_id
        self.email = email
        self.name = name
        self.permissions = permissions or ["read"]
        self.created_at = datetime.utcnow().isoformat()
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions or "admin" in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "name": self.name,
            "permissions": self.permissions,
            "created_at": self.created_at
        }


class TokenData:
    """Token payload data."""
    
    def __init__(self, user_id: str, permissions: List[str], exp: Optional[datetime] = None):
        self.user_id = user_id
        self.permissions = permissions
        self.exp = exp


def create_access_token(user_id: str, permissions: Optional[List[str]] = None, 
                       expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    if not JWT_AVAILABLE:
        raise ImportError("PyJWT not installed. Install with: pip install python-jose")
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode = {
        "sub": user_id,
        "permissions": permissions or ["read"],
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> TokenData:
    """Decode and validate JWT token."""
    if not JWT_AVAILABLE:
        raise ImportError("PyJWT not installed")
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        permissions = payload.get("permissions", ["read"])
        exp = payload.get("exp")
        
        if user_id is None:
            raise AuthenticationError("Invalid token: missing user_id")
        
        return TokenData(
            user_id=user_id,
            permissions=permissions,
            exp=datetime.fromtimestamp(exp) if exp else None
        )
    except JWTError as e:
        raise AuthenticationError(f"Invalid token: {e}")


def generate_api_key(name: str, user_id: str, permissions: Optional[List[str]] = None) -> str:
    """Generate a new API key."""
    # Generate random key
    key = f"llm_{secrets.token_urlsafe(32)}"
    
    # Store key metadata (don't store the key itself, only hash)
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    _api_keys[key_hash] = {
        "user_id": user_id,
        "name": name,
        "permissions": permissions or ["read"],
        "created_at": datetime.utcnow().isoformat(),
        "last_used": None
    }
    
    logger.info(f"Generated API key for user {user_id}")
    return key


def verify_api_key(api_key: str) -> Optional[User]:
    """Verify API key and return user."""
    if not api_key:
        return None
    
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    key_data = _api_keys.get(key_hash)
    
    if key_data:
        # Update last used
        key_data["last_used"] = datetime.utcnow().isoformat()
        return User(
            user_id=key_data["user_id"],
            name=key_data["name"],
            permissions=key_data["permissions"]
        )
    return None


def revoke_api_key(api_key: str) -> bool:
    """Revoke an API key."""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    if key_hash in _api_keys:
        del _api_keys[key_hash]
        return True
    return False


def check_rate_limit(identifier: str) -> bool:
    """Check if request is within rate limit."""
    now = datetime.utcnow()
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW)
    
    # Get requests in current window
    requests = _rate_limit_store.get(identifier, [])
    requests = [r for r in requests if r > window_start]
    
    if len(requests) >= RATE_LIMIT_REQUESTS:
        return False
    
    requests.append(now)
    _rate_limit_store[identifier] = requests
    return True


def get_rate_limit_info(identifier: str) -> Dict[str, Any]:
    """Get rate limit status for identifier."""
    now = datetime.utcnow()
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW)
    
    requests = _rate_limit_store.get(identifier, [])
    requests = [r for r in requests if r > window_start]
    
    return {
        "limit": RATE_LIMIT_REQUESTS,
        "remaining": max(0, RATE_LIMIT_REQUESTS - len(requests)),
        "reset_at": (requests[0] + timedelta(seconds=RATE_LIMIT_WINDOW)).isoformat() if requests else now.isoformat(),
        "window": RATE_LIMIT_WINDOW
    }


async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security),
                          request: Request = None) -> User:
    """
    Dependency to get current authenticated user.
    Supports both JWT tokens (Bearer) and API keys.
    """
    # Check for rate limiting
    client_ip = request.client.host if request and request.client else "unknown"
    if not check_rate_limit(client_ip):
        rate_info = get_rate_limit_info(client_ip)
        raise RateLimitExceededError(retry_after=RATE_LIMIT_WINDOW)
    
    # Try API key first (from header)
    if request:
        api_key = request.headers.get(API_KEY_HEADER)
        if api_key:
            user = verify_api_key(api_key)
            if user:
                return user
            raise AuthenticationError("Invalid API key")
    
    # Try JWT token
    if credentials:
        token = credentials.credentials
        
        # Check if it's an API key format
        if token.startswith("llm_"):
            user = verify_api_key(token)
            if user:
                return user
            raise AuthenticationError("Invalid API key")
        
        # Try JWT
        try:
            token_data = decode_token(token)
            return User(
                user_id=token_data.user_id,
                permissions=token_data.permissions
            )
        except AuthenticationError:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise AuthenticationError("Authentication failed")
    
    # No credentials provided - allow anonymous read-only access
    return User(user_id="anonymous", permissions=["read"])


async def get_current_active_user(user: User = Depends(get_current_user)) -> User:
    """Get current user with active check."""
    return user


async def require_permissions(*permissions: str):
    """Dependency factory to require specific permissions."""
    async def checker(user: User = Depends(get_current_user)) -> User:
        for permission in permissions:
            if not user.has_permission(permission):
                raise AuthorizationError(
                    f"Permission '{permission}' required. User has: {user.permissions}"
                )
        return user
    return checker


class AuthContext:
    """Context manager for authentication in non-route code."""
    
    def __init__(self, user: User):
        self.user = user
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def check_permission(self, permission: str):
        """Check permission within context."""
        if not self.user.has_permission(permission):
            raise AuthorizationError(f"Permission '{permission}' required")


def admin_required(func):
    """Decorator to require admin permission."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract user from kwargs or args
        user = kwargs.get('user') or kwargs.get('current_user')
        if not user:
            raise AuthenticationError("Authentication required")
        
        if not user.has_permission("admin"):
            raise AuthorizationError("Admin permission required")
        
        return await func(*args, **kwargs)
    return wrapper


def permission_required(*permissions: str):
    """Decorator factory to require specific permissions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get('user') or kwargs.get('current_user')
            if not user:
                raise AuthenticationError("Authentication required")
            
            for permission in permissions:
                if not user.has_permission(permission):
                    raise AuthorizationError(f"Permission '{permission}' required")
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# Default users for development
# =============================================================================

def init_default_users():
    """Initialize default users for development."""
    # Create a default admin API key
    default_key = "llm_dev_admin_key_change_in_production"
    key_hash = hashlib.sha256(default_key.encode()).hexdigest()
    
    if key_hash not in _api_keys:
        _api_keys[key_hash] = {
            "user_id": "admin",
            "name": "Default Admin",
            "permissions": ["read", "write", "delete", "admin"],
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None,
            "is_default": True
        }
        logger.warning("Created default admin API key. CHANGE IN PRODUCTION!")
        logger.warning(f"Default key: {default_key}")


# Initialize on module load
init_default_users()

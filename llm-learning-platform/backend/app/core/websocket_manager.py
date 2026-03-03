"""
WebSocket connection manager with improved reliability.
"""

import asyncio
import logging
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from fastapi import WebSocket, WebSocketDisconnect

from app.core.exceptions import WebSocketError, ConnectionError


logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a WebSocket connection."""
    websocket: WebSocket
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_count: int = 0


class WebSocketConnectionManager:
    """
    Manages WebSocket connections with improved reliability.
    
    Features:
    - Connection pooling per session
    - Automatic cleanup of dead connections
    - Message queuing for disconnected clients
    - Health monitoring
    """
    
    def __init__(self, max_connections_per_session: int = 10,
                 connection_timeout: int = 300):
        self.max_connections_per_session = max_connections_per_session
        self.connection_timeout = connection_timeout
        
        # Session -> ConnectionInfo mapping
        self._connections: Dict[str, Set[ConnectionInfo]] = {}
        self._lock = asyncio.Lock()
        
        # Message queues for disconnected clients (optional persistence)
        self._message_queues: Dict[str, List[Dict]] = {}
        
        # Callbacks
        self._on_connect: Optional[Callable[[str, WebSocket], None]] = None
        self._on_disconnect: Optional[Callable[[str, WebSocket], None]] = None
        self._on_message: Optional[Callable[[str, WebSocket, dict], None]] = None
    
    def set_callbacks(self, 
                     on_connect: Optional[Callable] = None,
                     on_disconnect: Optional[Callable] = None,
                     on_message: Optional[Callable] = None):
        """Set event callbacks."""
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._on_message = on_message
    
    async def connect(self, websocket: WebSocket, session_id: str,
                     metadata: Optional[Dict] = None) -> bool:
        """
        Accept and register a new WebSocket connection.
        
        Returns:
            bool: True if connection was accepted, False if rejected
        """
        try:
            await websocket.accept()
        except Exception as e:
            logger.error(f"Failed to accept WebSocket connection: {e}")
            return False
        
        async with self._lock:
            # Initialize session if needed
            if session_id not in self._connections:
                self._connections[session_id] = set()
            
            # Check connection limit
            if len(self._connections[session_id]) >= self.max_connections_per_session:
                logger.warning(f"Max connections reached for session {session_id}")
                await websocket.close(code=1008, reason="Too many connections")
                return False
            
            # Create connection info
            conn_info = ConnectionInfo(
                websocket=websocket,
                metadata=metadata or {}
            )
            
            self._connections[session_id].add(conn_info)
            logger.info(f"WebSocket connected for session {session_id}. "
                       f"Total connections: {len(self._connections[session_id])}")
        
        # Trigger callback
        if self._on_connect:
            try:
                await self._on_connect(session_id, websocket)
            except Exception as e:
                logger.error(f"Error in on_connect callback: {e}")
        
        return True
    
    async def disconnect(self, websocket: WebSocket, session_id: str) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            if session_id in self._connections:
                # Find and remove the connection
                conn_info_to_remove = None
                for conn_info in self._connections[session_id]:
                    if conn_info.websocket == websocket:
                        conn_info_to_remove = conn_info
                        break
                
                if conn_info_to_remove:
                    self._connections[session_id].discard(conn_info_to_remove)
                    logger.info(f"WebSocket disconnected for session {session_id}")
                    
                    # Clean up empty sessions
                    if not self._connections[session_id]:
                        del self._connections[session_id]
        
        # Trigger callback
        if self._on_disconnect:
            try:
                await self._on_disconnect(session_id, websocket)
            except Exception as e:
                logger.error(f"Error in on_disconnect callback: {e}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket) -> bool:
        """Send a message to a specific connection."""
        try:
            await websocket.send_json(message)
            return True
        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected while sending message")
            return False
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    async def broadcast(self, session_id: str, message: dict,
                       skip_websocket: Optional[WebSocket] = None) -> int:
        """
        Broadcast a message to all connections in a session.
        
        Returns:
            int: Number of successful sends
        """
        disconnected: List[ConnectionInfo] = []
        success_count = 0
        
        async with self._lock:
            connections = list(self._connections.get(session_id, []))
        
        for conn_info in connections:
            if skip_websocket and conn_info.websocket == skip_websocket:
                continue
            
            try:
                await conn_info.websocket.send_json(message)
                conn_info.message_count += 1
                conn_info.last_activity = datetime.utcnow()
                success_count += 1
            except WebSocketDisconnect:
                disconnected.append(conn_info)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(conn_info)
        
        # Clean up disconnected
        if disconnected:
            async with self._lock:
                if session_id in self._connections:
                    for conn_info in disconnected:
                        self._connections[session_id].discard(conn_info)
                        if self._on_disconnect:
                            asyncio.create_task(
                                self._safe_callback(self._on_disconnect, session_id, conn_info.websocket)
                            )
        
        return success_count
    
    async def send_to_session(self, session_id: str, message: dict) -> bool:
        """Send message to at least one connection in session."""
        async with self._lock:
            connections = self._connections.get(session_id)
            if not connections:
                # Queue message for later delivery
                if session_id not in self._message_queues:
                    self._message_queues[session_id] = []
                self._message_queues[session_id].append(message)
                return False
        
        # Try to send to any available connection
        result = await self.broadcast(session_id, message)
        return result > 0
    
    async def receive_json(self, websocket: WebSocket, timeout: Optional[float] = None) -> Optional[dict]:
        """Receive JSON message with timeout and error handling."""
        try:
            if timeout:
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=timeout
                )
            else:
                message = await websocket.receive_json()
            
            # Update activity
            async with self._lock:
                for session_conns in self._connections.values():
                    for conn_info in session_conns:
                        if conn_info.websocket == websocket:
                            conn_info.last_activity = datetime.utcnow()
                            break
            
            return message
        except asyncio.TimeoutError:
            return None
        except WebSocketDisconnect:
            raise
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            return None
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a session's connections."""
        async with self._lock:
            connections = self._connections.get(session_id)
            if not connections:
                return None
            
            return {
                "session_id": session_id,
                "num_connections": len(connections),
                "connections": [
                    {
                        "connected_at": conn_info.connected_at.isoformat(),
                        "last_activity": conn_info.last_activity.isoformat(),
                        "message_count": conn_info.message_count,
                        "metadata": conn_info.metadata
                    }
                    for conn_info in connections
                ]
            }
    
    async def get_all_sessions(self) -> Dict[str, int]:
        """Get count of connections per session."""
        async with self._lock:
            return {
                session_id: len(connections)
                for session_id, connections in self._connections.items()
            }
    
    async def cleanup_stale_connections(self, max_idle_seconds: int = 300) -> int:
        """Remove connections that have been idle too long."""
        now = datetime.utcnow()
        to_remove: List[Tuple[str, ConnectionInfo]] = []
        
        async with self._lock:
            for session_id, connections in self._connections.items():
                for conn_info in connections:
                    idle_time = (now - conn_info.last_activity).total_seconds()
                    if idle_time > max_idle_seconds:
                        to_remove.append((session_id, conn_info))
        
        # Close and remove stale connections
        closed_count = 0
        for session_id, conn_info in to_remove:
            try:
                await conn_info.websocket.close(code=1001, reason="Idle timeout")
            except:
                pass
            
            async with self._lock:
                if session_id in self._connections:
                    self._connections[session_id].discard(conn_info)
            
            if self._on_disconnect:
                await self._safe_callback(
                    self._on_disconnect, session_id, conn_info.websocket
                )
            
            closed_count += 1
        
        if closed_count > 0:
            logger.info(f"Cleaned up {closed_count} stale connections")
        
        return closed_count
    
    async def close_all_connections(self, code: int = 1001, reason: str = "Server shutdown") -> None:
        """Close all WebSocket connections."""
        async with self._lock:
            all_connections = [
                (session_id, conn_info)
                for session_id, connections in self._connections.items()
                for conn_info in connections
            ]
            self._connections.clear()
        
        for session_id, conn_info in all_connections:
            try:
                await conn_info.websocket.close(code=code, reason=reason)
            except:
                pass
            
            if self._on_disconnect:
                await self._safe_callback(
                    self._on_disconnect, session_id, conn_info.websocket
                )
        
        logger.info(f"Closed {len(all_connections)} WebSocket connections")
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs):
        """Safely execute a callback, catching exceptions."""
        try:
            result = callback(*args, **kwargs)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Callback error: {e}")


# Global connection manager instance
_connection_manager: Optional[WebSocketConnectionManager] = None


def get_connection_manager() -> WebSocketConnectionManager:
    """Get global WebSocket connection manager."""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = WebSocketConnectionManager()
    return _connection_manager


def set_connection_manager(manager: WebSocketConnectionManager) -> None:
    """Set global WebSocket connection manager."""
    global _connection_manager
    _connection_manager = manager

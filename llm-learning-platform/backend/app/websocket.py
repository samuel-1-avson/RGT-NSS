"""
WebSocket Real-time Communication Layer

Handles real-time training updates, live tokenization streaming,
and interactive computation visualization via Socket.IO.
"""

import asyncio
import json
import logging
from typing import Callable

import socketio

from app.config import get_settings

logger = logging.getLogger(__name__)

# Create Socket.IO server (async mode for FastAPI)
# CORS restricted to the same origins as the REST API
_settings = get_settings()
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",  # Explicitly allow all origins for the websocket to prevent Vercel handshake failures
    logger=False,
    engineio_logger=False,
)


class RealtimeManager:
    """Manages real-time WebSocket communication."""

    def __init__(self, server: socketio.AsyncServer):
        self.sio = server
        self._setup_handlers()

    def _setup_handlers(self):
        @self.sio.event
        async def connect(sid, environ):
            logger.info(f"Client connected: {sid}")
            await self.sio.emit("connected", {"sid": sid}, to=sid)

        @self.sio.event
        async def disconnect(sid):
            logger.info(f"Client disconnected: {sid}")

        @self.sio.on("join_training")
        async def join_training(sid, data):
            session_id = data.get("session_id", "")
            room = f"training_{session_id}"
            self.sio.enter_room(sid, room)
            await self.sio.emit("joined", {"room": room}, to=sid)

        @self.sio.on("leave_training")
        async def leave_training(sid, data):
            session_id = data.get("session_id", "")
            room = f"training_{session_id}"
            self.sio.leave_room(sid, room)

        @self.sio.on("subscribe_tokenization")
        async def subscribe_tokenization(sid, data):
            self.sio.enter_room(sid, "tokenization_stream")
            await self.sio.emit("subscribed", {"channel": "tokenization"}, to=sid)

        @self.sio.on("subscribe_attention")
        async def subscribe_attention(sid, data):
            self.sio.enter_room(sid, "attention_stream")
            await self.sio.emit("subscribed", {"channel": "attention"}, to=sid)

    async def emit_training_update(self, session_id: str, metrics: dict):
        """Broadcast training metrics to subscribers."""
        room = f"training_{session_id}"
        await self.sio.emit("training_update", {
            "session_id": session_id,
            "metrics": metrics,
        }, room=room)

    async def emit_training_complete(self, session_id: str, summary: dict):
        """Notify training completion."""
        room = f"training_{session_id}"
        await self.sio.emit("training_complete", {
            "session_id": session_id,
            "summary": summary,
        }, room=room)

    async def emit_tokenization_step(self, step_data: dict):
        """Stream BPE merge steps during tokenizer training."""
        await self.sio.emit(
            "tokenization_step",
            step_data,
            room="tokenization_stream",
        )

    async def emit_attention_update(self, attention_data: dict):
        """Stream attention computation steps."""
        await self.sio.emit(
            "attention_update",
            attention_data,
            room="attention_stream",
        )

    async def emit_generation_token(self, sid: str, token_data: dict):
        """Stream generated tokens one at a time."""
        await self.sio.emit("generation_token", token_data, to=sid)

    def training_callback(self, session_id: str) -> Callable:
        """Create a callback for the training engine to emit updates."""
        async def callback(metrics: dict):
            await self.emit_training_update(session_id, metrics)
        return callback


# Global instance
realtime = RealtimeManager(sio)

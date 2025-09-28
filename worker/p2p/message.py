# worker/p2p/message.py
"""Message handling for P2P communication."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Callable, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of P2P messages."""
    PING = "ping"
    PONG = "pong"
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    HEARTBEAT = "heartbeat"
    DISCOVERY = "discovery"
    CUSTOM = "custom"


@dataclass
class P2PMessage:
    """A P2P message structure."""
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    timestamp: float
    data: Dict[str, Any]
    message_id: Optional[str] = None

    def to_json(self) -> str:
        """Serialize message to JSON."""
        msg_dict = asdict(self)
        msg_dict["message_type"] = self.message_type.value
        return json.dumps(msg_dict)

    @classmethod
    def from_json(cls, json_str: str) -> "P2PMessage":
        """Deserialize message from JSON."""
        data = json.loads(json_str)
        data["message_type"] = MessageType(data["message_type"])
        return cls(**data)

    @classmethod
    def create_ping(cls, sender_id: str, recipient_id: str) -> "P2PMessage":
        """Create a ping message."""
        return cls(
            message_type=MessageType.PING,
            sender_id=sender_id,
            recipient_id=recipient_id,
            timestamp=time.time(),
            data={}
        )

    @classmethod
    def create_pong(cls, sender_id: str, recipient_id: str) -> "P2PMessage":
        """Create a pong message."""
        return cls(
            message_type=MessageType.PONG,
            sender_id=sender_id,
            recipient_id=recipient_id,
            timestamp=time.time(),
            data={}
        )

    @classmethod
    def create_heartbeat(cls, sender_id: str, node_info: Dict[str, Any]) -> "P2PMessage":
        """Create a heartbeat message."""
        return cls(
            message_type=MessageType.HEARTBEAT,
            sender_id=sender_id,
            recipient_id=None,  # Broadcast
            timestamp=time.time(),
            data=node_info
        )


class MessageHandler:
    """Handles P2P message routing and callbacks."""

    def __init__(self):
        self._handlers: Dict[MessageType, Callable[[
            P2PMessage], Optional[P2PMessage]]] = {}
        self._default_handler: Optional[Callable[[
            P2PMessage], Optional[P2PMessage]]] = None

    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[P2PMessage], Optional[P2PMessage]]
    ) -> None:
        """Register a message handler for a specific message type."""
        self._handlers[message_type] = handler
        logger.debug(f"Registered handler for message type: {message_type}")

    def set_default_handler(
        self,
        handler: Callable[[P2PMessage], Optional[P2PMessage]]
    ) -> None:
        """Set a default handler for unregistered message types."""
        self._default_handler = handler

    def handle_message(self, message: P2PMessage) -> Optional[P2PMessage]:
        """Handle an incoming message and return a response if needed."""
        try:
            handler = self._handlers.get(
                message.message_type, self._default_handler)
            if handler:
                return handler(message)
            else:
                logger.warning(
                    f"No handler for message type: {message.message_type}")
                return None
        except Exception as e:
            logger.error(f"Error handling message {message.message_type}: {e}")
            return None

    def _default_ping_handler(self, message: P2PMessage) -> P2PMessage:
        """Default ping handler that responds with pong."""
        return P2PMessage.create_pong(
            sender_id=message.recipient_id or "unknown",
            recipient_id=message.sender_id
        )

    def setup_default_handlers(self, node_id: str) -> None:
        """Setup default message handlers."""
        def ping_handler(message: P2PMessage) -> P2PMessage:
            return P2PMessage.create_pong(sender_id=node_id, recipient_id=message.sender_id)

        def pong_handler(message: P2PMessage) -> None:
            logger.debug(f"Received pong from {message.sender_id}")
            return None

        def heartbeat_handler(message: P2PMessage) -> None:
            logger.debug(
                "Received heartbeat from %s with stats: %s",
                message.sender_id,
                message.data.get("stats") if isinstance(
                    message.data, dict) else message.data,
            )
            return None

        self.register_handler(MessageType.PING, ping_handler)
        self.register_handler(MessageType.PONG, pong_handler)
        self.register_handler(MessageType.HEARTBEAT, heartbeat_handler)

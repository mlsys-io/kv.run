# worker/p2p/__init__.py
"""
P2P networking module based on libp2p.

This module provides a friendly wrapper around libp2p functionality
for distributed worker communication.
"""
from .node import P2PNode
from .discovery import DiscoveryService
from .message import MessageHandler, MessageType, P2PMessage
from .config import P2PConfig

__all__ = [
    "P2PNode",
    "DiscoveryService",
    "MessageHandler",
    "MessageType",
    "P2PMessage",
    "P2PConfig"
]

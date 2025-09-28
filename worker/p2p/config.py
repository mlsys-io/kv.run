# worker/p2p/config.py
"""P2P configuration settings."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class P2PConfig:
    """Configuration for P2P networking."""

    # Node identity
    node_id: str

    # Network settings
    listen_port: int
    listen_host: str

    # Discovery service
    discovery_url: str
    discovery_interval_sec: int

    # Connection settings
    max_connections: int
    connection_timeout_sec: int

    # Message handling
    message_buffer_size: int
    heartbeat_interval_sec: int

    # Optional bootstrap peers
    bootstrap_peers: List[str]

    @staticmethod
    def from_env(worker_id: Optional[str] = None) -> "P2PConfig":
        """Create P2P config from environment variables."""
        node_id = worker_id or os.getenv("P2P_NODE_ID", os.urandom(8).hex())

        listen_port = int(os.getenv("P2P_PORT", "0"))  # 0 = random port
        listen_host = os.getenv("P2P_HOST", "0.0.0.0")

        discovery_url = os.getenv(
            "P2P_DISCOVERY_URL", "http://localhost:8080/discovery")
        discovery_interval = int(os.getenv("P2P_DISCOVERY_INTERVAL", "30"))

        max_connections = int(os.getenv("P2P_MAX_CONNECTIONS", "100"))
        connection_timeout = int(os.getenv("P2P_CONNECTION_TIMEOUT", "10"))

        message_buffer_size = int(os.getenv("P2P_MESSAGE_BUFFER_SIZE", "1024"))
        heartbeat_interval = int(os.getenv("P2P_HEARTBEAT_INTERVAL", "30"))

        bootstrap_peers_str = os.getenv("P2P_BOOTSTRAP_PEERS", "")
        bootstrap_peers = [p.strip()
                           for p in bootstrap_peers_str.split(",") if p.strip()]

        return P2PConfig(
            node_id=node_id,
            listen_port=listen_port,
            listen_host=listen_host,
            discovery_url=discovery_url,
            discovery_interval_sec=discovery_interval,
            max_connections=max_connections,
            connection_timeout_sec=connection_timeout,
            message_buffer_size=message_buffer_size,
            heartbeat_interval_sec=heartbeat_interval,
            bootstrap_peers=bootstrap_peers,
        )

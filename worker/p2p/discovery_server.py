# worker/p2p/discovery_server.py
"""
Simple HTTP server for centralized P2P service discovery.

This is a reference implementation of the discovery service that P2P nodes
can register with and discover each other through.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List
import logging
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)


class NodeRegistration(BaseModel):
    """Node registration data model."""
    node_id: str
    address: str
    port: int
    last_seen: float
    capabilities: List[str]
    metadata: Dict[str, str]
    # Optional libp2p peer ID to construct full multiaddr (/p2p/<peer_id>)
    peer_id: str | None = None


class DiscoveryServer:
    """Simple in-memory discovery server."""

    def __init__(self):
        self.nodes: Dict[str, NodeRegistration] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.max_age = 600  # 10 minutes

    def register_node(self, node_data: NodeRegistration) -> None:
        """Register or update a node."""
        self.nodes[node_data.node_id] = node_data
        logger.info(f"Registered node: {node_data.node_id}")

    def unregister_node(self, node_id: str) -> bool:
        """Unregister a node."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Unregistered node: {node_id}")
            return True
        return False

    def get_nodes(self) -> List[NodeRegistration]:
        """Get all active nodes."""
        current_time = time.time()
        active_nodes = []

        for node in list(self.nodes.values()):
            if current_time - node.last_seen < self.max_age:
                active_nodes.append(node)
            else:
                # Remove stale nodes
                self.nodes.pop(node.node_id, None)
                logger.info(f"Removed stale node: {node.node_id}")

        return active_nodes

    def get_node(self, node_id: str) -> NodeRegistration | None:
        """Get a specific node."""
        node = self.nodes.get(node_id)
        if node:
            current_time = time.time()
            if current_time - node.last_seen < self.max_age:
                return node
            else:
                # Remove stale node
                self.nodes.pop(node_id, None)
                logger.info(f"Removed stale node: {node_id}")
        return None


# Global discovery server instance
discovery_server = DiscoveryServer()


if FASTAPI_AVAILABLE:
    # FastAPI application
    app = FastAPI(title="P2P Discovery Service", version="1.0.0")

    @app.post("/nodes")
    async def register_node(node_data: NodeRegistration):
        """Register or update a node."""
        node_data.last_seen = time.time()  # Update timestamp
        discovery_server.register_node(node_data)
        return {"status": "registered", "node_id": node_data.node_id}

    @app.get("/nodes")
    async def get_nodes():
        """Get all active nodes."""
        nodes = discovery_server.get_nodes()
        return [node.dict() for node in nodes]

    @app.get("/nodes/{node_id}")
    async def get_node(node_id: str):
        """Get a specific node."""
        node = discovery_server.get_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail="Node not found")
        return node.dict()

    @app.delete("/nodes/{node_id}")
    async def unregister_node(node_id: str):
        """Unregister a node."""
        success = discovery_server.unregister_node(node_id)
        if not success:
            raise HTTPException(status_code=404, detail="Node not found")
        return {"status": "unregistered", "node_id": node_id}

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "nodes_count": len(discovery_server.nodes),
            "timestamp": time.time()
        }

    def run_server(host: str = "0.0.0.0", port: int = 8080):
        """Run the discovery server."""
        logger.info(f"Starting discovery server on {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")

else:
    def run_server(host: str = "0.0.0.0", port: int = 8080):
        """Fallback when FastAPI is not available."""
        logger.error("FastAPI not available. Cannot run discovery server.")
        logger.error("Install with: pip install fastapi uvicorn")


if __name__ == "__main__":
    import sys

    if not FASTAPI_AVAILABLE:
        print("FastAPI and uvicorn are required to run the discovery server.")
        print("Install with: pip install fastapi uvicorn")
        sys.exit(1)

    # Simple CLI
    import argparse
    parser = argparse.ArgumentParser(description="P2P Discovery Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port to bind to")

    args = parser.parse_args()
    run_server(args.host, args.port)

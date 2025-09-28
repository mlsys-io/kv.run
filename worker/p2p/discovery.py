# worker/p2p/discovery.py
"""Centralized service discovery for P2P nodes."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set
import logging
import urllib.request
import urllib.parse
import urllib.error

logger = logging.getLogger(__name__)


class SimpleHttpClient:
    """Simple HTTP client used when httpx is unavailable."""

    def __init__(self, timeout=10.0):
        self.timeout = timeout

    async def post(self, url: str, json: dict):
        """Send a POST request."""
        # Run a synchronous HTTP request in an async environment
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_post, url, json)

    async def get(self, url: str):
        """Send a GET request."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_get, url)

    async def delete(self, url: str):
        """Send a DELETE request."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_delete, url)

    async def aclose(self):
        """Close the client."""
        pass  # No-op for this simple implementation

    def _sync_post(self, url: str, json_data: dict):
        """Synchronous POST request."""
        try:
            data = json.dumps(json_data).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return SimpleHttpResponse(response.read(), response.status)
        except urllib.error.HTTPError as e:
            return SimpleHttpResponse(e.read(), e.code)
        except Exception as e:
            raise Exception(f"HTTP POST failed: {e}")

    def _sync_get(self, url: str):
        """Synchronous GET request."""
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                return SimpleHttpResponse(response.read(), response.status)
        except urllib.error.HTTPError as e:
            return SimpleHttpResponse(e.read(), e.code)
        except Exception as e:
            raise Exception(f"HTTP GET failed: {e}")

    def _sync_delete(self, url: str):
        """Synchronous DELETE request."""
        try:
            req = urllib.request.Request(url)
            req.get_method = lambda: 'DELETE'
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return SimpleHttpResponse(response.read(), response.status)
        except urllib.error.HTTPError as e:
            return SimpleHttpResponse(e.read(), e.code)
        except Exception as e:
            raise Exception(f"HTTP DELETE failed: {e}")


class SimpleHttpResponse:
    """Simple HTTP response class."""

    def __init__(self, content: bytes, status_code: int):
        self.content = content
        self.status_code = status_code

    def json(self):
        """Parse the JSON response."""
        return json.loads(self.content.decode('utf-8'))

    def raise_for_status(self):
        """Raise an exception for 4xx/5xx status codes."""
        if self.status_code >= 400:
            raise Exception(
                f"HTTP {self.status_code}: {self.content.decode()}")


@dataclass
class NodeInfo:
    """Information about a P2P node."""
    node_id: str
    address: str
    port: int
    last_seen: float
    capabilities: List[str]
    metadata: Dict[str, str]
    peer_id: Optional[str] = None  # libp2p peer ID

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "NodeInfo":
        """Create from dictionary."""
        return cls(**data)

    @property
    def multiaddr(self) -> str:
        """Get libp2p multiaddress format."""
        base_addr = f"/ip4/{self.address}/tcp/{self.port}"
        if self.peer_id:
            return f"{base_addr}/p2p/{self.peer_id}"
        return base_addr


class DiscoveryService:
    """Handles centralized service discovery for P2P nodes."""

    def __init__(self, discovery_url: str, node_info: NodeInfo, interval_sec: int = 30):
        self.discovery_url = discovery_url.rstrip('/')
        self.node_info = node_info
        self.interval_sec = interval_sec
        self._known_peers: Dict[str, NodeInfo] = {}
        self._running = False
        self._discovery_task: Optional[asyncio.Task] = None
        self._client: Optional[SimpleHttpClient] = None

    async def start(self) -> None:
        """Start the discovery service."""
        if self._running:
            return

        self._running = True
    # Use the simple HTTP client (no httpx dependency)
        self._client = SimpleHttpClient(timeout=10.0)
        await self._register_node()

        # Start periodic discovery
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        logger.info(
            f"Started discovery service for node {self.node_info.node_id}")

    async def stop(self) -> None:
        """Stop the discovery service."""
        if not self._running:
            return

        self._running = False

        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass

        # Unregister ourselves
        if self._client:
            try:
                await self._unregister_node()
            except Exception as e:
                logger.warning(f"Failed to unregister node: {e}")
            finally:
                await self._client.aclose()
                self._client = None

        logger.info(
            f"Stopped discovery service for node {self.node_info.node_id}")

    async def get_peers(self) -> List[NodeInfo]:
        """Get list of known peers."""
        return list(self._known_peers.values())

    async def find_peer(self, node_id: str) -> Optional[NodeInfo]:
        """Find a specific peer by node ID."""
        return self._known_peers.get(node_id)

    async def update_node_info(self, **kwargs) -> None:
        """Update our node information."""
        for key, value in kwargs.items():
            if hasattr(self.node_info, key):
                setattr(self.node_info, key, value)
        self.node_info.last_seen = time.time()

        if self._running:
            await self._register_node()

    async def _discovery_loop(self) -> None:
        """Main discovery loop."""
        while self._running:
            try:
                await self._register_node()
                await self._fetch_peers()
                await asyncio.sleep(self.interval_sec)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Discovery loop error: {e}")
                await asyncio.sleep(5)  # Shorter retry interval on error

    async def _register_node(self) -> None:
        """Register our node with the discovery service."""
        if not self._client:
            return

        try:
            self.node_info.last_seen = time.time()
            response = await self._client.post(
                f"{self.discovery_url}/nodes",
                json=self.node_info.to_dict()
            )
            response.raise_for_status()
            logger.debug(f"Registered node {self.node_info.node_id}")
        except Exception as e:
            logger.error(f"Failed to register node: {e}")

    async def _unregister_node(self) -> None:
        """Unregister our node from the discovery service."""
        if not self._client:
            return

        try:
            response = await self._client.delete(
                f"{self.discovery_url}/nodes/{self.node_info.node_id}"
            )
            response.raise_for_status()
            logger.debug(f"Unregistered node {self.node_info.node_id}")
        except Exception as e:
            logger.error(f"Failed to unregister node: {e}")

    async def _fetch_peers(self) -> None:
        """Fetch list of peers from the discovery service."""
        if not self._client:
            return

        try:
            response = await self._client.get(f"{self.discovery_url}/nodes")
            response.raise_for_status()

            nodes_data = response.json()
            new_peers = {}

            for node_data in nodes_data:
                node_info = NodeInfo.from_dict(node_data)
                if node_info.node_id != self.node_info.node_id:  # Exclude ourselves
                    new_peers[node_info.node_id] = node_info

            # Update known peers
            old_peer_ids = set(self._known_peers.keys())
            new_peer_ids = set(new_peers.keys())

            # Log new peers
            for peer_id in new_peer_ids - old_peer_ids:
                logger.info(f"Discovered new peer: {peer_id}")

            # Log departed peers
            for peer_id in old_peer_ids - new_peer_ids:
                logger.info(f"Peer departed: {peer_id}")

            self._known_peers = new_peers

        except Exception as e:
            logger.error(f"Failed to fetch peers: {e}")

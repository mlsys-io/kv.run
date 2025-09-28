# worker/p2p/node.py
"""P2P Node implementation using libp2p."""
from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, List, Optional, Callable, Any
import logging

from .config import P2PConfig
from .discovery import DiscoveryService, NodeInfo
from .message import MessageHandler, P2PMessage, MessageType

logger = logging.getLogger(__name__)

try:
    from libp2p import new_host, IHost
    from libp2p.network.stream.net_stream import INetStream
    from libp2p.peer.peerinfo import info_from_p2p_addr
    from multiaddr import Multiaddr
    import trio
    import threading
    import queue
    import sniffio
    LIBP2P_AVAILABLE = True
except ImportError as e:
    logger.warning(
        f"libp2p not available, will use in-process fallback if needed: {e}")
    # Provide minimal shims so type hints remain optional
    new_host = None  # type: ignore
    IHost = object  # type: ignore
    INetStream = object  # type: ignore
    info_from_p2p_addr = None  # type: ignore
    Multiaddr = None  # type: ignore
    import threading
    import queue
    import trio
    LIBP2P_AVAILABLE = False

# No in-process fallback; require libp2p to work


class TrioHostAdapter:
    """Adapter to use libp2p Host in asyncio context using dedicated trio thread"""

    def __init__(self):
        self._host = None
        self._trio_thread = None
        self._request_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._running = False
        self._host_run_stop_event = None
        self._current_listen_addr = None

    async def create_host(self):
        """Create libp2p host in trio context"""
        await self._start_trio_thread()

        # Send create host request
        self._request_queue.put(("create_host", None))

        # Wait for response
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self._response_queue.get)

        if isinstance(response, Exception):
            raise response

        self._host = response
        return self._host

    async def listen(self, multiaddr):
        """Start listening on multiaddr"""
        if not self._host:
            raise RuntimeError("Host not created")

        logger.info(f"Sending listen request for {multiaddr}")
        self._request_queue.put(("listen", multiaddr))

        loop = asyncio.get_event_loop()
        try:
            # Add timeout to prevent hanging (extended)
            response = await asyncio.wait_for(
                loop.run_in_executor(None, self._response_queue.get),
                timeout=30.0
            )
            logger.info(f"Listen response received: {response}")
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for listen response")
            raise RuntimeError("Timeout waiting for listen response")

        if isinstance(response, Exception):
            raise response

    async def new_stream(self, peer_id, protocols):
        """Create new stream to peer"""
        if not self._host:
            raise RuntimeError("Host not created")

        self._request_queue.put(("new_stream", (peer_id, protocols)))

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self._response_queue.get)

        if isinstance(response, Exception):
            raise response

        return response

    async def write_stream(self, stream, data: bytes) -> bool:
        """Write data to a libp2p stream within the trio context."""
        if not self._host:
            raise RuntimeError("Host not created")

        self._request_queue.put(("stream_write", (stream, data)))

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self._response_queue.get)

        if isinstance(response, Exception):
            raise response

        return bool(response)

    async def close_stream(self, stream) -> None:
        """Close a libp2p stream inside the trio context."""
        if not self._host:
            return

        self._request_queue.put(("stream_close", stream))

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self._response_queue.get)

        if isinstance(response, Exception):
            raise response

    def get_id(self):
        """Get host ID"""
        if not self._host:
            raise RuntimeError("Host not created")
        return self._host.get_id()

    def get_addrs(self):
        """Get host addresses"""
        if not self._host:
            raise RuntimeError("Host not created")
        return self._host.get_addrs()

    def get_peerstore(self):
        """Get peerstore"""
        if not self._host:
            raise RuntimeError("Host not created")
        return self._host.get_peerstore()

    def set_stream_handler(self, protocol, handler):
        """Set stream handler"""
        if not self._host:
            raise RuntimeError("Host not created")
        return self._host.set_stream_handler(protocol, handler)

    async def _start_trio_thread(self):
        """Start trio thread for libp2p operations"""
        if self._running:
            return

        self._running = True

        def trio_worker():
            """Worker function that runs in trio thread"""
            trio.run(self._trio_main)

        self._trio_thread = threading.Thread(target=trio_worker, daemon=True)
        self._trio_thread.start()

        # Wait a moment for trio to start
        await asyncio.sleep(0.1)

    async def _trio_main(self):
        """Main trio function that handles requests"""
        host = None
        logger.info("Trio main loop started")

        async with trio.open_nursery() as nursery:
            while self._running:
                try:
                    # Check for new requests (non-blocking)
                    try:
                        request_type, args = self._request_queue.get_nowait()
                        logger.info(f"Processing trio request: {request_type}")
                    except queue.Empty:
                        await trio.sleep(0.01)  # Small delay
                        continue

                    try:
                        if request_type == "create_host":
                            logger.info("Creating libp2p host in trio context")
                            host = new_host()
                            logger.info(
                                f"Host created successfully: {host.get_id()}")
                            self._response_queue.put(host)
                        elif request_type == "listen":
                            if host:
                                if self._host_run_stop_event is not None:
                                    logger.warning(
                                        "Listen already active; skipping duplicate request")
                                    self._response_queue.put("OK")
                                    continue

                                logger.info(
                                    f"Starting to listen via host.run on {args}")

                                ready_event = trio.Event()
                                result_holder: Dict[str, Any] = {}

                                async def host_runner(listen_addr):
                                    cm = host.run([listen_addr])
                                    try:
                                        await cm.__aenter__()
                                    except Exception as exc:
                                        logger.error(
                                            f"Host.run failed during listen startup: {exc}")
                                        result_holder["error"] = exc
                                        ready_event.set()
                                        return

                                    self._host_run_stop_event = trio.Event()
                                    self._current_listen_addr = listen_addr
                                    result_holder["status"] = "ok"
                                    ready_event.set()

                                    try:
                                        await self._host_run_stop_event.wait()
                                    finally:
                                        try:
                                            await cm.__aexit__(None, None, None)
                                        except Exception as exit_error:
                                            logger.error(
                                                f"Error while shutting down host.run: {exit_error}")
                                        self._host_run_stop_event = None
                                        self._current_listen_addr = None

                                nursery.start_soon(host_runner, args)

                                await ready_event.wait()

                                if "error" in result_holder:
                                    self._response_queue.put(
                                        result_holder["error"])
                                else:
                                    addrs = host.get_addrs()
                                    if addrs:
                                        logger.info(
                                            f"Host addresses after listen start: {addrs}")
                                    else:
                                        logger.warning(
                                            "Host.run started but no addresses reported yet")
                                    self._response_queue.put("OK")
                            else:
                                logger.error("Cannot listen: host not created")
                                self._response_queue.put(
                                    RuntimeError("Host not created"))
                        elif request_type == "new_stream":
                            if host:
                                peer_id, protocols = args
                                logger.info(f"Creating stream to {peer_id}")
                                stream = await host.new_stream(peer_id, protocols)
                                logger.info("Stream created successfully")
                                self._response_queue.put(stream)
                            else:
                                self._response_queue.put(
                                    RuntimeError("Host not created"))
                        elif request_type == "stream_write":
                            stream, data = args
                            if stream:
                                try:
                                    await stream.write(data)
                                    self._response_queue.put(True)
                                except Exception as exc:
                                    logger.debug(
                                        "Stream write failed inside trio worker: %s",
                                        exc,
                                    )
                                    self._response_queue.put(exc)
                            else:
                                self._response_queue.put(
                                    RuntimeError("Invalid stream"))
                        elif request_type == "stream_close":
                            stream = args
                            if stream:
                                await stream.close()
                                self._response_queue.put(True)
                            else:
                                self._response_queue.put(
                                    RuntimeError("Invalid stream"))
                        elif request_type == "stop":
                            logger.info("Stopping trio main loop")
                            self._running = False
                            if self._host_run_stop_event:
                                self._host_run_stop_event.set()
                            self._response_queue.put("OK")
                            break

                    except Exception as e:
                        logger.error(
                            f"Error processing trio request {request_type}: {e}")
                        self._response_queue.put(e)

                except Exception as e:
                    logger.error(f"Trio worker error: {e}")
                    self._response_queue.put(e)

        logger.info("Trio main loop ended")

    async def stop(self):
        """Stop the trio thread"""
        if self._running:
            self._request_queue.put(("stop", None))

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._response_queue.get)

            self._running = False


class P2PNode:
    """
    A P2P node that provides friendly interface for distributed communication.

    This class wraps libp2p functionality and provides:
    - Automatic peer discovery through centralized service
    - Message routing and handling
    - Connection management
    - Health monitoring
    """

    def __init__(self, config: P2PConfig):
        self.config = config
        self.node_id = config.node_id
        self.message_handler = MessageHandler()

        # Internal state - use adapter for trio-asyncio bridge
        self._host_adapter: Optional[TrioHostAdapter] = None
        self._discovery: Optional[DiscoveryService] = None
        self._running = False
        self._connections = {}
        self._heartbeat_task = None
        self._fallback_mode = False

        # Setup default message handlers
        self.message_handler.setup_default_handlers(self.node_id)

        # Store actual listening port for proper address resolution
        self._actual_port = None

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "connections_established": 0,
            "connections_lost": 0,
        }

    async def start(self) -> None:
        """Start the P2P node."""
        if self._running:
            return

        logger.info(f"Starting P2P node {self.node_id}")

        try:
            # Start libp2p node; fail fast if it doesn't work
            await self._start_libp2p_node()
            await self._start_discovery()
            await self._start_heartbeat()

            self._running = True
            logger.info(f"P2P node {self.node_id} started successfully")

        except Exception as e:
            logger.error(f"Failed to start P2P node: {e}")
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the P2P node."""
        if not self._running:
            return

        logger.info(f"Stopping P2P node {self.node_id}")
        self._running = False

        # Stop heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Stop discovery
        if self._discovery:
            await self._discovery.stop()

        # Close connections
        for connection in list(self._connections.values()):
            try:
                if self._host_adapter:
                    await self._host_adapter.close_stream(connection)
                else:
                    await connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")

        self._connections.clear()

        # Stop libp2p host adapter
        if self._host_adapter:
            try:
                await self._host_adapter.stop()
                logger.info("Libp2p host stopped")
            except Exception as e:
                logger.warning(f"Error stopping libp2p host: {e}")

        # No in-proc registry to clean up

        logger.info(f"P2P node {self.node_id} stopped")

    async def send_message(
        self,
        message: P2PMessage,
        target_node_id: Optional[str] = None
    ) -> bool:
        """
        Send a message to a peer or broadcast to all peers.

        Args:
            message: The message to send
            target_node_id: Specific peer to send to, or None for broadcast

        Returns:
            True if message was sent successfully
        """
        if not self._running:
            logger.warning("Node not running, cannot send message")
            return False

        try:
            if target_node_id:
                return await self._send_to_peer(message, target_node_id)
            else:
                return await self._broadcast_message(message)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False

    async def ping_peer(self, peer_id: str) -> bool:
        """Ping a specific peer and wait for pong response."""
        if not self._host_adapter:
            logger.error(f"Cannot ping {peer_id}: libp2p host not available")
            return False

        ping_msg = P2PMessage.create_ping(self.node_id, peer_id)
        return await self.send_message(ping_msg, peer_id)

    async def get_connected_peers(self) -> List[str]:
        """Get list of currently connected peer IDs."""
        return list(self._connections.keys())

    async def get_discovered_peers(self) -> List[NodeInfo]:
        """Get list of peers discovered through service discovery."""
        if self._discovery:
            return await self._discovery.get_peers()
        return []

    def register_message_handler(
        self,
        message_type: MessageType,
        handler: Callable[[P2PMessage], Optional[P2PMessage]]
    ) -> None:
        """Register a custom message handler."""
        self.message_handler.register_handler(message_type, handler)

    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {
            **self.stats,
            "node_id": self.node_id,
            "running": self._running,
            "connected_peers": len(self._connections),
            "discovered_peers": len(self._discovery._known_peers) if self._discovery else 0,
        }

    async def _start_libp2p_node(self) -> None:
        """Start the libp2p host using trio adapter."""
        try:
            # Create host adapter for trio-asyncio bridge
            self._host_adapter = TrioHostAdapter()

            # Create libp2p host in trio context
            await self._host_adapter.create_host()
            logger.info(
                f"Created libp2p host with ID: {self._host_adapter.get_id()}")

            logger.info("Starting to listen on address...")

            # Use configured host/port when provided; fall back to system assignment
            configured_port = self.config.listen_port
            port = configured_port if configured_port and configured_port > 0 else 0
            bind_host = self.config.listen_host or "0.0.0.0"

            if bind_host == "localhost":
                bind_host = "127.0.0.1"

            # Store the requested port (may be 0 if system-assigned)
            self._actual_port = port

            listen_addr = f"/ip4/{bind_host}/tcp/{port}"
            logger.info(f"Attempting to listen on {listen_addr}")
            await self._host_adapter.listen(Multiaddr(listen_addr))
            logger.info("Listen request acknowledged")

            # Set stream handler for incoming connections
            self._host_adapter.set_stream_handler(
                "/p2p/1.0.0", self._handle_incoming_stream)

            # Get actual listening address and extract port
            addrs = []
            for _ in range(20):
                addrs = self._host_adapter.get_addrs()
                if addrs:
                    break
                await asyncio.sleep(0.1)

            if addrs:
                addr_str = str(addrs[0])
                logger.info(f"Listening on: {addr_str}")
                # Parse the multiaddr to get the actual port
                parts = addr_str.split('/')
                if len(parts) >= 5 and parts[4].isdigit():
                    self._actual_port = int(parts[4])
                    logger.info(
                        f"Extracted actual listening port: {self._actual_port}")
            else:
                logger.warning(
                    "Libp2p listen did not report any addresses; using configured port fallback")
                if self._actual_port == 0 and configured_port and configured_port > 0:
                    self._actual_port = configured_port

        except Exception as e:
            logger.error(f"Failed to create libp2p host: {e}")
            import traceback
            traceback.print_exc()
            # Without fallback mode, libp2p failure should cause node startup to fail
            self._host_adapter = None
            raise RuntimeError(
                f"P2P node startup failed: libp2p host creation failed: {e}")

    # No fallback node

    async def _start_discovery(self) -> None:
        """Start the discovery service."""
        # Use the stored actual port or fall back to config
        listen_addr = self.config.listen_host
        listen_port = self._actual_port or self.config.listen_port

        # Ensure we have a valid port
        if listen_port == 0:
            import random
            listen_port = random.randint(40000, 50000)
            logger.warning(
                f"No valid port found, using random port: {listen_port}")

        # If address is 0.0.0.0, change to 127.0.0.1 for connectivity
        if listen_addr == "0.0.0.0":
            listen_addr = "127.0.0.1"

        # Get peer ID from libp2p host
        peer_id = None
        if self._host_adapter:
            peer_id = str(self._host_adapter.get_id())

        node_info = NodeInfo(
            node_id=self.node_id,
            address=listen_addr,
            port=listen_port,
            last_seen=time.time(),
            capabilities=["worker", "p2p"],
            metadata={"version": "1.0.0"},
            peer_id=peer_id
        )

        logger.info(f"Node {self.node_id} multiaddr: {node_info.multiaddr}")

        self._discovery = DiscoveryService(
            discovery_url=self.config.discovery_url,
            node_info=node_info,
            interval_sec=self.config.discovery_interval_sec
        )

        await self._discovery.start()

    async def _start_heartbeat(self) -> None:
        """Start the heartbeat task."""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to all connected peers."""
        while self._running:
            try:
                heartbeat_msg = P2PMessage.create_heartbeat(
                    sender_id=self.node_id,
                    node_info={"timestamp": time.time(),
                               "stats": self.get_stats()}
                )
                await self._broadcast_message(heartbeat_msg)
                await asyncio.sleep(self.config.heartbeat_interval_sec)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)

    async def _handle_incoming_stream(self, stream) -> None:
        """Handle incoming libp2p stream."""
        peer_id = str(stream.muxed_conn.peer_id)
        logger.info(f"New connection from peer: {peer_id}")

        self._connections[peer_id] = stream
        self.stats["connections_established"] += 1

        try:
            while self._running:
                # Read message from stream
                data = await stream.read(self.config.message_buffer_size)
                if not data:
                    break

                # Process message
                try:
                    message = P2PMessage.from_json(data.decode())
                    self.stats["messages_received"] += 1

                    # Handle message and get response
                    response = self.message_handler.handle_message(message)
                    if response:
                        response_data = response.to_json().encode()
                        await stream.write(response_data)
                        self.stats["messages_sent"] += 1

                except Exception as e:
                    logger.error(
                        f"Error processing message from {peer_id}: {e}")

        except Exception as e:
            message = str(e)
            if message:
                logger.error(f"Stream error with {peer_id}: {message}")
            else:
                logger.debug(f"Stream closed for peer {peer_id}")
        finally:
            # Clean up connection
            self._connections.pop(peer_id, None)
            self.stats["connections_lost"] += 1
            try:
                if self._host_adapter:
                    await self._host_adapter.close_stream(stream)
                else:
                    await stream.close()
            except Exception as close_error:
                logger.debug(
                    "Error while closing stream for %s: %s",
                    peer_id,
                    close_error,
                )
                pass

    async def _send_to_peer(self, message: P2PMessage, peer_id: str) -> bool:
        """Send message to a specific peer."""
        if not self._host_adapter:
            logger.error(
                f"Cannot send to {peer_id}: libp2p host not available")
            return False

        # Check if we have an active connection
        if peer_id in self._connections:
            return await self._send_via_connection(message, peer_id)

        # Try to establish new connection
        if await self._connect_to_peer(peer_id):
            return await self._send_via_connection(message, peer_id)

        return False

    async def _broadcast_message(self, message: P2PMessage) -> bool:
        """Broadcast message to all connected peers."""
        if not self._host_adapter:
            logger.error("Cannot broadcast: libp2p host not available")
            return False

        if not self._connections:
            logger.warning("No connected peers to broadcast to")
            return True  # No peers to send to, but not an error

        success_count = 0
        for peer_id in list(self._connections.keys()):
            if await self._send_via_connection(message, peer_id):
                success_count += 1

        return success_count > 0

    # Removed in-proc broadcast

    async def _send_via_connection(self, message: P2PMessage, peer_id: str) -> bool:
        """Send message via existing connection."""
        connection = self._connections.get(peer_id)
        if not connection:
            return False

        try:
            data = message.to_json().encode()
            if self._host_adapter:
                success = await self._host_adapter.write_stream(connection, data)
                if not success:
                    raise RuntimeError("Stream write rejected")
            else:
                await connection.write(data)
            self.stats["messages_sent"] += 1
            return True
        except Exception as e:
            error_message = str(e)
            if "stream is closed" in error_message.lower():
                logger.info(
                    "Connection to %s is closed; dropping stream before retrying",
                    peer_id,
                )
            else:
                logger.error(f"Failed to send message to {peer_id}: {e}")
            # Remove failed connection
            self._connections.pop(peer_id, None)
            return False

    # Removed in-proc send

    async def _connect_to_peer(self, peer_id: str) -> bool:
        """Establish connection to a peer."""
        if not self._discovery:
            return False

        peer_info = await self._discovery.find_peer(peer_id)
        if not peer_info:
            logger.warning(f"Peer {peer_id} not found in discovery")
            return False

        if not self._host_adapter:
            logger.error("Cannot connect: libp2p host not available")
            return False

        try:
            # Create multiaddr for the peer
            logger.info(
                f"Attempting to connect to {peer_id} at {peer_info.multiaddr}")
            peer_addr = Multiaddr(peer_info.multiaddr)
            peer_info_obj = info_from_p2p_addr(peer_addr)

            # Add peer address to peer store
            self._host_adapter.get_peerstore().add_addrs(
                peer_info_obj.peer_id, [peer_addr], 3600)

            # Open stream to peer using adapter
            stream = await self._host_adapter.new_stream(
                peer_info_obj.peer_id, ["/p2p/1.0.0"])

            self._connections[peer_id] = stream
            self.stats["connections_established"] += 1

            logger.info(f"Connected to peer {peer_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to peer {peer_id}: {e}")
            return False

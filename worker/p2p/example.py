#!/usr/bin/env python3
"""
P2P module integration test script

This script will:
1. Start a simple HTTP discovery service
2. Start two P2P nodes (use Mock mode if libp2p is unavailable)
3. Test peer discovery and message communication
4. Test disconnection and reconnection
"""

import asyncio
import time
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Optional
import threading
import http.server
import socketserver
from urllib.parse import urlparse, parse_qs

# Add the worker directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import P2P modules
try:
    from p2p import P2PConfig, P2PNode, P2PMessage, MessageType
    from p2p.discovery import NodeInfo
    P2P_AVAILABLE = True
except ImportError as e:
    logger.error(f"P2P module import failed: {e}")
    P2P_AVAILABLE = False
    sys.exit(1)

# Import discovery server FastAPI app if available
try:
    from p2p.discovery_server import FASTAPI_AVAILABLE as DISCOVERY_FASTAPI_AVAILABLE
    if DISCOVERY_FASTAPI_AVAILABLE:
        from p2p.discovery_server import app as discovery_app
    else:
        discovery_app = None
except Exception as e:
    logger.error(f"Failed to import discovery server components: {e}")
    DISCOVERY_FASTAPI_AVAILABLE = False
    discovery_app = None


class DiscoveryServerRunner:
    """Run the FastAPI-based discovery server from discovery_server.py using uvicorn."""

    def __init__(self, host: str = '127.0.0.1', port: int = 8080):
        self.host = host
        self.port = port
        self._server = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the FastAPI discovery server in a background thread."""
        if not DISCOVERY_FASTAPI_AVAILABLE or discovery_app is None:
            raise RuntimeError(
                "FastAPI discovery server is not available. Install fastapi and uvicorn.")
        try:
            import uvicorn
            config = uvicorn.Config(discovery_app, host=self.host,
                                    port=self.port, log_level="info")
            self._server = uvicorn.Server(config)

            # Run the server in a daemon thread
            self._thread = threading.Thread(
                target=self._server.run,
                daemon=True
            )
            self._thread.start()

            # Give the server a moment to start
            time.sleep(1)
            logger.info(
                f"FastAPI discovery server started on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start FastAPI discovery server: {e}")
            raise

    def stop(self):
        """Signal the server to stop and wait briefly."""
        try:
            if self._server is not None:
                self._server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=3)
            logger.info("Discovery server stopped")
        except Exception as e:
            logger.error(f"Error stopping discovery server: {e}")


class P2PTestNode:
    """P2P test node wrapper"""

    def __init__(self, node_id: str, discovery_url: str):
        self.node_id = node_id
        self.discovery_url = discovery_url
        self.node: Optional[P2PNode] = None
        self.received_messages = []

    async def start(self):
        """Start the node"""
        # Use different fixed ports for each node to ensure connectivity
        port_map = {"worker_1": 40001, "worker_2": 40002}
        listen_port = port_map.get(self.node_id, 0)

        config = P2PConfig(
            node_id=self.node_id,
            listen_port=listen_port,
            listen_host="127.0.0.1",
            discovery_url=self.discovery_url,
            discovery_interval_sec=5,
            max_connections=10,
            connection_timeout_sec=5,
            message_buffer_size=1024,
            heartbeat_interval_sec=10,
            bootstrap_peers=[]
        )

        self.node = P2PNode(config)

        # Register message handlers
        def handle_task_request(message: P2PMessage) -> Optional[P2PMessage]:
            logger.info(
                f"[{self.node_id}] Received task request: {message.data}")
            self.received_messages.append(message)

            # Return response
            return P2PMessage(
                message_type=MessageType.TASK_RESPONSE,
                sender_id=self.node_id,
                recipient_id=message.sender_id,
                timestamp=time.time(),
                data={
                    "result": f"Task processed by {self.node_id}",
                    "original_task": message.data
                }
            )

        def handle_task_response(message: P2PMessage) -> None:
            logger.info(
                f"[{self.node_id}] Received task response: {message.data}")
            self.received_messages.append(message)
            return None

        def handle_custom_message(message: P2PMessage) -> None:
            logger.info(
                f"[{self.node_id}] Received custom message: {message.data}")
            self.received_messages.append(message)
            return None

        self.node.register_message_handler(
            MessageType.TASK_REQUEST, handle_task_request)
        self.node.register_message_handler(
            MessageType.TASK_RESPONSE, handle_task_response)
        self.node.register_message_handler(
            MessageType.CUSTOM, handle_custom_message)

        try:
            await self.node.start()
            logger.info(f"[{self.node_id}] Node started successfully")
        except Exception as e:
            logger.error(f"[{self.node_id}] Failed to start node: {e}")
            raise

    async def stop(self):
        """Stop the node"""
        if self.node:
            await self.node.stop()
            logger.info(f"[{self.node_id}] Node stopped")

    async def send_task_request(self, target_node_id: str, task_data: dict) -> bool:
        """Send a task request"""
        if not self.node:
            return False

        message = P2PMessage(
            message_type=MessageType.TASK_REQUEST,
            sender_id=self.node_id,
            recipient_id=target_node_id,
            timestamp=time.time(),
            data=task_data
        )

        return await self.node.send_message(message, target_node_id)

    async def broadcast_message(self, message_data: dict) -> bool:
        """Broadcast a message"""
        if not self.node:
            return False

        message = P2PMessage(
            message_type=MessageType.CUSTOM,
            sender_id=self.node_id,
            recipient_id=None,  # broadcast
            timestamp=time.time(),
            data=message_data
        )

        return await self.node.send_message(message)

    async def ping_peer(self, peer_id: str) -> bool:
        """Ping other nodes"""
        if not self.node:
            return False
        return await self.node.ping_peer(peer_id)

    def get_stats(self) -> dict:
        """Get node statistics"""
        if not self.node:
            return {}
        return self.node.get_stats()


async def run_p2p_test():
    """Run P2P test"""
    logger.info("=== Starting P2P module integration test ===")

    # 1. Start discovery service
    logger.info("1. Start discovery service...")
    # use a different port to avoid conflicts and start the FastAPI-based server
    discovery_runner = DiscoveryServerRunner(host='127.0.0.1', port=8081)
    discovery_runner.start()

    # Wait for service to start
    await asyncio.sleep(1)

    # 2. Create and start two nodes
    logger.info("2. Create and start P2P nodes...")
    node1 = P2PTestNode("worker_1", "http://127.0.0.1:8081")
    node2 = P2PTestNode("worker_2", "http://127.0.0.1:8081")

    try:
        await node1.start()
        await node2.start()

        # Wait for nodes to register to discovery service
        logger.info("3. Wait for peer discovery...")
        await asyncio.sleep(5)  # increase wait time

        # 3. Test peer discovery
        logger.info("4. Test peer discovery...")
        if node1.node and node2.node:
            peers1 = await node1.node.get_discovered_peers()
            peers2 = await node2.node.get_discovered_peers()

            logger.info(
                f"Node1 discovered peers: {[p.node_id for p in peers1]}")
            logger.info(
                f"Node2 discovered peers: {[p.node_id for p in peers2]}")

            # Show detailed peer info
            for peer in peers1:
                logger.info(
                    f"Node1 discovered peer details: {peer.node_id} at {peer.multiaddr}")
            for peer in peers2:
                logger.info(
                    f"Node2 discovered peer details: {peer.node_id} at {peer.multiaddr}")

        # 4. Test ping communication
        logger.info("5. Test ping communication...")
        ping_result = await node1.ping_peer("worker_2")
        logger.info(f"Node1 ping Node2 result: {ping_result}")

        # Wait a moment
        await asyncio.sleep(2)  # increase wait time

        # 5. Test task request/response
        logger.info("6. Test task request/response...")
        task_data = {
            "task_type": "data_processing",
            "input_data": [1, 2, 3, 4, 5],
            "parameters": {"batch_size": 32}
        }

        send_result = await node1.send_task_request("worker_2", task_data)
        logger.info(f"Task send result: {send_result}")

        # Wait for message processing
        await asyncio.sleep(2)

        # 6. Test broadcast message
        logger.info("7. Test broadcast message...")
        broadcast_data = {
            "announcement": "System maintenance notice",
            "scheduled_time": time.time() + 3600,
            "sender": "worker_1"
        }

        broadcast_result = await node1.broadcast_message(broadcast_data)
        logger.info(f"Broadcast result: {broadcast_result}")

        # Wait for message processing
        await asyncio.sleep(2)

        # 7. Show statistics
        logger.info("8. Show node statistics...")
        stats1 = node1.get_stats()
        stats2 = node2.get_stats()

        logger.info(f"Node1 stats: {stats1}")
        logger.info(f"Node2 stats: {stats2}")

        # 8. Show received messages
        logger.info("9. Show received messages...")
        logger.info(
            f"Node1 received message count: {len(node1.received_messages)}")
        logger.info(
            f"Node2 received message count: {len(node2.received_messages)}")

        for i, msg in enumerate(node2.received_messages):
            logger.info(
                f"Node2 message {i+1}: {msg.message_type.value} from {msg.sender_id}")

        # 9. Test disconnect and reconnect
        logger.info("10. Test disconnection...")
        await node2.stop()

        # Wait a moment
        await asyncio.sleep(2)

        # Try sending a message (should fail)
        logger.info("11. Test sending a message to a disconnected node...")
        failed_send = await node1.send_task_request("worker_2", {"test": "reconnect"})
        logger.info(
            f"Result of sending message to disconnected node: {failed_send}")

        # Restart node2
        logger.info("12. Restart Node2...")
        node2 = P2PTestNode("worker_2", "http://127.0.0.1:8081")
        await node2.start()

        # Wait for re-discovery
        await asyncio.sleep(3)

        # Test communication after reconnection
        logger.info("13. Test communication after reconnection...")
        reconnect_result = await node1.send_task_request("worker_2", {"test": "after_reconnect"})
        logger.info(f"Result after reconnection: {reconnect_result}")

        # Wait for processing
        await asyncio.sleep(2)

        # Final stats
        logger.info("14. Final statistics...")
        final_stats1 = node1.get_stats()
        final_stats2 = node2.get_stats()

        logger.info(f"Node1 final stats: {final_stats1}")
        logger.info(f"Node2 final stats: {final_stats2}")

        logger.info("=== P2P test completed ===")

    except Exception as e:
        logger.error(f"Error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        await node1.stop()
        await node2.stop()
        discovery_runner.stop()


def main():
    """Main function"""
    if not P2P_AVAILABLE:
        logger.error("P2P module not available, please check imports")
        return

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        logger.info("Interrupt signal received, exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        asyncio.run(run_p2p_test())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

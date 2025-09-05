"""
vLLM Engine Integration

This module provides integration with vLLM for efficient LLM inference.
"""

from typing import List, Dict, Any, Optional
import asyncio

import structlog


logger = structlog.get_logger(__name__)


class vLLMEngine:
    """vLLM inference engine wrapper"""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9
    ):
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.engine = None
        self.logger = logger.bind(component="vllm_engine")
    
    async def initialize(self) -> None:
        """Initialize vLLM engine"""
        try:
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs
            
            self.logger.info("Initializing vLLM engine", model_path=self.model_path)
            
            # Create engine arguments
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                trust_remote_code=True
            )
            
            # Create engine
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            
            self.logger.info("vLLM engine initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize vLLM engine", error=str(e))
            raise
    
    async def generate(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """Generate text using vLLM"""
        
        if not self.engine:
            raise RuntimeError("vLLM engine not initialized")
        
        try:
            from vllm import SamplingParams
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            # Generate responses
            results = []
            for prompt in prompts:
                request_id = f"req_{len(results)}"
                
                # Add request to engine
                self.engine.add_request(
                    request_id=request_id,
                    prompt=prompt,
                    sampling_params=sampling_params
                )
                
                # Wait for completion
                async for request_output in self.engine.generate():
                    if request_output.request_id == request_id:
                        if request_output.finished:
                            output_text = request_output.outputs[0].text
                            results.append(output_text)
                            break
            
            return results
            
        except Exception as e:
            self.logger.error("Generation failed", error=str(e))
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        **kwargs
    ):
        """Generate text with streaming using vLLM"""
        
        if not self.engine:
            raise RuntimeError("vLLM engine not initialized")
        
        try:
            from vllm import SamplingParams
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                **kwargs
            )
            
            request_id = "streaming_req"
            
            # Add request to engine
            self.engine.add_request(
                request_id=request_id,
                prompt=prompt,
                sampling_params=sampling_params
            )
            
            # Stream results
            async for request_output in self.engine.generate():
                if request_output.request_id == request_id:
                    yield request_output.outputs[0].text
                    if request_output.finished:
                        break
                        
        except Exception as e:
            self.logger.error("Streaming generation failed", error=str(e))
            raise
    
    async def shutdown(self) -> None:
        """Shutdown vLLM engine"""
        if self.engine:
            # vLLM doesn't have an explicit shutdown method
            # The engine will be cleaned up when the object is destroyed
            self.engine = None
            self.logger.info("vLLM engine shutdown")


class vLLMServer:
    """vLLM server for serving models via HTTP API"""
    
    def __init__(
        self,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 8000,
        tensor_parallel_size: int = 1
    ):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.tensor_parallel_size = tensor_parallel_size
        self.process = None
        self.logger = logger.bind(component="vllm_server")
    
    async def start(self) -> None:
        """Start vLLM server"""
        try:
            import subprocess
            
            self.logger.info("Starting vLLM server", 
                           model_path=self.model_path,
                           host=self.host,
                           port=self.port)
            
            # Build command
            cmd = [
                "python", "-m", "vllm.entrypoints.api_server",
                "--model", self.model_path,
                "--host", self.host,
                "--port", str(self.port),
                "--tensor-parallel-size", str(self.tensor_parallel_size),
                "--trust-remote-code"
            ]
            
            # Start server process
            self.process = subprocess.Popen(cmd)
            
            # Wait for server to start
            await asyncio.sleep(10)
            
            self.logger.info("vLLM server started successfully")
            
        except Exception as e:
            self.logger.error("Failed to start vLLM server", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop vLLM server"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            self.logger.info("vLLM server stopped")
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self.process is not None and self.process.poll() is None
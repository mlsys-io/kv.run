#Edited by: Junyi Shen

import requests
from typing import Dict, Optional
from text_generation.errors import parse_error
import ast
from PIL import Image
from io import BytesIO
import base64
import argparse
import time

class Request:
    def __init__(self, inputs: str, stream: bool, parameters: Dict):
        self.inputs = inputs
        self.stream = stream
        self.parameters = parameters
        
    def dict(self):
        return {
            "inputs": self.inputs,
            "stream": self.stream,
            "parameters": self.parameters,
        }
class Client_diffsuion:
    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 20,
    ):
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.timeout = timeout
        
    def generate(
        self,
        prompt: str,
        images_per_prompt: int = 1,
        inference_steps: int = 10,
        image_input: Optional[str] = None,
        image_strength: Optional[float] = None,
    ):
        parameters = {
            "images_per_prompt": images_per_prompt,
            "inference_steps": inference_steps,
            "image_input": image_input,
            "image_strength": image_strength,
        }
        
        request = Request(
            inputs=prompt, 
            stream=False, 
            parameters=parameters,
        )
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=request.dict(),
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
        )
        
        if response.status_code == 404:
            raise parse_error(
                response.status_code,
                {"error": "Service not found.", "errory_type": "generation"},
            )
            
        payload = response.json()
        if response.status_code != 200:
            raise parse_error(response.status_code, payload)
        
        image_str = payload["generated_text"]
        images = ast.literal_eval(image_str)
        for image in images:
            photo = Image.open(BytesIO(base64.b64decode(image)))
            photo.save(f"image_{images.index(image)}_{time.time()}.png")
        return 
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--url", type=str, default="http://127.0.0.1:3000")
    argparser.add_argument("--prompt", type=str, default="A painting of two people.")
    argparser.add_argument("--input_image", type=str, default=None)
    args = argparser.parse_args()
    
    client = Client_diffsuion(args.url)
    input_image = args.input_image
    
    if input_image is not None:
        image = Image.open(input_image)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        base64_bytes = base64.b64encode(img_bytes)
        byte_str = str(base64_bytes, "utf-8")
        
    client.generate(
        prompt=args.prompt, 
        images_per_prompt=1, 
        inference_steps=10,
        image_input=byte_str if input_image else None,
        image_strength=0.5 if input_image else None,
        )
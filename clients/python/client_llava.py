#Edited by: Junyi Shen

import requests
from typing import Dict, Optional, List
from text_generation.types import Parameters, Grammar, Response
from text_generation.errors import parse_error
import base64
import argparse

class Client_Llava:
    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 10,
    ):
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.timeout = timeout
        
    def generate(
        self,
        prompt: str,
        input_image: Optional[str] = None,
        do_sample: bool = False,
        max_new_tokens: int = 20,
        best_of: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        return_full_text: bool = False,
        seed: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: bool = False,
        decoder_input_details: bool = False,
        top_n_tokens: Optional[int] = None,
        grammar: Optional[Grammar] = None,
        lora_id: Optional[str] = None,
    ) -> Response:
        
        parameters = {
            "best_of": best_of if best_of is not None else 1,
            "details": True,
            "do_sample": do_sample,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "frequency_penalty": frequency_penalty,
            "return_full_text": return_full_text,
            "seed": seed,
            "stop": stop_sequences if stop_sequences is not None else [],
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "truncate": truncate,
            "typical_p": typical_p,
            "watermark": watermark,
            "decoder_input_details": decoder_input_details,
            "top_n_tokens": top_n_tokens,
            "grammar": grammar,
            "lora_id": lora_id,
        }
        
        with open(input_image, "rb") as f:
            image = base64.b64encode(f.read()).decode("utf-8")
            image = f"data:image/png;base64,{image}"
            
        prompt = f"![]({image}){prompt}\n\n"
        
        request = {
            "inputs": prompt,
            "stream": False,
            "parameters": parameters,
        }
        
        request = {
            "inputs": prompt,
            "parameters": parameters,
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=request,
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
        print(payload)
        return Response(**payload)
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--url", type=str, default="http://127.0.0.1:3000")
    argparser.add_argument("--prompt", type=str, default="What is in the picture?")
    argparser.add_argument("--input_image", type=str, default="server/examples/images/4.png")
    args = argparser.parse_args()
    
    client = Client_Llava(args.url)
        
    response = client.generate(
        prompt=args.prompt, 
        input_image=args.input_image,
        )
    
    print(response.generated_text)
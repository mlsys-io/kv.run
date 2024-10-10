from typing import Optional
from dataclasses import dataclass
import torch
from huggingface_hub import hf_hub_download
import json
from text_generation_server.models_diffuser.stable_diffusion import Stable_Diffusion_Model
from text_generation_server.models_diffuser.stable_diffusion_3 import Stable_Diffusion_3_Model
from text_generation_server.models_diffuser.stable_diffusion_3_i2i import Stable_Diffusion_3_i2i_Model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)

@dataclass
class Stbale_Diffusion_Request():
    id: int
    prompt: str
    negative_prompt: str = ""
    num_images_per_prompt: int = 1
    num_inference_steps: int = 50
    input_image: Optional[bytes] = None
    strength: Optional[float] = 0.6
    output_type: Optional[str] = "pil"
    output: list[bytes] = None
    nsfw: Optional[list[bool]] = False
    
def get_model(
    model_id: str,
    dtype: Optional[str] = "fp16",
    ):
    try:
        model_id, function = model_id.split(":")
    except ValueError:
        function = None
    model_index_path = hf_hub_download(repo_id=model_id, filename="model_index.json")

    with open(model_index_path, "r") as f:
        model_index = json.load(f)
        model_type = model_index['_class_name']
        
    if model_type == "StableDiffusionPipeline":
        return Stable_Diffusion_Model(
            model_id=model_id
        )
    elif model_type == "StableDiffusion3Pipeline":
        if function == "i2i":
            return Stable_Diffusion_3_i2i_Model(
                model_id=model_id,
                dtype=dtype
            )
        else:  
            return Stable_Diffusion_3_Model(
                model_id=model_id,
                dtype=dtype
            )
     
    else:
        raise ValueError(f"Unknown model type: {model_type}")
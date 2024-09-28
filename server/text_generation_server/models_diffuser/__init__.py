from PIL import Image
from typing import Optional
from dataclasses import dataclass
@dataclass
class Stbale_Diffusion_Request():
    id: str
    prompt: str
    negative_prompt: str = ""
    num_images_per_prompt: int = 1
    num_inference_steps: int = 50
    input_image: Optional[bytes] = None
    strength: Optional[float] = 0.6
    output_type: str = "pil"
    output: list[bytes] = None
    nsfw: Optional[bool] = False
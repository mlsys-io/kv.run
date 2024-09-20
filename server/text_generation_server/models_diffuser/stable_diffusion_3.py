# Modified from Diffusers Repo: diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3
# Editor: Junyi Shen

import torch
import time
from dataclasses import dataclass
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
    retrieve_timesteps,
    )
from PIL import Image
from typing import Optional, List, Union

@dataclass
class Stbale_Diffusion_Request():
    id: str
    prompt: str
    negative_prompt: str
    num_images_per_prompt: int = 1
    output_type: str = "pil"
    output: Image.Image = None
    nsfw: bool = False

@dataclass
class StableDiffusion3Batch():
    requests: List[Stbale_Diffusion_Request]
    stage: str
    prompts: list[str]
    negative_prompts: list[str]
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1
    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    num_inference_steps: int = 28
    timesteps: List[int] = None
    latents: Optional[torch.Tensor] = None
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    t: List[int] = None
    
    @classmethod
    def from_pb(cls, requests):
        prompts = []
        negative_prompts = []
        for request in requests:
            prompts.append(request.prompt)
            negative_prompts.append(request.negative_prompt)
        return cls(requests, "prefill", prompts, negative_prompts)
    
    

class Stable_Diffusion_3_Model:
    def __init__(
        self, 
        model_path,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype = torch.float16,
        ) -> None:
        self.model = StableDiffusion3Pipeline.from_pretrained(model_path, dtype=dtype)
        
        self.device = device
        self.model.to(device)
        self.model._guidance_scale = 7.0
        self.model._clip_skip = None
        self.model._joint_attention_kwargs = None
        self.model._interrupt = False
        
    @torch.no_grad()
    def generate_token(
        self, batch: StableDiffusion3Batch,
    ):  
        if batch.stage == "prefill":
            batch, time = self.prefill(batch)
            batch.stage = "sample"
            
        batch, time = self.sample(batch)
        if batch.t >= len(batch.timesteps):
            for (request,latent) in zip(batch.requests, batch.latents):
                image = self.decode(request, latent.unsqueeze(0), dtype = torch.float16)
                request.output = image[0]
            return None, time, batch.requests
        return batch, time, None
    
    def prefill(self, batch: StableDiffusion3Batch):
        s_time = time.time()
        
        prompt = batch.prompts
        prompt_2 = batch.prompt_2
        prompt_3 = batch.prompt_3
        negative_prompt = batch.negative_prompts
        negative_prompt_2 = batch.negative_prompt_2
        negative_prompt_3 = batch.negative_prompt_3
        
        num_images_per_prompt = batch.num_images_per_prompt
        prompt_embeds = batch.prompt_embeds if batch.prompt_embeds is not None else None
        negative_prompt_embeds = batch.negative_prompt_embeds if batch.negative_prompt_embeds is not None else None
        pooled_prompt_embeds = batch.pooled_prompt_embeds if batch.pooled_prompt_embeds is not None else None
        negative_pooled_prompt_embeds = batch.negative_pooled_prompt_embeds if batch.negative_pooled_prompt_embeds is not None else None
        
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        (
            prompt_embeds, 
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.model.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.model.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=self.device,
            clip_skip=self.model.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
        )
        
        if self.model.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        
        timesteps, num_inference_steps = retrieve_timesteps(
            self.model.scheduler, batch.num_inference_steps, self.device, batch.timesteps,
        )
        self.model._num_timesteps = len(timesteps)
        
        num_channels_latents = self.model.transformer.config.in_channels
        height = batch.height or self.model.default_sample_size * self.model.vae_scale_factor
        width = batch.width or self.model.default_sample_size * self.model.vae_scale_factor
        latents = self.model.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            batch.generator,
            batch.latents,
        )
        
        batch.latents = latents
        batch.prompt_embeds = prompt_embeds
        batch.negative_prompt_embeds = negative_prompt_embeds
        batch.pooled_prompt_embeds = pooled_prompt_embeds
        batch.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        batch.timesteps = timesteps
        batch.num_inference_steps = num_inference_steps
        batch.t = 0

        return batch,(s_time, time.time() - s_time)
            
    def sample(self, batch: StableDiffusion3Batch):
        s_time = time.time()
        
        latents = batch.latents
        t = batch.timesteps[batch.t]
        
        latent_model_input = torch.cat([latents] * 2) if self.model.do_classifier_free_guidance else latents
        timestep = t.expand(latent_model_input.shape[0])  
              
        noise_pred = self.model.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=batch.prompt_embeds,
            pooled_projections=batch.pooled_prompt_embeds,
            joint_attention_kwargs=self.model.joint_attention_kwargs,
            return_dict=False,
            )[0]

        if self.model.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.model.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
        # compute the previous noisy sample x_t -> x_t-1
        latents_dtype = latents.dtype
        latents = self.model.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        if latents.dtype != latents_dtype:
            if torch.backends.mps.is_available():
            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                latents = latents.to(latents_dtype)

        batch.latents = latents
        batch.t += 1
        return batch, (s_time, time.time() - s_time)    

    
    def decode(
        self, 
        request: Stbale_Diffusion_Request, 
        latent,
        dtype = torch.float16,
        ):
            output_type = request.output_type
            if output_type == "latent":
                image = latent
            else:
                latent = (latent / self.model.vae.config.scaling_factor) + self.model.vae.config.shift_factor
                image = self.model.vae.decode(latent, return_dict=False)[0]
                image = self.model.image_processor.postprocess(image, output_type=output_type)
            return image
        
if __name__ == "__main__":
    server = Stable_Diffusion_3_Model("stabilityai/stable-diffusion-3-medium-diffusers")
    #server.model('A dog').images[0].save("test0.png")
    req = Stbale_Diffusion_Request(0, "A dog", "")
    req2 = Stbale_Diffusion_Request(1, "A cat", "")
    batch = StableDiffusion3Batch.from_pb([req,req2])
    while batch is not None:
        batch, t, requests = server.generate_token(batch)
    print(requests[0].output)
    requests[0].output.save("test.png")
    
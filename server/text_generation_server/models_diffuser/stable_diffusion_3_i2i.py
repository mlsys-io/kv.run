# Modified from Diffusers Repo: diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3
# Editor: Junyi Shen

import torch
import time
from dataclasses import dataclass
from server.text_generation_server.models_diffuser.models.pipeline_sd3i2i import StableDiffusion3Img2ImgPipeline
from PIL import Image
from typing import Optional, List, Union
from server.text_generation_server.models_diffuser.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler


@dataclass
class Stbale_Diffusion_Request():
    id: str
    prompt: str
    negative_prompt: str = ""
    num_images_per_prompt: int = 1
    num_inference_steps: int = 50
    input_image: Optional[Image.Image] = None
    strength: float = 0.6
    output_type: str = "pil"
    output: Image.Image = None
    nsfw: Optional[bool] = False

@dataclass
class StableDiffusion3ImageBatch():
    requests: List[Stbale_Diffusion_Request]
    stage: str
    prompts: list[str]
    negative_prompts: list[str]
    num_images_per_prompt: list[int]
    num_inference_steps: list[int] 
    input_images: List[Image.Image] = None,
    strength: List[float] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    num_inference_steps: int = 28
    timesteps: List[int] = None
    latents: Optional[torch.Tensor] = None
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    height: Optional[int] = 512
    width: Optional[int] = 512
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    
    counter: List[int] = None
    sigmas: List[torch.Tensor] = None
    
    @classmethod
    def from_pb(cls, requests):
        prompts = []
        negative_prompts = []
        num_images_per_prompt = []
        num_inference_steps = []
        input_images = []
        strength = []
        for request in requests:
            prompts.append(request.prompt)
            negative_prompts.append(request.negative_prompt)
            num_images_per_prompt.append(request.num_images_per_prompt)
            for i in range(request.num_images_per_prompt):
                num_inference_steps.append(request.num_inference_steps)
                input_images.append(request.input_image)
                strength.append(request.strength)
        return cls(requests, "prefill", prompts, negative_prompts, num_images_per_prompt, num_inference_steps, input_images, strength)
    

class Stable_Diffusion_3_i2i_Model:
    def __init__(
        self, 
        model_path,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype = torch.float16,
        ) -> None:
        self.model = StableDiffusion3Img2ImgPipeline.from_pretrained(model_path, dtype=dtype)
        
        self.device = device
        self.model.to(device)
        self.model._guidance_scale = 7.0
        self.model._clip_skip = None
        self.model._interrupt = False
        self.model.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.model.scheduler.config)

    @torch.no_grad()
    def generate_token(
        self, batch: StableDiffusion3ImageBatch,
    ):  
        if batch.stage == "prefill":
            batch, time = self.prefill(batch)
            batch.stage = "sample"
            
        batch, time = self.sample(batch)
            
        Stop = False
        i = 0
        j = 0
        finished = []
        latents = []
        propmt_embeds = []
        negative_prompt_embeds = []
        pool_prompt_embeds = []
        pool_negative_prompt_embeds = []
        for r in range(len(batch.requests)):
            j += batch.num_images_per_prompt[r]
            if batch.counter[i] >= batch.num_inference_steps[i]:
                request = batch.requests[r]
                image = self.decode(request, batch.latents[i:j], dtype = torch.float16)
                request.output = image
                finished.append(request)
                
                if Stop is not True:
                    Stop = True
                    requests = batch.requests[:r]
                    num_images_per_prompt = batch.num_images_per_prompt[:r]
                    num_inference_steps = batch.num_inference_steps[:r]
                    timesteps = batch.timesteps[:i]
                    counter = batch.counter[:i]
                    sigmas = batch.sigmas[:i]
                    latents.append(batch.latents[:i])
                    propmt_embeds.append(batch.prompt_embeds[:i])
                    negative_prompt_embeds.append(batch.negative_prompt_embeds[:i])
                    pool_prompt_embeds.append(batch.pooled_prompt_embeds[:i])
                    pool_negative_prompt_embeds.append(batch.negative_pooled_prompt_embeds[:i])
                i = j
            else:
                if not Stop:
                    i = j
                    continue
                else:
                    requests.append(batch.requests[r])
                    num_images_per_prompt.append(batch.num_images_per_prompt[r])
                    num_inference_steps += batch.num_inference_steps[i:j]
                    counter += batch.counter[i:j]
                    timesteps += batch.timesteps[i:j]
                    sigmas += batch.sigmas[i:j]
                    latents.append(batch.latents[i:j])
                    propmt_embeds.append(batch.prompt_embeds[i:j])
                    negative_prompt_embeds.append(batch.negative_prompt_embeds[i:j])
                    pool_prompt_embeds.append(batch.pooled_prompt_embeds[i:j])
                    pool_negative_prompt_embeds.append(batch.negative_pooled_prompt_embeds[i:j])
                    i = j
                        
        if Stop:
            if len(counter) > 0:
                batch.requests = requests
                batch.num_images_per_prompt = num_images_per_prompt
                batch.num_inference_steps = num_inference_steps
                batch.counter = counter
                batch.sigmas = sigmas
                batch.timesteps = timesteps
                batch.latents = torch.cat(latents)
                batch.prompt_embeds = torch.cat(propmt_embeds * 2, dim=0)
                batch.negative_prompt_embeds = torch.cat(negative_prompt_embeds * 2, dim=0)
                batch.pooled_prompt_embeds = torch.cat(pool_prompt_embeds * 2, dim=0)
                batch.negative_pooled_prompt_embeds = torch.cat(pool_negative_prompt_embeds * 2, dim=0)
            else:
                batch = None
        
        return batch, time, finished
    
    def prefill(self, batch: StableDiffusion3ImageBatch):
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
        
        batch_size = sum(num_images_per_prompt)
            
        (prompt_embeds, negative_prompt_embeds,pooled_prompt_embeds,negative_pooled_prompt_embeds,) = self.model.encode_prompt(
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
            
        images = self.model.image_processor.preprocess(batch.input_images, height=batch.height, width=batch.width)
        
        timesteps = []
        sigmas = []
        begin_index = []
        for i in range(batch_size):
            _timesteps = batch.timesteps[i] if batch.timesteps is not None else None
            _timesteps, _sigmas = self.model.scheduler.set_timesteps(
                num_inference_steps=batch.num_inference_steps[i],
                device=self.device,
                )
            _, _,_begin_index = self.model.get_timesteps(_timesteps, batch.num_inference_steps[i], batch.strength[i], self.device)
            sigmas.append(_sigmas)
            timesteps.append(_timesteps)
            begin_index.append(_begin_index)
        batch.sigmas = sigmas
        batch.timesteps = timesteps
        batch.counter = begin_index

        #latent_timestep = torch.tensor([t[0] for t in timesteps])
        if batch.latents is None:
            latents = self.model.prepare_latents(
                images,
                begin_index,
                batch_size,
                sigmas,
                prompt_embeds.dtype,
                self.device,
                batch.generator,
            )
        
        batch.latents = latents
        batch.prompt_embeds = prompt_embeds
        batch.negative_prompt_embeds = negative_prompt_embeds
        batch.pooled_prompt_embeds = pooled_prompt_embeds
        batch.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds

        return batch,(s_time, time.time() - s_time)
            
    def sample(self, batch: StableDiffusion3ImageBatch):
        s_time = time.time()
        batch_size = len(batch.counter)
        latents = batch.latents
        
        t = [batch.timesteps[i][batch.counter[i]].unsqueeze(0) for i in range(batch_size)]
        t_input = torch.cat(t*2) if self.model.do_classifier_free_guidance else torch.tensor(t, device=self.device)
        latent_model_input = torch.cat([latents] * 2) if self.model.do_classifier_free_guidance else latents
              
        noise_pred = self.model.transformer(
            hidden_states=latent_model_input,
            timestep=t_input,
            encoder_hidden_states=batch.prompt_embeds,
            pooled_projections=batch.pooled_prompt_embeds,
            return_dict=False,
            )[0]

        if self.model.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.model.guidance_scale * (noise_pred_text - noise_pred_uncond)
            
        # compute the previous noisy sample x_t -> x_t-1
        latents_dtype = latents.dtype
        new_latents = []
        for i in range(batch_size):
            prev_sample = self.model.scheduler.step(
                model_output = noise_pred[i].unsqueeze(0), 
                timestep = t[i], 
                sample = latents[i].unsqueeze(0), 
                counter = batch.counter[i],
                sigmas = batch.sigmas[i],
                return_dict=False)[0]
            
            batch.counter[i] += 1
            new_latents.append(prev_sample)
        
        latents = torch.cat(new_latents)
        
        if latents.dtype != latents_dtype:
            if torch.backends.mps.is_available():
            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                latents = latents.to(latents_dtype)

        batch.latents = latents
        return batch, (s_time, time.time() - s_time)    

    
    def decode(
        self, 
        request: Stbale_Diffusion_Request, 
        latent,
        dtype = torch.float16,
        ):
            latent = latent.unsqueeze(0) if len(latent.shape) == 3 else latent
            output_type = request.output_type
            if output_type == "latent":
                image = latent
            else:
                latent = (latent / self.model.vae.config.scaling_factor) + self.model.vae.config.shift_factor
                image = self.model.vae.decode(latent, return_dict=False)[0]
                image = self.model.image_processor.postprocess(image, output_type=output_type)
            return image
        
if __name__ == "__main__":
    server = Stable_Diffusion_3_i2i_Model("stabilityai/stable-diffusion-3-medium-diffusers")
    req = Stbale_Diffusion_Request(0, "painting", "", 1, 50, Image.open("server/examples/images/0.png"))
    req2 = Stbale_Diffusion_Request(1, "scientific style", "", 1, 50, Image.open("server/examples/images/0.png"))
    batch = StableDiffusion3ImageBatch.from_pb([req,req2])
    while batch is not None:
        batch, t, requests = server.generate_token(batch)
        if requests is not None:
            for request in requests:
                print(request.output, request.nsfw)
                for i, pic in enumerate(request.output):
                    pic.save(f"test_{request.id}_{i}.png")
    
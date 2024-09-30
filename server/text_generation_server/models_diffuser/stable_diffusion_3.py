# Modified from Diffusers Repo: diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3
# Editor: Junyi Shen

import torch
import time
from dataclasses import dataclass
from text_generation_server.models_diffuser.models.pipeline_sd3 import StableDiffusion3Pipeline
from PIL import Image
from typing import Optional, List, Union
from text_generation_server.models_diffuser.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
import io, base64
from text_generation_server.pb import generate_pb2
from opentelemetry import trace
tracer = trace.get_tracer(__name__)
from text_generation_server.models.types import (
    Batch,
    Tokens,
    Generation,
    GeneratedText,
)

@dataclass
class StableDiffusion3Batch():
    batch_id: int
    requests: List[generate_pb2.DiffusionRequest]
    stage: str
    prompts: list[str]
    negative_prompts: list[str]
    num_images_per_prompt: list[int]
    num_inference_steps: list[int] 
    prompts_2: Optional[Union[str, List[str]]] = None,
    prompts_3: Optional[Union[str, List[str]]] = None,
    negative_prompts_2: Optional[Union[str, List[str]]] = None,
    negative_prompts_3: Optional[Union[str, List[str]]] = None,
    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    timesteps: List[int] = None
    latents: Optional[torch.Tensor] = None
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    
    counter: List[int] = None
    sigmas: List[torch.Tensor] = None
    
    def to_pb(self) -> generate_pb2.CachedBatch:
        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[r.id for r in self.requests],
            size=len(self),
            max_tokens=128, #ramdom number
        )
        
    @classmethod
    def from_pb(cls, pb: generate_pb2.DiffusionBatch) -> "StableDiffusion3Batch":
        requests = pb.requests
        prompts = []
        negative_prompts = []
        num_images_per_prompt = []
        num_inference_steps = []
        for request in requests:
            prompts.append(request.prompt)
            negative_prompts.append(request.negative_prompt)
            num_images_per_prompt.append(request.num_images_per_prompt)
            for i in range(request.num_images_per_prompt):
                num_inference_steps.append(request.num_inference_steps)
        return cls(pb.id, requests, "prefill", prompts, negative_prompts, num_images_per_prompt, num_inference_steps)
    
    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]):
        return self
    
    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["StableDiffusion3Batch"]) -> "StableDiffusion3Batch":
        requests = []
        prompts = []
        negative_prompts = []
        num_images_per_prompt = []
        num_inference_steps = []
        prompts_2 = []
        prompts_3 = []
        negative_prompts_2 = []
        negative_prompts_3 = []
        prompt_embeds = []
        negative_prompt_embeds = []
        time_steps = []
        latents = []
        generator = []
        height = []
        width = []
        pooled_prompt_embeds = []
        negative_pooled_prompt_embeds = []
        counter = []
        sigmas = []
        
        for batch in batches:
            requests += batch.requests
            prompts += batch.prompts
            negative_prompts += batch.negative_prompts
            num_images_per_prompt += batch.num_images_per_prompt
            num_inference_steps += batch.num_inference_steps
            prompts_2 += batch.prompts_2
            prompts_3 += batch.prompts_3
            negative_prompts_2 += batch.negative_prompts_2
            negative_prompts_3 += batch.negative_prompts_3
            prompt_embeds.append(batch.prompt_embeds)
            negative_prompt_embeds.append(batch.negative_prompt_embeds)
            time_steps += batch.timesteps
            latents.append(batch.latents)
            pooled_prompt_embeds.append(batch.pooled_prompt_embeds)
            negative_pooled_prompt_embeds.append(batch.negative_pooled_prompt_embeds)
            counter += batch.counter
            sigmas += batch.sigmas
            
        return cls(
            batch_id = batches[0].batch_id,
            requests = requests, 
            stage = "sample", 
            prompts = prompts, 
            negative_prompts = negative_prompts, 
            num_images_per_prompt = num_images_per_prompt, 
            num_inference_steps = num_inference_steps, 
            prompts_2 = prompts_2,
            prompts_3 = prompts_3,
            negative_prompts_2 = negative_prompts_2,
            negative_prompts_3 = negative_prompts_3,
            prompt_embeds = torch.cat(prompt_embeds, dim=0),
            negative_prompt_embeds = torch.cat(negative_prompt_embeds, dim=0),
            time_steps = time_steps,
            latents = torch.cat(latents, dim=0),
            pooled_prompt_embeds = torch.cat(pooled_prompt_embeds, dim=0),
            negative_pooled_prompt_embeds = torch.cat(negative_pooled_prompt_embeds, dim=0),
            counter = counter,
            sigmas = sigmas
            )

    def __len__(self):
        return len(self.requests)     

class Stable_Diffusion_3_Model:
    def __init__(
        self, 
        model_id: str,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype = torch.float16,
        ) -> None:
        self.model = StableDiffusion3Pipeline.from_pretrained(model_id, dtype=dtype)
        
        self.device = device
        self.model.to(device)
        self.model._guidance_scale = 7.0
        self.model._clip_skip = None
        self.model._joint_attention_kwargs = None
        self.model._interrupt = False
        self.model.scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.model.scheduler.config)
  
    @torch.no_grad()
    def generate_token(
        self, batch: StableDiffusion3Batch,
    ):  
        if batch.stage == "prefill":
            batch, timing = self.prefill(batch)
            batch.stage = "sample"
        else:   
            batch, timing = self.sample(batch)
        
        s_time = time.time()
        Stop = False
        i = 0
        j = 0
        latents = []
        propmt_embeds = []
        negative_prompt_embeds = []
        pool_prompt_embeds = []
        pool_negative_prompt_embeds = []
        generations = []
        
        for r in range(len(batch.requests)):
            j += batch.num_images_per_prompt[r]
            if batch.counter[i] >= batch.num_inference_steps[i]:
                image = self.decode(batch.latents[i:j], dtype = torch.float16)
                generated_text = GeneratedText(
                    text = str(image),
                    generated_tokens = False,
                    finish_reason = 0,
                    seed = None,
                    )
                
                if not Stop:
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
            else:
                generated_text = None
                if Stop:
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
            generation = Generation(
                request_id = batch.requests[r].id,
                prefill_tokens = None,
                tokens = None,
                generated_text = generated_text,
                top_tokens=None,
                )
            generations.append(generation)
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
        
        return generations, batch, (timing, time.time() - s_time)
    
    def prefill(self, batch: StableDiffusion3Batch):
        s_time = time.time()
        
        prompt = batch.prompts
        prompt_2 = batch.prompts_2
        prompt_3 = batch.prompts_3
        negative_prompt = batch.negative_prompts
        negative_prompt_2 = batch.negative_prompts_2
        negative_prompt_3 = batch.negative_prompts_3
        
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
        
        timesteps = []
        sigmas = []
        for i in range(batch_size):
            _timesteps = batch.timesteps[i] if batch.timesteps is not None else None
            _timesteps, _sigmas = self.model.scheduler.set_timesteps(
                num_inference_steps=batch.num_inference_steps[i],
                device=self.device,
                )
            sigmas.append(_sigmas)
            timesteps.append(_timesteps)
        batch.sigmas = sigmas
        batch.timesteps = timesteps
                
        num_channels_latents = self.model.transformer.config.in_channels
        height = batch.height or self.model.default_sample_size * self.model.vae_scale_factor
        width = batch.width or self.model.default_sample_size * self.model.vae_scale_factor
        latents = self.model.prepare_latents(
            batch_size,
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
        batch.counter = [0] * batch_size

        return batch,(s_time, time.time() - s_time)
            
    def sample(self, batch: StableDiffusion3Batch):
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
            joint_attention_kwargs=self.model.joint_attention_kwargs,
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
        latent,
        output_type = "pil",
        dtype = torch.float16,
        ):
            latent = latent.unsqueeze(0) if len(latent.shape) == 3 else latent
            if output_type == "latent":
                image = latent
            else:
                latent = (latent / self.model.vae.config.scaling_factor) + self.model.vae.config.shift_factor
                image = self.model.vae.decode(latent, return_dict=False)[0]
                image = self.model.image_processor.postprocess(image, output_type=output_type)
                # transfer into bytes
                images = []
                for _image in image:
                    buffered = io.BytesIO()
                    _image.save(buffered, format="PNG")
                    img_bytes = base64.b64encode(buffered.getvalue())
                    #_image.save(f"test_{time.time()}.png")
                    images.append(img_bytes)
                return images
            return image
        
    def warmup(self, batch: StableDiffusion3Batch):
        batch, _ = self.prefill(batch)
        batch, _ = self.sample(batch)
        return batch
        
if __name__ == "__main__":
    server = Stable_Diffusion_3_Model("stabilityai/stable-diffusion-3-medium-diffusers")
    req = generate_pb2.DiffusionRequest(
        id=0,
        prompt="A cat",
        negative_prompt="",
        num_images_per_prompt=1,
        num_inference_steps=10
        )
    req2 = generate_pb2.DiffusionRequest(
        id=1,
        prompt="A dog",
        negative_prompt="",
        num_images_per_prompt=1,
        num_inference_steps=15
        )
    batch = generate_pb2.DiffusionBatch(requests=[req,req2])
    batch = StableDiffusion3Batch.from_pb(batch)
    while batch is not None:
        generations, batch, t = server.generate_token(batch)
        for generation in generations:
            if generation.generated_text is not None:
                print(generation.generated_text.text)
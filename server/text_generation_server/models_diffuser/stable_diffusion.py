# Modified from Diffusers Repo: diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion
# Editor: Junyi Shen

import torch
import time
from dataclasses import dataclass
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    retrieve_timesteps,
    rescale_noise_cfg,
    )
from PIL import Image
from typing import Optional, List, Union
from text_generation_server.models_diffuser.schedulers.scheduling_pndm import PNDMScheduler
import base64, io
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
class StableDiffusionBatch:
    batch_id: int
    requests: List[generate_pb2.Request]
    stage: str
    prompts: list[str]
    negative_prompts: list[str]
    num_images_per_prompt: list[int]
    num_inference_steps: list[int] 
    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    ip_adapter_image: Optional[Image.Image] = None
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None
    timesteps: List[int] = None
    sigmas: List[float] = None
    latents: Optional[torch.Tensor] = None
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    height: Optional[int] = None
    width: Optional[int] = None
    eta: float = 0.0
    
    extra_step_kwargs: Optional[dict] = None
    added_cond_kwargs: Optional[dict] = None
    timestep_cond: Optional[List[int]] = None
    counter: List[int] = None
    ets: list[list[torch.Tensor]] = None
    cur_sample: list[torch.Tensor] = None
    
    def to_pb(self) -> generate_pb2.CachedBatch:
        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[r.id for r in self.requests],
            size=len(self),
            max_tokens=1024, #ramdom number
        )
            
    @classmethod
    def from_pb(cls, pb: generate_pb2.Batch) -> "StableDiffusionBatch":
        requests = pb.requests
        prompts = []
        negative_prompts = []
        num_images_per_prompt = []
        num_inference_steps = []
        for request in requests:
            prompts.append(request.inputs)
            negative_prompts.append("")
            num_images_per_prompt.append(request.images_per_prompt)
            for i in range(request.images_per_prompt):
                num_inference_steps.append(request.inference_steps)
        return cls(pb.id, requests, "prefill", prompts, negative_prompts, num_images_per_prompt, num_inference_steps)
    
    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]):
        return self
    
    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["StableDiffusionBatch"]) -> "StableDiffusionBatch":
        requests = []
        prompts = []
        negative_prompts = []
        num_images_per_prompt = []
        num_inference_steps = []
        prompt_embeds = []
        negative_prompt_embeds = []
        time_steps = []
        latents = []
        counter = []
        ets = []
        cur_sample = []
        
        for batch in batches:
            requests += batch.requests
            prompts += batch.prompts
            negative_prompts += batch.negative_prompts
            num_images_per_prompt += batch.num_images_per_prompt
            num_inference_steps += batch.num_inference_steps
            prompt_embeds.append(batch.prompt_embeds)
            negative_prompt_embeds.append(batch.negative_prompt_embeds)
            time_steps += batch.timesteps
            latents.append(batch.latents)
            counter += batch.counter
            ets += batch.ets
            cur_sample += batch.cur_sample
        
        return cls(
            batch_id = batches[0].batch_id,
            requests = requests, 
            stage = "sample", 
            prompts = prompts, 
            negative_prompts = negative_prompts, 
            num_images_per_prompt = num_images_per_prompt, 
            num_inference_steps = num_inference_steps, 
            prompt_embeds = torch.cat(prompt_embeds, dim=0),
            negative_prompt_embeds = torch.cat(negative_prompt_embeds, dim=0),
            time_steps = time_steps,
            latents = torch.cat(latents, dim=0),
            counter = counter,
            ets = ets,
            cur_sample = cur_sample
            )

    def __len__(self):
        return len(self.requests)     
    
class Stable_Diffusion_Model:
    def __init__(
        self, 
        model_id: str,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype = torch.float16,
        ) -> None:
        self.model = StableDiffusionPipeline.from_pretrained(model_id)
        self.dtype = dtype
        self.device = device
        self.model.to(device)
        self.model._guidance_scale = 7.5
        self.model._guidance_rescale = 0.0
        self.model._clip_skip = None
        self.model._cross_attention_kwargs = None
        self.model._interrupt = False
        
        self.model.unet.get_time_embed = self.get_time_embed
        self.model.scheduler = PNDMScheduler.from_config(self.model.scheduler.config)
    
    @property
    def info(self) -> generate_pb2.InfoResponse:
        return generate_pb2.InfoResponse(
            requires_padding=False,
            dtype=str(self.dtype),
            device_type=self.device.type,
            window_size=None,
            speculate=None,
        )
        
    @property 
    def batch_type(self):
        return StableDiffusionBatch
    
    def get_time_embed(
        self, sample: torch.Tensor, timestep: Union[torch.Tensor, float, int]
    ) -> Optional[torch.Tensor]:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(sample.shape[0])
        else:
            timesteps = torch.cat([timesteps] * 2) if self.model.do_classifier_free_guidance else timesteps
        timesteps = timesteps.to(sample.device)
        t_emb = self.model.unet.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        return t_emb
    
    @torch.no_grad()
    def generate_token(
        self, batch: StableDiffusionBatch,
    ):  
        if batch.stage == "prefill":
            batch, timing = self.prefill(batch)
            batch.stage = "sample"
        else:
            batch, timing = self.sample(batch)
            
        s_time = time.time_ns()
        Stop = False
        i = 0
        j = 0
        latents = []
        propmt_embeds = []
        negative_prompt_embeds = []
        generations = []
        for r in range(len(batch.requests)):
            j += batch.num_images_per_prompt[r]
            if batch.counter[i] >= batch.num_inference_steps[i]:
                image, nsfw = self.decode(batch.latents[i:j], dtype = torch.float16)
                generated_text = GeneratedText(
                    text = str(image),
                    generated_tokens = batch.num_inference_steps[i],
                    finish_reason = 1,
                    seed = None,
                    )
                
                if not Stop:
                    Stop = True
                    requests = batch.requests[:r]
                    num_images_per_prompt = batch.num_images_per_prompt[:r]
                    num_inference_steps = batch.num_inference_steps[:r]
                    timesteps = batch.timesteps[:i]
                    counter = batch.counter[:i]
                    latents.append(batch.latents[:i])
                    propmt_embeds.append(batch.prompt_embeds[:i])
                    negative_prompt_embeds.append(batch.negative_prompt_embeds[:i])
                    ets = batch.ets[:i]
                    cur_sample = batch.cur_sample[:i]
            else:
                generated_text = None
                if Stop:
                    requests.append(batch.requests[r])
                    num_images_per_prompt.append(batch.num_images_per_prompt[r])
                    num_inference_steps += batch.num_inference_steps[i:j]
                    counter += batch.counter[i:j]
                    timesteps += batch.timesteps[i:j]
                    latents.append(batch.latents[i:j])
                    propmt_embeds.append(batch.prompt_embeds[i:j])
                    negative_prompt_embeds.append(batch.negative_prompt_embeds[i:j])
                    ets += batch.ets[i:j]
                    cur_sample += batch.cur_sample[i:j]
                    
            generation = Generation(
                request_id = batch.requests[r].id,
                prefill_tokens = None,
                tokens = Tokens(
                    token_ids = [0],
                    logprobs = [1.0],
                    texts = ['-'],
                    is_special = [True],
                    ),
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
                batch.timesteps = timesteps
                batch.latents = torch.cat(latents)
                batch.prompt_embeds = torch.cat(propmt_embeds * 2, dim=0)
                batch.negative_prompt_embeds = torch.cat(negative_prompt_embeds * 2, dim=0)
                batch.ets = ets
                batch.cur_sample = cur_sample
            else:
                batch = None
        
        return generations, batch, (timing, time.time_ns() - s_time)
    
    def prefill(self, batch: StableDiffusionBatch):
        s_time = time.time_ns()
        prompts = batch.prompts
        negative_prompts = batch.negative_prompts
        num_images_per_prompt = batch.num_images_per_prompt

        prompt_embeds = batch.prompt_embeds if batch.prompt_embeds is not None else None
        negative_prompt_embeds = batch.negative_prompt_embeds if batch.negative_prompt_embeds is not None else None
        lora_scale = (
            self.model.cross_attention_kwargs.get("scale", None) if self.model.cross_attention_kwargs is not None else None
        )
        
        batch_size = sum(batch.num_images_per_prompt)
            
        prompt_embeds, negative_prompt_embeds = self.model.encode_prompt(
            prompts,
            self.device,
            1, # Expand later
            self.model.do_classifier_free_guidance,
            negative_prompts,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.model.clip_skip,
        )
        
        # The following can be merge into the encode_prompt function
        chunks_prompt_embeds = torch.chunk(prompt_embeds, len(prompts))
        chunks_negative_prompt_embeds = torch.chunk(negative_prompt_embeds, len(negative_prompts))
        prompt_embeds = torch.cat([chunks_prompt_embeds[i].repeat(num_images_per_prompt[i], 1, 1) for i in range(len(prompts))])
        negative_prompt_embeds = torch.cat([chunks_negative_prompt_embeds[i].repeat(num_images_per_prompt[i], 1, 1) for i in range(len(negative_prompts))])
        
        if self.model.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            
        ip_adapter_image = batch.ip_adapter_image
        ip_adapter_image_embeds = batch.ip_adapter_image_embeds
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.model.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                self.device,
                batch_size,
                self.model.do_classifier_free_guidance,
            )
            
        timesteps = []
        num_inference_steps = []
        for i in range(batch_size):
            _timesteps = batch.timesteps[i] if batch.timesteps is not None else None
            sigma = batch.sigmas[i] if batch.sigmas is not None else None
            _timesteps, _num_inference_steps = retrieve_timesteps(
                self.model.scheduler, batch.num_inference_steps[i], self.device, _timesteps, sigma
            )
            timesteps.append(_timesteps)
            num_inference_steps.append(_num_inference_steps)
        batch.timesteps = timesteps
        batch.num_inference_steps = num_inference_steps   
        
        num_channels_latents = self.model.unet.config.in_channels
        height = batch.height or self.model.unet.config.sample_size * self.model.vae_scale_factor
        width = batch.width or self.model.unet.config.sample_size * self.model.vae_scale_factor
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
        
        extra_step_kwargs = self.model.prepare_extra_step_kwargs(batch.generator, batch.eta)
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )
        timestep_cond = None
        if self.model.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.model.guidance_scale - 1).repeat(batch_size)
            timestep_cond = self.model.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.model.unet.config.time_cond_proj_dim
            ).to(device=self.device, dtype=latents.dtype)
        
        batch.latents = latents
        batch.prompt_embeds = prompt_embeds
        batch.negative_prompt_embeds = negative_prompt_embeds
        batch.extra_step_kwargs = extra_step_kwargs
        batch.added_cond_kwargs = added_cond_kwargs
        batch.timestep_cond = timestep_cond
        
        batch.counter = [0] * batch_size
        batch.ets = [[]] * batch_size
        batch.cur_sample = [None] * batch_size
        
        return batch, time.time_ns() - s_time
            
        
    def sample(self, batch: StableDiffusionBatch):
        s_time = time.time_ns()
        batch_size = len(batch.counter)
        latents = batch.latents

        t = torch.cat([batch.timesteps[i][batch.counter[i]].unsqueeze(0) for i in range(batch_size)])
        latent_model_input = torch.cat([latents] * 2) if self.model.do_classifier_free_guidance else latents
        latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)
        noise_pred = self.model.unet(
            latent_model_input,
            t,
            encoder_hidden_states=batch.prompt_embeds,
            timestep_cond=batch.timestep_cond,
            cross_attention_kwargs=self.model.cross_attention_kwargs,
            added_cond_kwargs=batch.added_cond_kwargs,
            return_dict=False,
            )[0]

        if self.model.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.model.guidance_scale * (noise_pred_text - noise_pred_uncond)

        if self.model.do_classifier_free_guidance and self.model.guidance_rescale > 0.0:
            # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
            noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.model.guidance_rescale)

        # compute the previous noisy sample x_t -> x_t-1
        new_latents = []
        for i in range(batch_size):
            (prev_sample, ets, cur_sample) = self.model.scheduler.step(
                model_output = noise_pred[i].unsqueeze(0), 
                timestep = t[i], 
                sample = latents[i].unsqueeze(0), 
                counter = batch.counter[i],
                num_inference_steps = batch.num_inference_steps[i],
                ets = batch.ets[i],
                cur_sample = batch.cur_sample[i],
                **batch.extra_step_kwargs, 
                return_dict=False)
            
            batch.ets[i] = ets
            batch.cur_sample[i] = cur_sample
            batch.counter[i] += 1
            new_latents.append(prev_sample)
        
        batch.latents = torch.cat(new_latents)
        return batch, time.time_ns() - s_time

    
    def decode(
        self, 
        latent,
        output_type = "pil",
        dtype = torch.float16,
        ):
            latent = latent.unsqueeze(0) if len(latent.shape) == 3 else latent
            if not output_type == "latent":
                image = self.model.vae.decode(latent / self.model.vae.config.scaling_factor, return_dict=False, generator=None)[0]
                image, has_nsfw_concept = self.model.run_safety_checker(image, self.device, dtype)
            else:
                image = latent
                has_nsfw_concept = None
            if has_nsfw_concept is None:
                do_denormalize = [True] * image.shape[0]
            else:
                do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

            image = self.model.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
            # transfer into bytes
            images = []
            for _image in image:
                buffered = io.BytesIO()
                _image.save(buffered, format="PNG")
                img_bytes = base64.b64encode(buffered.getvalue())
                #_image.save(f"test_{time.time()}.png")
                images.append(img_bytes)
            return images, has_nsfw_concept
    
    def warmup(self, batch: StableDiffusionBatch):
        batch, _ = self.prefill(batch)
        return 102400
        
if __name__ == "__main__":
    server = Stable_Diffusion_Model("CompVis/stable-diffusion-v1-4")
    req = generate_pb2.Request(
        id=0,
        inputs="A cat",
        images_per_prompt=1,
        inference_steps=40
        )
    req2 = generate_pb2.Request(
        id=1,
        inputs="A dog",
        images_per_prompt=1,
        inference_steps=50
        )
    batch = generate_pb2.Batch(requests=[req,req2])
    batch = StableDiffusionBatch.from_pb(batch)
    while batch is not None:
        generations, batch, t = server.generate_token(batch)
        for generation in generations:
            if generation.generated_text is not None:
                print(generation.generated_text.text)
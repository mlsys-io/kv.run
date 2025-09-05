"""
Supervised Fine-Tuning (SFT) Module

This module implements supervised fine-tuning using the TRL library.
It supports various adapter methods like LoRA, QLoRA, etc.
"""

import os
from pathlib import Path
from typing import Callable, Optional

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer

from mloc.common.constants import AdapterType
from .base_module import BaseModule


class SFTModule(BaseModule):
    """Supervised Fine-Tuning module using TRL"""
    
    async def execute(
        self,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> None:
        """Execute SFT training"""
        
        self.logger.info("Starting SFT training")
        
        try:
            # Step 1: Validate configuration and prepare environment
            self._validate_config()
            self._prepare_environment()
            self._log_system_info()
            
            await self._report_progress(0.05, progress_callback)
            
            # Step 2: Load model and tokenizer
            model, tokenizer = await self._load_model_and_tokenizer()
            await self._report_progress(0.15, progress_callback)
            
            # Step 3: Load and prepare dataset
            train_dataset = await self._load_and_prepare_dataset(tokenizer)
            await self._report_progress(0.25, progress_callback)
            
            # Step 4: Setup training arguments
            training_args = self._setup_training_arguments()
            await self._report_progress(0.30, progress_callback)
            
            # Step 5: Create trainer and start training
            trainer = await self._create_trainer(
                model, tokenizer, train_dataset, training_args, progress_callback
            )
            
            await self._report_progress(0.35, progress_callback)
            
            # Step 6: Train the model (this will take most of the time)
            await self._train_model(trainer, progress_callback)
            
            # Step 7: Save model and artifacts
            await self._save_model_and_artifacts(trainer)
            await self._report_progress(1.0, progress_callback)
            
            self.logger.info("SFT training completed successfully")
            
        except Exception as e:
            self.logger.error("SFT training failed", error=str(e))
            raise
    
    async def _load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        self.logger.info("Loading model and tokenizer")
        
        model_path = str(self.model_dir)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Apply adapter if specified
        if self.config.spec.model.adapter:
            model = self._apply_adapter(model)
        
        self.logger.info(
            "Model and tokenizer loaded",
            model_type=type(model).__name__,
            vocab_size=len(tokenizer),
            pad_token=tokenizer.pad_token
        )
        
        return model, tokenizer
    
    def _apply_adapter(self, model):
        """Apply adapter (LoRA, QLoRA, etc.) to the model"""
        adapter_config = self.config.spec.model.adapter
        
        self.logger.info("Applying adapter", adapter_type=adapter_config.type)
        
        if adapter_config.type in [AdapterType.LORA, AdapterType.QLORA]:
            # Configure LoRA
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=adapter_config.r,
                lora_alpha=adapter_config.lora_alpha,
                lora_dropout=adapter_config.lora_dropout,
                target_modules=adapter_config.target_modules,
                bias=adapter_config.bias
            )
            
            model = get_peft_model(model, peft_config)
            
            # Print trainable parameters
            model.print_trainable_parameters()
            
        else:
            raise ValueError(f"Unsupported adapter type: {adapter_config.type}")
        
        return model
    
    async def _load_and_prepare_dataset(self, tokenizer):
        """Load and prepare training dataset"""
        self.logger.info("Loading and preparing dataset")
        
        # For now, assume dataset is already downloaded and processed
        # In a full implementation, this would handle different dataset formats
        dataset_path = self.dataset_dir
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset directory not found: {dataset_path}")
        
        # Load dataset
        dataset = load_from_disk(str(dataset_path))
        
        # Apply preprocessing if specified
        if self.config.spec.dataset and self.config.spec.dataset.preprocessing:
            dataset = self._preprocess_dataset(dataset, tokenizer)
        
        self.logger.info(
            "Dataset loaded",
            num_samples=len(dataset),
            features=list(dataset.features.keys())
        )
        
        return dataset
    
    def _preprocess_dataset(self, dataset, tokenizer):
        """Preprocess dataset with tokenization"""
        preprocessing = self.config.spec.dataset.preprocessing
        
        def tokenize_function(examples):
            # This is a simple example - would need to be customized
            # based on the dataset format and prompt template
            
            if preprocessing.prompt_template:
                # Apply prompt template
                texts = [
                    preprocessing.prompt_template.format(**example)
                    for example in examples
                ]
            else:
                # Use text field directly
                texts = examples.get("text", examples.get("input", []))
            
            return tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=preprocessing.max_seq_length,
                return_overflowing_tokens=False,
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        return tokenized_dataset
    
    def _setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments from configuration"""
        hyperparams = self.config.spec.hyperparameters
        
        if not hyperparams:
            raise ValueError("Training hyperparameters not specified")
        
        # Create output directory
        output_dir = self.artifacts_dir / "checkpoints"
        output_dir.mkdir(exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=hyperparams.num_train_epochs,
            per_device_train_batch_size=hyperparams.per_device_train_batch_size,
            per_device_eval_batch_size=hyperparams.per_device_eval_batch_size,
            gradient_accumulation_steps=hyperparams.gradient_accumulation_steps,
            learning_rate=hyperparams.learning_rate,
            weight_decay=hyperparams.weight_decay,
            warmup_steps=hyperparams.warmup_steps,
            logging_dir=str(self.artifacts_dir / "logs"),
            logging_steps=hyperparams.logging_steps,
            eval_steps=hyperparams.eval_steps,
            save_steps=hyperparams.save_steps,
            save_total_limit=hyperparams.save_total_limit,
            fp16=hyperparams.fp16,
            bf16=hyperparams.bf16,
            gradient_checkpointing=hyperparams.gradient_checkpointing,
            dataloader_pin_memory=hyperparams.dataloader_pin_memory,
            remove_unused_columns=hyperparams.remove_unused_columns,
            report_to=[],  # Disable wandb, tensorboard for now
        )
        
        self.logger.info("Training arguments configured", 
                        epochs=hyperparams.num_train_epochs,
                        batch_size=hyperparams.per_device_train_batch_size,
                        learning_rate=hyperparams.learning_rate)
        
        return training_args
    
    async def _create_trainer(
        self, 
        model, 
        tokenizer, 
        train_dataset, 
        training_args,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> SFTTrainer:
        """Create SFT trainer"""
        
        self.logger.info("Creating SFT trainer")
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            max_seq_length=self.config.spec.dataset.preprocessing.max_seq_length if 
                          self.config.spec.dataset and self.config.spec.dataset.preprocessing 
                          else 2048,
        )
        
        # Add custom callback for progress reporting
        if progress_callback:
            trainer.add_callback(ProgressCallback(progress_callback))
        
        return trainer
    
    async def _train_model(
        self,
        trainer: SFTTrainer,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> None:
        """Train the model"""
        
        self.logger.info("Starting model training")
        
        # Start training
        trainer.train()
        
        self.logger.info("Model training completed")
    
    async def _save_model_and_artifacts(self, trainer: SFTTrainer) -> None:
        """Save trained model and artifacts"""
        
        self.logger.info("Saving model and artifacts")
        
        # Save the final model
        final_model_dir = self.artifacts_dir / "final_model"
        trainer.save_model(str(final_model_dir))
        
        # Save tokenizer
        trainer.tokenizer.save_pretrained(str(final_model_dir))
        
        # If using adapter, save adapter weights separately
        if self.config.spec.model.adapter:
            adapter_dir = self.artifacts_dir / "adapter_weights"
            trainer.model.save_pretrained(str(adapter_dir))
        
        # Save training metadata
        metadata = {
            "task_id": self.task_id,
            "task_type": self.config.spec.task_type,
            "model_name": self.config.spec.model.source.identifier,
            "adapter_type": self.config.spec.model.adapter.type if self.config.spec.model.adapter else None,
            "training_epochs": self.config.spec.hyperparameters.num_train_epochs,
            "final_loss": trainer.state.log_history[-1].get("train_loss") if trainer.state.log_history else None,
        }
        
        self._save_metadata(metadata)
        
        self.logger.info("Model and artifacts saved successfully")


class ProgressCallback:
    """Custom callback for progress reporting"""
    
    def __init__(self, progress_callback: Callable[[float], None]):
        self.progress_callback = progress_callback
        self.total_steps = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.total_steps = state.max_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.total_steps:
            # Map training progress (35% - 95% of overall progress)
            train_progress = state.global_step / self.total_steps
            overall_progress = 0.35 + (train_progress * 0.60)
            
            # Run callback in async context
            import asyncio
            if asyncio.iscoroutinefunction(self.progress_callback):
                asyncio.create_task(self.progress_callback(overall_progress))
            else:
                self.progress_callback(overall_progress)
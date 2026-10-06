import os
from typing import Optional

import safetensors.torch
import torch
from transformers import Trainer

from swift.tuner_plugin import Tuner, tuners_map
from swift.tuners import LoraConfig, Swift
from swift.utils import deep_getattr, get_logger, get_multimodal_target_regex
from swift.optimizers import optimizers_map
from swift.optimizers.base import OptimizerCallback

logger = get_logger()


def is_cross_attn_param(parameter_name: str) -> bool:
    """判断是否是 cross_attn 参数"""
    return 'cross_attn' in parameter_name


class CustomTuner(Tuner):
    """Full-parameter training of cross_attn while LoRA training others"""

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        model = Swift.from_pretrained(model, model_id, **kwargs)
        cross_attn_path = os.path.join(model_id, 'cross_attn.safetensors')
        if os.path.exists(cross_attn_path):
            try:
                cross_attn_state_dict = safetensors.torch.load_file(cross_attn_path)
                model.load_state_dict(cross_attn_state_dict, strict=False)
                logger.info(f'Loaded cross_attn weights from {cross_attn_path}')
            except Exception as e:
                logger.warning(f'Failed to load cross_attn weights from {cross_attn_path}: {e}')
        return model

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        if state_dict is None:
            state_dict = {}
            for n, p in model.named_parameters():
                if p.requires_grad:
                    state_dict[n] = p.detach().cpu()
        model.save_pretrained(save_directory, state_dict=state_dict, safe_serialization=safe_serialization, **kwargs)
        # Save cross_attn parameters separately
        cross_attn_state_dict = {k: v for k, v in state_dict.items() if is_cross_attn_param(k)}
        if cross_attn_state_dict:
            cross_attn_path = os.path.join(save_directory, 'cross_attn.safetensors')
            safetensors.torch.save_file(
                cross_attn_state_dict, cross_attn_path, metadata={'format': 'pt'})
            logger.info(f'Saved cross_attn weights to {cross_attn_path}')

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        # model.model_meta.model_arch is already a MultiModelKeys object
        # Get target regex for LoRA
        # Note: cross_attn will be excluded from LoRA and trained with full parameters
        target_regex = get_multimodal_target_regex(
            model, 
            freeze_llm=False, 
            freeze_vit=False, 
            freeze_aligner=False
        )
        # Modify target_regex to exclude cross_attn
        # Add negative lookahead to exclude cross_attn from LoRA targets
        if target_regex:
            # Add exclusion for cross_attn: (?!.*cross_attn)
            # This ensures cross_attn modules won't get LoRA adapters
            if target_regex.startswith('^('):
                target_regex = '^(?!.*cross_attn)' + target_regex[1:]
            elif target_regex.startswith('^'):
                target_regex = '^(?!.*cross_attn)' + target_regex[1:]
        logger.info(f'target_regex (after excluding cross_attn): {target_regex}')
        
        lora_config = LoraConfig(
            task_type='CAUSAL_LM', r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=target_regex)
        model = Swift.prepare_model(model, lora_config)
        
        # Configure training: cross_attn uses full-parameter training, others use LoRA
        # First freeze all parameters
        model.requires_grad_(False)
        # Then enable LoRA parameters
        for n, p in model.named_parameters():
            if 'lora' in n:
                p.requires_grad = True
        # Finally enable cross_attn parameters for full-parameter training
        for n, p in model.named_parameters():
            if is_cross_attn_param(n) and 'lora' not in n:
                p.requires_grad = True
        
        # Print trainable parameters info
        total_params = 0
        trainable_params = 0
        cross_attn_params = 0
        lora_params = 0
        for n, p in model.named_parameters():
            total_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
                if is_cross_attn_param(n):
                    cross_attn_params += p.numel()
                elif 'lora' in n:
                    lora_params += p.numel()
        
        logger.info(f'Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,} '
                    f'({100 * trainable_params / total_params:.4f}%)')
        logger.info(f'  - cross_attn (full-param): {cross_attn_params:,} '
                    f'({100 * cross_attn_params / total_params:.4f}%)')
        logger.info(f'  - LoRA parameters: {lora_params:,} '
                    f'({100 * lora_params / total_params:.4f}%)')
        
        return model


class CustomOptimizerCallback(OptimizerCallback):
    """cross_attn uses full-parameter training, others use LoRA."""

    def create_optimizer(self):
        args = self.args
        model = self.trainer.model
        decay_parameters = set(Trainer.get_decay_parameter_names(None, model))
        
        lora_parameters = []
        cross_attn_parameters = []
        
        for n, p in model.named_parameters():
            if p.requires_grad:
                if is_cross_attn_param(n):
                    cross_attn_parameters.append((n, p))
                elif 'lora' in n:
                    lora_parameters.append((n, p))
        
        logger.info(f'Found {len(lora_parameters)} LoRA parameter groups')
        logger.info(f'Found {len(cross_attn_parameters)} cross_attn parameter groups')
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in lora_parameters if n in decay_parameters],
                'weight_decay': args.weight_decay,
                'lr': args.learning_rate,
            },
            {
                'params': [p for n, p in lora_parameters if n not in decay_parameters],
                'weight_decay': 0.0,
                'lr': args.learning_rate,
            },
            {
                'params': [p for n, p in cross_attn_parameters if n in decay_parameters],
                'weight_decay': args.weight_decay,
                'lr': args.learning_rate,
            },
            {
                'params': [p for n, p in cross_attn_parameters if n not in decay_parameters],
                'weight_decay': 0.0,
                'lr': args.learning_rate,
            },
        ]
        
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
        return optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


tuners_map['custom'] = CustomTuner
optimizers_map['custom'] = CustomOptimizerCallback
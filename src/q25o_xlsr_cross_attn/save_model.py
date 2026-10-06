"""
Build a Qwen2.5-Omni checkpoint with XLS-R and cross-attention modules.

This script is intentionally strict: it verifies XLS-R injection key-by-key and
refuses to save a checkpoint that contains NaN/Inf in the added modules.
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import torch


DEFAULT_BASE_MODEL = "model/Qwen2.5-Omni-7B"
DEFAULT_XLSR_PATH = "model/wav2vec2-xls-r-300m"
DEFAULT_SAVE_PATH = "model/model_stage2_SFA+AFA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Qwen2.5-Omni checkpoint with XLS-R and cross_attn modules."
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL, help="Source Qwen2.5-Omni checkpoint.")
    parser.add_argument("--xlsr_path", default=DEFAULT_XLSR_PATH, help="Source Wav2Vec2/XLS-R checkpoint.")
    parser.add_argument("--save_path", default=DEFAULT_SAVE_PATH, help="Output checkpoint directory.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow saving into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--reset-xlsr-layer-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset thinker.xlsr.weights to ones before saving.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def ensure_output_dir(save_path: Path, overwrite: bool) -> None:
    if save_path.exists() and not save_path.is_dir():
        raise FileExistsError(f"{save_path} exists and is not a directory.")
    if save_path.exists() and any(save_path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{save_path} already exists and is not empty. Pass --overwrite to replace/update it."
        )
    save_path.mkdir(parents=True, exist_ok=True)


def resolve_save_path(save_path: Path, overwrite: bool) -> Path:
    if overwrite or not save_path.exists() or not any(save_path.iterdir()):
        return save_path

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fallback = save_path.with_name(f"{save_path.name}_rebuild-{timestamp}")
    print(f"{save_path} already exists and is not empty.")
    print(f"Saving to {fallback} instead. Pass --overwrite to replace the requested directory.")
    return fallback


def assert_module_exists(model: torch.nn.Module) -> None:
    missing = []
    if not hasattr(model, "thinker"):
        missing.append("thinker")
    else:
        if not hasattr(model.thinker, "xlsr"):
            missing.append("thinker.xlsr")
        if not hasattr(model.thinker, "cross_attn"):
            missing.append("thinker.cross_attn")
    if missing:
        raise AttributeError(f"Custom model is missing required module(s): {', '.join(missing)}")


def assert_finite_state(module: torch.nn.Module, module_name: str) -> None:
    bad = []
    with torch.no_grad():
        for name, tensor in module.state_dict().items():
            if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                bad.append(name)
                if len(bad) >= 8:
                    break
    if bad:
        joined = ", ".join(f"{module_name}.{name}" for name in bad)
        raise ValueError(f"Non-finite tensor(s) found before save: {joined}")


def reset_xlsr_layer_weights(model: torch.nn.Module) -> None:
    weights = getattr(model.thinker.xlsr, "weights", None)
    if weights is None:
        raise AttributeError("thinker.xlsr.weights is missing.")
    if weights.ndim != 1 or weights.numel() != model.thinker.xlsr.num_layers:
        raise ValueError(
            "Unexpected thinker.xlsr.weights shape: "
            f"{tuple(weights.shape)}, expected ({model.thinker.xlsr.num_layers},)"
        )
    with torch.no_grad():
        weights.fill_(1.0)
    print("Reset thinker.xlsr.weights to all ones.")


def load_xlsr_into_target(source_model: torch.nn.Module, target_module: torch.nn.Module) -> None:
    source_state = source_model.state_dict()
    target_state = target_module.state_dict()

    source_keys = set(source_state)
    target_keys = set(target_state)
    if source_keys != target_keys:
        missing = sorted(source_keys - target_keys)[:20]
        unexpected = sorted(target_keys - source_keys)[:20]
        raise ValueError(
            "XLS-R state_dict keys do not match target thinker.xlsr.w2v2_model. "
            f"missing_in_target={missing}, unexpected_in_target={unexpected}"
        )

    for key, source_tensor in source_state.items():
        target_tensor = target_state[key]
        if source_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {key}: source={tuple(source_tensor.shape)}, "
                f"target={tuple(target_tensor.shape)}"
            )
        if source_tensor.is_floating_point() and not torch.isfinite(source_tensor).all():
            raise ValueError(f"Source XLS-R tensor is non-finite: {key}")

    target_module.load_state_dict(source_state, strict=True)
    print(f"Loaded {len(source_state)} XLS-R tensors into thinker.xlsr.w2v2_model.")


def verify_xlsr_injection(source_model: torch.nn.Module, target_module: torch.nn.Module) -> None:
    print("\n--- Verifying XLS-R injection ---")
    source_state = source_model.state_dict()
    target_state = target_module.state_dict()
    mismatches = []

    with torch.no_grad():
        for key, source_tensor in source_state.items():
            target_tensor = target_state[key].to(device=source_tensor.device)
            if not torch.equal(source_tensor, target_tensor):
                diff = (source_tensor - target_tensor).abs().max().item()
                mismatches.append((key, diff))
                if len(mismatches) >= 8:
                    break

    if mismatches:
        detail = ", ".join(f"{key}: max_abs_diff={diff}" for key, diff in mismatches)
        raise ValueError(f"XLS-R verification failed: {detail}")

    print("XLS-R injection verified: every tensor matches exactly.")


def copy_optional_files(base_model: Path, save_path: Path) -> None:
    for filename in ["spk_dict.pt", "chat_template.jinja", "chat_template.json"]:
        src = base_model / filename
        dst = save_path / filename
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"Copied {filename}.")


def main() -> None:
    args = parse_args()

    from transformers import Wav2Vec2Model

    from module.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
    from module.processing_qwen2_5_omni import Qwen2_5OmniProcessor

    base_model = Path(args.base_model)
    xlsr_path = Path(args.xlsr_path)
    save_path = resolve_save_path(Path(args.save_path), args.overwrite)
    dtype = resolve_dtype(args.dtype)

    ensure_output_dir(save_path, args.overwrite)

    print(f"Loading Qwen2.5-Omni from {base_model}...")
    processor = Qwen2_5OmniProcessor.from_pretrained(str(base_model), trust_remote_code=True)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        str(base_model),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    assert_module_exists(model)

    print(f"Loading Wav2Vec2-XLS-R from {xlsr_path}...")
    xlsr_model = Wav2Vec2Model.from_pretrained(str(xlsr_path))

    target_module = model.thinker.xlsr.w2v2_model
    target_param = next(target_module.parameters())
    xlsr_model = xlsr_model.to(device=target_param.device, dtype=target_param.dtype)

    load_xlsr_into_target(xlsr_model, target_module)
    verify_xlsr_injection(xlsr_model, target_module)

    if args.reset_xlsr_layer_weights:
        reset_xlsr_layer_weights(model)

    assert_finite_state(model.thinker.xlsr, "thinker.xlsr")
    assert_finite_state(model.thinker.cross_attn, "thinker.cross_attn")

    print(f"Saving model to {save_path}...")
    model.save_pretrained(str(save_path), safe_serialization=True)

    print(f"Saving processor to {save_path}...")
    processor.save_pretrained(str(save_path))
    copy_optional_files(base_model, save_path)

    print("\nAll done. Saved checkpoint passed XLS-R/cross_attn finite checks.")


if __name__ == "__main__":
    main()

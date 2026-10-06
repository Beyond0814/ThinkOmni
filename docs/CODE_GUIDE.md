# ThinkOmni Code Guide

This document explains the implementation in this repository. Run commands from the repository root.

## 1. End-to-end data flow

ThinkOmni uses the following path:

~~~text
JSON annotation
    │
    ├── audio waveform ──> Qwen audio encoder
    │                         │
    │                         ├── semantic audio representation
    │                         └── XLS-R acoustic representation
    │
    ├── spectrogram ────────> vision encoder (Stage 3)
    │
    └── text instruction ───> tokenizer
                                  │
                                  v
                  Thinker + cross-attention fusion
                                  │
                                  v
             reasoning + detection + localization
~~~

The three training stages are cumulative:

| Stage | Code path | Modalities | Main purpose |
|---|---|---|---|
| SFA | train_stage1_SFA.sh | Audio and text | Learn semantic forensic analysis |
| AFA | train_stage2_SFA_AFA.sh | Audio, XLS-R, and text | Add acoustic evidence |
| MFR | train_stage3_SFA_AFA_MFR.sh | Audio, XLS-R, spectrogram, and text | Add spectral-visual evidence |

## 2. Source tree

### src/q25o_xlsr_cross_attn

This directory registers the custom Qwen2.5-Omni-XLSR model with ms-swift.

| File | Responsibility |
|---|---|
| custom_model.py | Model registration, processor loading, template registration, audio encoding, and data collation |
| custom_plugin.py | Custom tuner and optimizer registration |
| save_model.py | XLS-R weight injection and checkpoint verification |
| module/configuration_qwen2_5_omni.py | Configuration fields for the XLS-R branch |
| module/modeling_qwen2_5_omni.py | XLS-R encoder, cross-attention, fusion, and modified Thinker path |
| module/processing_qwen2_5_omni.py | Qwen2.5-Omni processor definitions |
| module/modular_qwen2_5_omni.py | Modular Qwen2.5-Omni model components |
| module/__init__.py | Module package initialization |

The model is registered under:

~~~text
model_type = qwen2_5_omni_xlsr
template   = qwen2_5_omni_xlsr
~~~

This is why the Stage 2 and Stage 3 commands must load custom_model.py.

## 3. custom_model.py

### Model loading

Qwen2_5OmniXlsrLoader performs three operations:

1. Loads Qwen2_5OmniConfig and applies THINKOMNI_XLSR_PATH when it is set.
2. Loads Qwen2_5OmniProcessor and patches the Qwen multimodal utilities.
3. Loads Qwen2_5OmniForConditionalGeneration and registers the custom Thinker functions.

The loader also enables low CPU memory loading and patches visual input embedding behavior for training.

### Model registration

The model architecture is divided into logical groups:

~~~text
language_model : thinker.model, thinker.lm_head
vision_tower   : thinker.xlsr
aligner        : thinker.cross_attn
generator      : audio generation and speech synthesis modules
~~~

These groups are used by ms-swift to decide which modules are frozen, LoRA-adapted, or fully trainable.

### Audio and image tags

Qwen2_5OmniXlsrTemplate maps the generic dataset tags to Qwen special tokens:

~~~text
<audio>  -> <|audio_bos|><|AUDIO|><|audio_eos|>
<image>  -> <|vision_bos|><|IMAGE|><|vision_eos|>
<video>  -> Qwen video tokens
~~~

Audio is loaded at the processor sampling rate. During training, the waveform is aligned to a fixed length of 16000 × 300 samples by truncation or zero padding. This keeps the XLS-R input shape stable across batches.

### Batch collation

The custom collator keeps the raw aligned waveform in the audios field. The model receives both the normal Qwen multimodal inputs and the waveform tensor required by the XLS-R branch.

When padding-free training is enabled, packed text and multimodal inputs are handled by ms-swift while the raw audio batch remains available for the custom model.

## 4. modeling_qwen2_5_omni.py

### XLS-R encoder

Qwen2_5OmniXLSR loads:

~~~text
Wav2Vec2FeatureExtractor
Wav2Vec2Model
~~~

from the configured XLS-R directory. The XLS-R hidden states are kept as acoustic features and passed to the fusion module.

### Cross-attention fusion

Qwen2_5OmniCrossAttn receives:

- Qwen audio features;
- XLS-R hidden states;
- valid audio lengths;
- optional attention masks.

The XLS-R sequence is projected to a lower-rank representation and interpolated to the temporal resolution of the Qwen audio features. It then produces key and value representations for cross-attention. A gated residual path combines the acoustic information with the original Qwen representation.

The implementation checks the resulting packed token count against the Qwen audio token count. A mismatch raises an error instead of silently training with misaligned features.

### Auxiliary forensic modules

The file also defines supporting modules for forensic learning:

- positional encoding for long audio sequences;
- global forgery discrimination;
- local temporal fusion;
- cross-attention projections;
- trainable fusion and gating layers.

These modules preserve both global authenticity evidence and local boundary evidence.

## 5. custom_plugin.py

The custom plugin registers:

~~~text
tuner    = custom
optimizer = custom
~~~

### CustomTuner

The tuner uses a hybrid update strategy:

- LoRA adapters are trained on the selected multimodal linear layers.
- Parameters whose name contains cross_attn are trained in full precision.
- All other parameters are frozen unless they are explicitly selected by the training configuration.

During checkpoint loading, the tuner looks for cross_attn.safetensors and restores those parameters separately.

During saving, it writes the trainable state and additionally stores cross-attention parameters in cross_attn.safetensors. This makes the cross-attention weights easy to inspect and restore.

### CustomOptimizerCallback

The optimizer creates separate parameter groups for:

1. LoRA parameters with weight decay;
2. LoRA parameters without weight decay;
3. cross-attention parameters with weight decay;
4. cross-attention parameters without weight decay.

All groups use the configured learning rate. This separation prevents frozen parameters from entering the optimizer and keeps the cross-attention update explicit.

## 6. save_model.py

The checkpoint builder is the bridge between the base Qwen checkpoint and the custom XLS-R model.

~~~bash
python src/q25o_xlsr_cross_attn/save_model.py \
  --base_model model/Qwen2.5-Omni-7B \
  --xlsr_path model/wav2vec2-xls-r-300m \
  --save_path model/model_stage2_SFA+AFA
~~~

Its validation sequence is:

1. load the Qwen model and processor;
2. confirm that thinker.xlsr and thinker.cross_attn exist;
3. load the official XLS-R checkpoint;
4. compare source and target state-dict keys;
5. compare every tensor shape;
6. copy the XLS-R tensors into thinker.xlsr.w2v2_model;
7. verify exact tensor equality;
8. reset the XLS-R layer weights when requested;
9. check all added tensors for NaN and Inf;
10. save model, processor, and optional speaker metadata.

The script refuses to overwrite a non-empty output directory unless --overwrite is supplied.

## 7. Loss and training scripts

The shell scripts are thin experiment wrappers around ms-swift.

Important arguments include:

| Argument | Role |
|---|---|
| model | Base model or previous-stage checkpoint |
| dataset | Training JSON file |
| model_type | Qwen model registration name |
| template | Dataset template registration name |
| external_plugins | Custom model and plugin files |
| train_type | LoRA or custom tuner |
| loss_type | Detection and localization loss |
| loss_scale | Token-level loss weighting |
| freeze_vit / freeze_aligner | Controls module updates |
| vit_lr / aligner_lr | Separate multimodal learning rates |
| max_length | Maximum text sequence length |

The FCML-related configuration used by the scripts is:

~~~text
loss_type  = psd_detection_localization
loss_scale = psd_think_det_loc
~~~

The scripts determine their own repository root, so relative paths remain valid when the script is launched from another working directory.

## 8. Dataset and output contract

Each annotation must provide the audio path and messages. Stage 3 also provides a spectrogram path.

~~~json
{
  "audios": ["datasets/example/audio.wav"],
  "images": ["datasets/example/spectrogram.png"],
  "messages": [
    {
      "role": "user",
      "content": "<audio><image> Determine whether the speech contains manipulated regions."
    },
    {
      "role": "assistant",
      "content": "Reasoning: ...\nDetection Result: partially fake\nLocalization Result: 0.43-1.36"
    }
  ]
}
~~~

The evaluation parser expects these labels:

~~~text
Detection Result: fully real | fully fake | partially fake
Localization Result: None | start-end, start-end, ...
~~~

Keep the field names, result headings, and interval separator unchanged.

## 9. Evaluation code

script/evaluate.py:

1. reads one JSONL prediction file;
2. extracts the audio path and infers the dataset name;
3. parses detection and localization results;
4. normalizes numeric labels 0, 1, and 2;
5. calculates detection accuracy and weighted metrics;
6. calculates TIoU-based AP and mAP;
7. writes an Excel summary.

Run it with:

~~~bash
python script/evaluate.py path/to/predictions.jsonl
python script/evaluate.py path/to/predictions.jsonl --output results.xlsx
~~~

## 10. Typical debugging order

When an XLS-R run fails, check in this order:

1. The XLS-R directory contains a valid Hugging Face Wav2Vec2 checkpoint.
2. PYTHONPATH includes src/q25o_xlsr_cross_attn and src.
3. custom_model.py is passed through external_plugins or custom_register_path.
4. The model uses model_type qwen2_5_omni_xlsr and template qwen2_5_omni_xlsr.
5. THINKOMNI_XLSR_PATH points to the same XLS-R directory used during checkpoint construction.
6. Every audio and spectrogram path exists.
7. Audio tensor lengths and packed feature lengths are consistent.
8. The checkpoint contains the cross-attention parameters when custom training is resumed.


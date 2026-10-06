from functools import partial
from typing import Any, Dict, List, Literal, Optional

import torch
import numpy as np
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers import PretrainedConfig, PreTrainedModel
from swift.model import (Model, ModelGroup, ModelMeta, MultiModelKeys, get_model_processor, register_model,
                         register_model_arch, ModelLoader)
from swift.model.models.qwen import patch_qwen_vl_utils
from swift.model.patcher import patch_get_input_embeddings
from swift.model.utils import use_submodel_func
from swift.template.template_inputs import StdTemplateInputs
from swift.template.utils import Context, findall
from swift.template.vision_utils import load_audio
from swift.utils import get_env_args, get_logger, is_deepspeed_enabled, Processor, to_float_dtype, get_packed_seq_params
from swift.template import Template, TemplateMeta, get_template, register_template


logger = get_logger()

import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

register_model_arch(
    MultiModelKeys(
        'qwen2_5_omni_xlsr',
        # `freeze_llm`, `freeze_vit`, `freeze_aligner`将根据下面的值来决定其行为。
        # 例如：全参数训练，若设置`freeze_vit=True`，将冻结以`thinker.audio_tower`和`thinker.visual`为前缀的模型层的参数。
        # LoRA训练，若设置`freeze_vit=False`，将额外为以`thinker.audio_tower`和`thinker.visual`为前缀的Linear层添加LoRA。
        language_model=['thinker.model', 'thinker.lm_head'],
        vision_tower=['thinker.xlsr'],
        aligner=['thinker.cross_attn'],
        generator=['thinker.audio_tower', 'thinker.audio_tower.proj', 'thinker.visual', 'thinker.visual.merger', 'talker', 'token2wav'],
        # vision_tower=['thinker.audio_tower', 'thinker.visual', 'thinker.xlsr'],
        # aligner=['thinker.audio_tower.proj', 'thinker.visual.merger', 'thinker.cross_attn'],
        # generator=['talker', 'token2wav'],
        # vision_tower=['thinker.visual'],
        # aligner=['thinker.visual.merger'],
        # generator=['thinker.xlsr', 'thinker.cross_attn', 'thinker.audio_tower', 'thinker.audio_tower.proj', 'talker', 'token2wav'],
        # generator的部分将永远不进行训练或处于冻结状态。
        # 如果你希望`thinker.audio_tower`, `thinker.audio_tower.proj`永远不进行训练，你可以放置到generator中，并将其从vision_tower, aligner中移除。
    ))


class Qwen2_5OmniXlsrLoader(ModelLoader):

    def get_config(self, model_dir: str) -> PretrainedConfig:
        from module.configuration_qwen2_5_omni import Qwen2_5OmniConfig
        config = Qwen2_5OmniConfig.from_pretrained(model_dir, trust_remote_code=True)
        # Checkpoints created on another machine may contain an old absolute
        # XLS-R path. Prefer an explicit environment override, then resolve
        # the repository-relative default from the current checkout.
        xlsr_path = os.getenv('THINKOMNI_XLSR_PATH')
        if xlsr_path:
            config.xlsr_path = xlsr_path
        elif not Path(config.xlsr_path).exists():
            config.xlsr_path = str(Path(__file__).resolve().parents[2] / 'model' / 'wav2vec2-xls-r-300m')
        enable_audio_output = get_env_args('ENABLE_AUDIO_OUTPUT', bool, None)
        if enable_audio_output is not None:
            config.enable_audio_output = enable_audio_output
        return config

    def get_processor(self, model_dir: str, config: PretrainedConfig) -> Processor:
        from module.processing_qwen2_5_omni import Qwen2_5OmniProcessor
        from qwen_omni_utils import vision_process
        processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, trust_remote_code=True)
        # Control constants in qwen_omni_utils library via environment variables,
        # e.g., `MAX_PIXELS`, etc.
        patch_qwen_vl_utils(vision_process)
        return processor

    def get_model(self, model_dir: str, config: PretrainedConfig, processor: Processor,
                  model_kwargs) -> PreTrainedModel:
        from module.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration
        print('Run qwen2_5_omni_xlsr...')
        self.auto_model_cls = self.auto_model_cls or Qwen2_5OmniForConditionalGeneration
        # 降低加载 checkpoint 时的显存尖峰（多模态 Omni + ZeRO-2 单卡放全量权重时易在 shard 合并处 OOM）
        mk = dict(model_kwargs)
        mk.setdefault('low_cpu_mem_usage', True)
        model = super().get_model(model_dir, config, processor, mk)
        # For multimodal model consistency, we replace the model's forward/generate functions
        # with those of its language_model.
        # Handle additional parts separately.
        use_submodel_func(model, 'thinker')
        # Avoid inplace operations on leaf_variable during training
        # (replacing parts of input_embeds with images_embeds)
        patch_get_input_embeddings(model.thinker.visual, 'patch_embed')
        # Some custom settings for model/config (usually not needed; configure based on
        # specific model if errors occur during training/inference)
        model.config.keys_to_ignore_at_inference += ['hidden_states', 'attention_mask']
        model.config.talker_config.pad_token_id = None
        return model

register_model(
    ModelMeta(
        'qwen2_5_omni_xlsr',
        [
            ModelGroup([
                Model('Qwen/Qwen2.5-Omni-3B', 'Qwen/Qwen2.5-Omni-3B'),
                Model('Qwen/Qwen2.5-Omni-7B', 'Qwen/Qwen2.5-Omni-7B'),
            ]),
        ],
        # 用来获取model和processor的函数。
        Qwen2_5OmniXlsrLoader,
        template='qwen2_5_omni_xlsr',
        is_multimodal=True,  # 是否是多模态模型
        model_arch='qwen2_5_omni_xlsr',  # 通常只为多模态模型设置
        # 用于model_type的自动匹配
        architectures=['Qwen2_5OmniModel', 'Qwen2_5OmniForConditionalGeneration'],
        # 用来提示用户依赖版本（可删除）
        requires=['transformers>=4.50', 'soundfile', 'qwen_omni_utils', 'decord'],
        # 用来提示用户（可删除）
        tags=['vision', 'video', 'audio'],
        # 全参数训练/merge-lora需要额外保存的文件
        additional_saved_files=['spk_dict.pt'],
    ))



class Qwen2_5OmniXlsrTemplate(Template):
    use_model = True  # 是否在预处理的过程中需要model参与
    # 需要注意是：并不是所有的多模态模型都能支持padding_free/packing。`transformers`库内的模型通常可以支持
    support_padding_free = True  # 是否支持padding_free和packing（多模态模型）
    norm_bbox = 'none'  # grounding任务使用绝对坐标还是norm1000坐标
    version = 'omni_v2_5'
    # 这里的tokens将不会被裁剪（例如设置`--truncation_strategy left/right`）
    # 并会使用简略方式打印（调用`template.safe_decode`）
    placeholder_tokens = ['<|IMAGE|>', '<|AUDIO|>', '<|VIDEO|>']

    def init_processor(self, processor: Processor) -> None:
        """在初始化processor时，额外初始化所需的一些常量"""
        if processor is None:
            return
        super().init_processor(processor)
        if self.version == 'omni_v2_5':
            from module.processing_qwen2_5_omni import Qwen2_5OmniProcessorKwargs
            default = Qwen2_5OmniProcessorKwargs._defaults
        elif self.version == 'omni_v3':
            from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import Qwen3OmniMoeProcessorKwargs
            default = Qwen3OmniMoeProcessorKwargs._defaults
        self.seconds_per_chunk = default['videos_kwargs']['seconds_per_chunk']
        self.position_id_per_seconds = default['videos_kwargs']['position_id_per_seconds']
        self.use_audio_in_video = get_env_args('use_audio_in_video', bool, False)
        self.sampling_rate = get_env_args('sampling_rate', int, self.processor.feature_extractor.sampling_rate)


    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        """读取多模态数据，并替换通用多模态tag。
        例如：图像tag从`<image>` -> `<|vision_bos|><|IMAGE|><|vision_eos|>`"""
        # 读取多模态数据也可以在`_encode`函数中进行，怎么方便怎么来。
        from qwen_omni_utils import fetch_image, fetch_video
        if media_type == 'image':
            inputs.images[index] = fetch_image({'image': inputs.images[index]})
            return ['<|vision_bos|><|IMAGE|><|vision_eos|>']
        elif media_type == 'audio':
            if self.mode != 'vllm':  # 'vllm'推理场景下不需要处理
                inputs.audios[index] = load_audio(inputs.audios[index], self.sampling_rate)
            return ['<|audio_bos|><|AUDIO|><|audio_eos|>']
        elif media_type == 'video':
            video = inputs.videos[index]
            _video = fetch_video({'video': video})
            if isinstance(_video, torch.Tensor):
                _video = _video.to(torch.uint8)
            inputs.videos[index] = _video
            if self.use_audio_in_video:
                import librosa
                if video.startswith('http://') or video.startswith('https://'):
                    import audioread
                    video = audioread.ffdec.FFmpegAudioFile(video)
                video = librosa.load(video, sr=self.sampling_rate)[0]
                inputs.audios.insert(inputs.audio_idx, (video, 'video'))
                inputs.audio_idx += 1
                return ['<|vision_bos|><|audio_bos|><|VIDEO|><|audio_eos|><|vision_eos|>']
            else:
                return ['<|vision_bos|><|VIDEO|><|vision_eos|>']


    def replace_ref(self, ref: str, index: int, inputs: StdTemplateInputs) -> List[Context]:
        """替换grounding任务的通用tag: `<ref-object>`"""
        if self.bbox_format == 'legacy':
            return [f'<|object_ref_start|>{ref}<|object_ref_end|>']
        else:
            return [ref]

    def replace_bbox(self, bbox: List[int], index: int, inputs: StdTemplateInputs) -> List[Context]:
        """替换grounding任务的通用tag: `<bbox>`"""
        if self.bbox_format == 'legacy':
            return [f'<|box_start|>{self._get_bbox_str(bbox)}<|box_end|>']
        else:
            return [str(bbox)]

    def packing_row(self, row: List[Dict[str, Any]]) -> Dict[str, Any]:
        """支持packing & mrope。通常情况不需要继承该函数，这里为了自定义mrope的position_ids。"""
        position_ids = []
        for r in row:
            r = r.copy()
            r['input_ids'] = torch.tensor(r['input_ids'])[None]
            position_ids.append(self._get_position_ids(r))
        packed = super().packing_row(row)
        packed['position_ids'] = torch.concat(position_ids, dim=-1)
        return packed

    def _get_new_tokens_use_audio_in_video(self, i, *, video_grid_thw, video_second_per_grid, audio_lengths,
                                           video_token_id, audio_token_id):
        """辅助函数，用于支持`use_audio_in_video`为True的情况"""
        merge_size = self.processor.image_processor.merge_size
        grid_thw = video_grid_thw[i]
        height = grid_thw[1] // merge_size
        width = grid_thw[2] // merge_size
        audio_token_indices = torch.arange(audio_lengths[i])
        video_token_indices = torch.arange(grid_thw[0]).reshape(-1, 1, 1)

        video_token_indices = torch.broadcast_to(video_token_indices,
                                                 (video_token_indices.shape[0], height, width)).reshape(-1)
        video_token_indices = (video_token_indices * video_second_per_grid[i] * self.position_id_per_seconds)
        tokens_per_chunk = int(self.position_id_per_seconds * self.seconds_per_chunk)
        video_chunk_indexes = self.processor.get_chunked_index(video_token_indices, tokens_per_chunk)
        audio_chunk_indexes = self.processor.get_chunked_index(audio_token_indices, tokens_per_chunk)

        res = []
        for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
            if j < len(video_chunk_indexes):
                video_seq_length = video_chunk_indexes[j][1] - video_chunk_indexes[j][0]
                res += video_token_id * video_seq_length
            if j < len(audio_chunk_indexes):
                audio_seq_length = audio_chunk_indexes[j][1] - audio_chunk_indexes[j][0]
                res += audio_token_id * audio_seq_length
        return res

    def align_audio(self, audio, target_len=16000*300):
        """
        audio: Tensor | np.ndarray
            shape [T] or [1, T]
        return: Tensor [target_len]
        """
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        if audio.dim() == 2:
            audio = audio.squeeze(0)

        cur_len = audio.size(0)

        if cur_len == target_len:
            return audio

        if cur_len > target_len:
            # 截断
            return audio[:target_len]

        # padding
        pad_len = target_len - cur_len
        return torch.nn.functional.pad(audio, (0, pad_len))

    def _encode(self, inputs: StdTemplateInputs) -> Dict[str, Any]:
        """这里决定如何将text/images/audios/videos -> input_ids、labels、loss_scale以及pixel_values等多模态内容
        这里的处理逻辑通常可以从对应模型的预处理代码实现中借鉴。
        推荐：请先做推理对齐再做训练"""
        encoded = Template._encode(self, inputs)  # 处理纯文本部分，具体请参考自定义模型文档
        logger.info_once('Run qwen2_5_omni template')
        processor = self.processor
        # 获取多模态内容
        media_inputs = processor(
            text='',
            audio=inputs.audios or None,
            images=inputs.images or None,
            videos=inputs.videos or None,
            do_resize=False,
            return_tensors='pt')
        # 我们不使用`processor`产生的input_ids和attention_mask。因为其不产生`labels`。
        media_inputs.pop('input_ids')
        media_inputs.pop('attention_mask')
        media_inputs = to_float_dtype(media_inputs, self.model_info.torch_dtype)

        input_ids = encoded['input_ids']
        labels = encoded['labels']
        loss_scale = encoded.get('loss_scale', None)
        # audio模态
        audio_token_id = self._tokenize('<|AUDIO|>')
        idx_list = findall(input_ids, audio_token_id)  # 查找所有的audio_token
        feature_attention_mask = media_inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            audio_lengths = ((audio_feature_lengths - 1) // 2 + 1 - 2) // 2 + 1
        else:
            audio_lengths = None
        audio_lengths_origin = audio_lengths
        # video_audios_mask用于处理`use_audio_in_video`，区分是纯audio(0)还是video中的audio(1)
        video_audios_mask = []
        audios = []
        for i, audio in enumerate(inputs.audios):
            if isinstance(audio, tuple) and audio[1] == 'video':
                inputs.audios[i] = audio[0]
                video_audios_mask.append(True)
                raw_audio = audio[0]
            else:
                video_audios_mask.append(False)
                raw_audio = audio
            aligned_audio = self.align_audio(raw_audio)
            audios.append(aligned_audio)
        video_audios_mask = torch.tensor(video_audios_mask)
        if idx_list:
            # 过滤掉video中的audio的内容（将在video部分处理）
            if self.use_audio_in_video:
                audio_lengths = audio_lengths[~video_audios_mask]

            def _get_new_audio_tokens(i):
                return audio_token_id * audio_lengths[i]

            # 对input_ids的多模态tokens进行展开
            input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                _get_new_audio_tokens)

        # image和video模态
        for media_type in ['image', 'video']:
            token = f'<|{media_type.upper()}|>'
            token_id = self._tokenize(token)
            idx_list = findall(input_ids, token_id)
            if idx_list:
                merge_size = processor.image_processor.merge_size
                media_grid_thw = media_inputs.get(f'{media_type}_grid_thw')
                if media_type == 'video' and self.use_audio_in_video:
                    audio_lengths = audio_lengths_origin[video_audios_mask]
                    video_second_per_grid = media_inputs['video_second_per_grid']
                    _get_new_tokens_use_audio_in_video = partial(
                        self._get_new_tokens_use_audio_in_video,
                        video_grid_thw=media_grid_thw,
                        video_second_per_grid=video_second_per_grid,
                        audio_lengths=audio_lengths,
                        video_token_id=token_id,
                        audio_token_id=audio_token_id)
                    input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                        _get_new_tokens_use_audio_in_video)

                else:

                    def _get_new_tokens(i):
                        token_len = (media_grid_thw[i].prod() // (merge_size**2))
                        return token_id * token_len

                    input_ids, labels, loss_scale = self._extend_tokens(input_ids, labels, loss_scale, idx_list,
                                                                        _get_new_tokens)

        encoded['input_ids'] = input_ids
        encoded['labels'] = labels
        encoded['loss_scale'] = loss_scale
        encoded['audios'] = audios
        encoded.update(media_inputs)  # 将多模态内容加入其中
        return encoded

    def _post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """该函数通常用于解决混合模型训练zero2/zero3卡住的问题，
        即有的进程为纯文本数据未过vit，有的进程含图片数据过了vit。这里将创建dummy_image来解决。

        该函数将被注册在`model.forward`前的pre_forward_hook中。
        该函数需返回 含多模态信息的input_embeds。
        """
        # if not self.is_training:
        #     return inputs

        # input_ids = inputs['input_ids']
        # input_features = inputs.get('input_features')
        # feature_attention_mask = inputs.get('feature_attention_mask')

        # base_model = self.get_base_model(model)
        # inputs_embeds = base_model.thinker.model.embed_tokens(input_ids)
        # thinker_config = model.config.thinker_config
        # # 辅助函数，用于处理text/image/video混合模态数据场景。（内部会创建dummy_image）
        # inputs_embeds = self._get_inputs_embeds_hf(inputs_embeds, inputs, model.thinker.visual, self.processor,
        #                                            thinker_config)
        # # 含audio的混合模态数据场景
        # if input_features is None:
        #     if is_deepspeed_enabled() and not is_deepspeed_zero3_enabled():
        #         # 注意: 由于transformers实现中，经过audio部分模型层的次数与audio数量相关
        #         # 因此zero3在不同进程audios数不同场景下会卡住（需修改transformers代码修复）。此场景请使用zero2。
        #         input_features = input_ids.new_zeros([1, 128, 128], dtype=model.thinker.audio_tower.dtype)
        #         feature_attention_mask = input_ids.new_ones([1, 128], dtype=torch.bool)
        #         audio_embeds = model.thinker.get_audio_features(input_features, feature_attention_mask)
        #         inputs_embeds = inputs_embeds + audio_embeds.mean() * 0.
        # else:
        #     audio_embeds = model.thinker.get_audio_features(input_features, feature_attention_mask)
        #     audio_mask = (input_ids == thinker_config.audio_token_index).unsqueeze(-1).expand_as(inputs_embeds)
        #     audio_embeds = audio_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        #     inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_embeds)

        # return {'inputs_embeds': inputs_embeds}
        return inputs

    def _get_position_ids(self, inputs: Dict[str, Any]):
        """辅助函数，用来获取mrope的position_ids"""
        feature_attention_mask = inputs.get('feature_attention_mask')
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None
        video_second_per_grid = inputs.pop('video_second_per_grid', None)
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids, _ = self.model.thinker.get_rope_index(
            input_ids,
            inputs.get('image_grid_thw'),
            inputs.get('video_grid_thw'),
            attention_mask,
            self.use_audio_in_video,
            audio_feature_lengths,
            video_second_per_grid,
        )
        return self._concat_text_position_ids(position_ids)  # 第一维为text_position_ids

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """传入dataloader的`collate_fn`"""
        res = super()._data_collator(batch, padding_to=padding_to)
        if not self.padding_free and self.is_training:
            # 其中padding_free/packing场景将会在packing_row中处理position_ids。
            res['position_ids'] = self._get_position_ids(res)
        if 'position_ids' in res:
            # 创建`packed_seq_params`以支持padding_free/packing & flash-attn
            position_ids = res['position_ids']
            res['position_ids'] = position_ids[1:]
            res['text_position_ids'] = text_position_ids = position_ids[0]
            # https://github.com/huggingface/transformers/pull/40194
            # res.update(get_packed_seq_params(text_position_ids))
            if text_position_ids.shape[0] == 1:
                res.update(get_packed_seq_params(text_position_ids))
        return res

    def _data_collator_mm_data(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """处理`_data_collator`函数中的多模态部分。（该函数兼容padding_free/packing）"""
        res = super()._data_collator_mm_data(batch)
        video_second_per_grid = self.gather_list(batch, 'video_second_per_grid')
        if video_second_per_grid:
            res['video_second_per_grid'] = video_second_per_grid
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        feature_attention_mask = [
            b['feature_attention_mask'] for b in batch if b.get('feature_attention_mask') is not None
        ]
        if input_features:
            res['input_features'] = torch.concat(input_features)
            res['feature_attention_mask'] = torch.concat(feature_attention_mask)

        # -------- raw audios (for XLS-R) --------
        audios = [b['audios'] for b in batch if b.get('audios') is not None]
        if audios:
            # audios: List[Tensor [4_800_000]]
            res['audios'] = torch.stack([a[0] if isinstance(a, list) else a for a in audios], dim=0)  # [B, 4_800_000]
        # res['audios'] = torch.stack([b['audios'][0] if isinstance(b['audios'], list) else b['audios'] for b in batch], dim=0)
        return res

    def generate(self, model, *args, **kwargs):
        """`TransformersEngine`会调用template.generate方法进行文本生成，这里继承进行自定义。"""
        if kwargs.get('video_grid_thw') is not None:
            kwargs['use_audio_in_video'] = self.use_audio_in_video
        return super().generate(model, *args, **kwargs)


register_template(
    TemplateMeta('qwen2_5_omni_xlsr', prefix=[], prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
                 chat_sep=['<|im_end|>\n'], suffix=['<|im_end|>'],
                 system_prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
                 default_system='You are a helpful assistant.', stop_words=['<|endoftext|>'],
                 agent_template='hermes',
                 template_cls=Qwen2_5OmniXlsrTemplate))


if __name__ == '__main__':
    # 测试与debug
    model_path = os.getenv('THINKOMNI_MODEL_PATH', 'model/Qwen2.5-Omni-3B')
    audio_path = os.getenv('THINKOMNI_AUDIO_PATH', 'data/example.wav')
    model, processor = get_model_processor(model_path, model_type='qwen2_5_omni_xlsr')
    print('Load model')
    template = get_template(processor, template_type='qwen2_5_omni_xlsr')
    data = {
        # 'messages': [
        #     {'role': 'user', 'content': '描述视频<video>与图片<image>内容。'},
        #     {'role': 'assistant', 'content': '一个小孩和一只猫咪。'},
        # ],
        # 'videos': ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'],
        # 'images': ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png'],
        'messages': [
            {'role': 'user', 'content': '<audio>Identify suspicious regions in the speech that may be fake.'},
            {'role': 'assistant', 'content': "<think>\nBoundary Analysis: Distinct phase discontinuities and abrupt amplitude shifts are detected at the 0.76s and 1.24s splice points, indicating a digital insertion where the forged segment fails to align with the zero-crossing points of the surrounding carrier audio.\nSpectral Artifacts: The frequency spectrum within the 0.76s-1.24s window exhibits characteristic vocoder banding and a lack of high-frequency harmonics, contrasting sharply with the organic spectral density observed in the authentic speech before 0.76s and after 1.24s.\nProsodic Features: During the localized segment (0.76s-1.24s), the pitch contour becomes unnaturally flat and lacks the expected emotional modulation for the semantic context, disrupting the rhythmic flow established in the preceding phrase 'wallahu a'lam'.\nEnvironmental Consistency: The ambient noise floor drops significantly between 0.76s and 1.24s, creating a 'spectral vacuum' that does not match the room tone or reverberation characteristics present in the rest of the recording.\n</think>\n\nDetection Result: partially fake\nLocalization Result: 0.76-1.24"},
        ],
        "audios": [audio_path]
    }
    template.set_mode('train')
    encoded = template.encode(data)
    print('input_ids: ' + template.safe_decode(encoded['input_ids']))
    print('labels: ' + template.safe_decode(encoded['labels']))
    print('keys: ' + str(encoded.keys()))
    print('audios: ' + str(encoded['audios']))

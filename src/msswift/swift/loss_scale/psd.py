import os
import re
from typing import List, Tuple

from .base import ConfigLossScale


class PSDThinkDetLocLossScale(ConfigLossScale):
    """PSD loss scale: think / detection (class-weighted) / localization.

    Detection segment weight is per-sample from detection_class_weights by parsing
    \"Detection Result: 0 | 1 | 2\" (0=fully real, 1=fully fake, 2=partially fake),
    or legacy text \"fully real\" / \"fully fake\" / \"partially fake\".
    Weights are inverse-frequency (fully fake 24.33% -> highest weight).
    """
    loss_scale_config = 'psd.json'
    is_binary = False
    # Support both response styles:
    # 1) no-think: "Detection Result: ...\nLocalization Result: ..."
    # 2) think: "<think>...</think>\n\nDetection Result: ...\nLocalization Result: ..."
    _DETECTION_PATTERN = re.compile(r'(?:^|\n+)Detection Result:\s*([^\n]+)', re.IGNORECASE)
    _LOCALIZATION_PATTERN = re.compile(r'(?:^|\n+)Localization Result:\s*[^\n]*', re.IGNORECASE)

    def get_loss_scale(self, context: str, **kwargs) -> Tuple[List[str], List[float]]:
        if not isinstance(context, str):
            return super().get_loss_scale(context, **kwargs)
        w_think = float(os.environ.get('PSD_WEIGHT_THINK', self.loss_scale_map.get('think', 0.3)))
        w_loc = float(os.environ.get('PSD_WEIGHT_LOCALIZATION', self.loss_scale_map.get('localization', 0.5)))
        # detection 段总权重再乘上 detection_scale（默认 0.2）
        det_scale = float(os.environ.get('PSD_DETECTION_SCALE', self.loss_scale_map.get('detection_scale', 0.2)))
        class_weights = self.loss_scale_map.get('detection_class_weights', {})
        if isinstance(class_weights, dict):
            w_det_by_class = {k: float(v) for k, v in class_weights.items()}
        else:
            w_det_by_class = {'fully fake': 0.24, 'fully real': 0.36, 'partially fake': 0.4}
        # 支持数据中 "Detection Result: 0|1|2"（与 loss 中 detection_label 0/1/2 一致）
        for num, name in [('0', 'fully real'), ('1', 'fully fake'), ('2', 'partially fake')]:
            w_det_by_class.setdefault(num, w_det_by_class.get(name, 0.4))

        think_part = ''
        det_part = ''
        det_class = None
        loc_part = ''
        m_think = re.search(r'<think>.*?</think>', context, re.DOTALL)
        if m_think:
            think_part = m_think.group(0)
            rest_after_think = context[m_think.end():]
        else:
            rest_after_think = context
        m_det = self._DETECTION_PATTERN.search(rest_after_think)
        m_loc = self._LOCALIZATION_PATTERN.search(rest_after_think)
        if m_det:
            det_part = m_det.group(0)
            det_class = m_det.group(1).strip().lower()
        if m_loc:
            loc_part = m_loc.group(0)
        # det_class 可能是 "0"/"1"/"2" 等
        w_det = w_det_by_class.get(det_class, w_det_by_class.get('partially fake', 0.4)) * det_scale

        segments = []
        weights = []
        if think_part:
            segments.append(think_part)
            weights.append(w_think)
        if det_part:
            segments.append(det_part)
            weights.append(w_det)
        if loc_part:
            segments.append(loc_part)
            weights.append(w_loc)
        if not segments:
            return [context], [1.]
        return segments, weights

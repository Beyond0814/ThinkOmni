# Copyright (c) ModelScope Contributors. All rights reserved.
from .causal_lm import CustomCrossEntropyLoss
from .embedding import ContrastiveLoss, CosineSimilarityLoss, InfonceLoss, OnlineContrastiveLoss
from .reranker import ListwiseRerankerLoss, PointwiseRerankerLoss
from .psd import PSDDetectionLocalizationLoss
from .psd_spatial_localization import PSDSpatialLocalizationLoss

loss_map = {
    'cross_entropy': CustomCrossEntropyLoss,  # examples
    'psd_detection_localization': PSDDetectionLocalizationLoss,
    'psd_spatial_localization': PSDSpatialLocalizationLoss,
    # embedding
    'cosine_similarity': CosineSimilarityLoss,
    'contrastive': ContrastiveLoss,
    'online_contrastive': OnlineContrastiveLoss,
    'infonce': InfonceLoss,
    # # reranker
    'pointwise_reranker': PointwiseRerankerLoss,
    'listwise_reranker': ListwiseRerankerLoss,
}

# Copyright (c) OpenMMLab. All rights reserved.
from .citys_metric import CityscapesMetric
from .depth_metric import DepthMetric
from .iou_metric import IoUMetric
from .retinal_metrics import RetinalMetrics#这里是我新增的评测指标
from mmengine.registry import METRICS

__all__ = ['IoUMetric', 'CityscapesMetric', 'DepthMetric','RetinalMetrics']

# 注册自定义的评测指标
METRICS.register_module(name='RetinalMetrics', module=RetinalMetrics)
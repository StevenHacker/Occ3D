#!/usr/bin/env python3
# -*-coding:utf-8 -*-
"""
可见性计算库
============

使用方法:
    from visibility import compute_visibility, compute_frame_visibility
    from visibility import config
    
    # 修改配置
    config.ANGLE_THRESH_DEG = 0.3
    config.MIN_BLOCKERS = 5
    
    # 计算
    results = compute_frame_visibility(label_3d_list, points)
"""

from . import config
from .config import (
    TOTAL_SAMPLES,
    SIDE_FACES_ONLY,
    ANGLE_THRESH_DEG,
    MIN_BLOCKERS,
    MAX_SCENE_POINTS,
    SPATIAL_FILTER,
    VISIBILITY_THRESHOLDS,
    get_config_dict,
    print_config
)

from .core import (
    compute_visibility,
    compute_frame_visibility,
    classify_visibility
)

try:
    from .visualizer import (
        draw_visibility,
        draw_legend,
        project_point,
        project_bbox_corners,
        HAS_CV2
    )
except ImportError:
    HAS_CV2 = False

__version__ = '1.0.0'
__all__ = [
    'compute_visibility',
    'compute_frame_visibility',
    'classify_visibility',
    'draw_visibility',
    'config'
]

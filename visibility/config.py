#!/usr/bin/env python3
# -*-coding:utf-8 -*-
"""
可见性计算配置文件
==================

所有可调参数集中在此文件中
"""

# ============================================================
# 采样参数
# ============================================================

# 总采样点数（会按面积分配到各可见面）
TOTAL_SAMPLES = 60

# 只考虑侧面（前后左右），不考虑顶面和底面
# 对于车辆等物体，顶底面通常不重要
SIDE_FACES_ONLY = True

# 最小有效尺寸（米），小于此值的尺寸会被强制设为此值
MIN_SIZE = 0.01


# ============================================================
# 射线遮挡检测参数
# ============================================================

# 射线角度阈值（度）
# 用于定义射线的"宽度"，适应点云稀疏性
# 较小的值 -> 更严格的遮挡判断
# 较大的值 -> 更宽松的遮挡判断
ANGLE_THRESH_DEG = 0.5

# 最少遮挡点数
# 需要至少这么多点在射线锥体内才判定为遮挡
# 用于避免单个噪声点造成误判
MIN_BLOCKERS = 3

# 射线起点排除距离（米）
# 排除传感器附近的点，避免噪声干扰
RAY_START_MARGIN = 0.5

# 射线终点排除距离（米）
# 排除目标点附近的点，避免把目标本身判断为遮挡物
RAY_END_MARGIN = 0.3


# ============================================================
# 性能参数
# ============================================================

# 场景点云最大采样数
MAX_SCENE_POINTS = 8000

# 空间过滤：只检查传感器→bbox方向上的点云
# 对于真实LiDAR点云效果较好，随机点云可关闭
SPATIAL_FILTER = False

# 空间过滤扩展角度（度）
SPATIAL_FILTER_ANGLE = 20.0


# ============================================================
# 可见性分类阈值
# ============================================================

# 可见性分数 -> 状态分类
VISIBILITY_THRESHOLDS = {
    'VISIBLE': 0.70,   # >= 70% 为可见
    'PARTIAL': 0.30,   # >= 30% 为部分可见
    'OCCLUDED': 0.05,  # >= 5% 为遮挡
    # < 5% 为不可见 (BLOCKED)
}


# ============================================================
# 坐标系定义
# ============================================================

# bbox size 字段顺序: [width, height, length]
SIZE_ORDER = {
    'width': 0,   # size[0] = 宽度
    'height': 1,  # size[1] = 高度
    'length': 2,  # size[2] = 长度
}

# bbox orientation 字段: phi(yaw), theta(pitch), psi(roll)
# 旋转顺序: ZYX (先Z后Y最后X)
ROTATION_ORDER = 'ZYX'


# ============================================================
# 可视化参数
# ============================================================

# 可视化样式: 'full', 'score', 'status', 'compact'
VIS_STYLE = 'full'

# 是否绘制3D框线框
VIS_DRAW_WIREFRAME = True

# 是否添加图例
VIS_ADD_LEGEND = True

# 字体大小
VIS_FONT_SCALE = 0.6

# 线条粗细
VIS_THICKNESS = 2

# 状态颜色 (BGR格式)
STATUS_COLORS = {
    'VISIBLE':  (0, 255, 0),    # 绿色
    'PARTIAL':  (0, 255, 255),  # 黄色
    'OCCLUDED': (0, 165, 255),  # 橙色
    'BLOCKED':  (0, 0, 255),    # 红色
    'ERROR':    (128, 128, 128) # 灰色
}


# ============================================================
# 采集车过滤参数
# ============================================================

# 采集车尺寸（米），用于过滤原点附近的点
EGO_CAR_LENGTH = 5.0
EGO_CAR_WIDTH = 3.0
EGO_CAR_HEIGHT = 3.0


# ============================================================
# 辅助函数
# ============================================================

def get_config_dict():
    """返回所有配置作为字典"""
    return {
        'total_samples': TOTAL_SAMPLES,
        'side_faces_only': SIDE_FACES_ONLY,
        'angle_thresh_deg': ANGLE_THRESH_DEG,
        'min_blockers': MIN_BLOCKERS,
        'max_scene_points': MAX_SCENE_POINTS,
        'ray_start_margin': RAY_START_MARGIN,
        'ray_end_margin': RAY_END_MARGIN,
        'spatial_filter': SPATIAL_FILTER,
        'spatial_filter_angle': SPATIAL_FILTER_ANGLE,
    }


def print_config():
    """打印当前配置"""
    print("=" * 50)
    print("当前配置")
    print("=" * 50)
    print(f"采样参数:")
    print(f"  总采样点数: {TOTAL_SAMPLES}")
    print(f"  只考虑侧面: {SIDE_FACES_ONLY}")
    print(f"射线检测参数:")
    print(f"  角度阈值: {ANGLE_THRESH_DEG}°")
    print(f"  最少遮挡点数: {MIN_BLOCKERS}")
    print(f"  射线起点排除: {RAY_START_MARGIN}m")
    print(f"  射线终点排除: {RAY_END_MARGIN}m")
    print(f"性能优化参数:")
    print(f"  空间过滤: {SPATIAL_FILTER}")
    print(f"  过滤角度: ±{SPATIAL_FILTER_ANGLE}°")
    print(f"  最大点数: {MAX_SCENE_POINTS}")
    print("=" * 50)


if __name__ == '__main__':
    print_config()

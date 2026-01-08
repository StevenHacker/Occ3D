#!/usr/bin/env python3
"""
旋转角度调试脚本
用于验证bbox的旋转是否正确
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

def test_rotation():
    """测试旋转矩阵"""
    
    # 测试数据 (用户的两轮电瓶车)
    bbox = {
        'position': {'x': 17.51, 'y': 2.57, 'z': -1.56},
        'size': ['0.8466', '1.4962', '1.8604'],  # [w, h, l]
        'orientation': {
            'phi': '3.1166',    # ≈ π, yaw
            'theta': '0.0046',  # ≈ 0, pitch
            'psi': '0.0048'     # ≈ 0, roll
        }
    }
    
    # 解析
    l = float(bbox['size'][2])  # 长 ≈ 1.86m
    w = float(bbox['size'][0])  # 宽 ≈ 0.85m
    h = float(bbox['size'][1])  # 高 ≈ 1.50m
    
    phi = float(bbox['orientation']['phi'])
    theta = float(bbox['orientation']['theta'])
    psi = float(bbox['orientation']['psi'])
    
    print("=" * 50)
    print("bbox 尺寸:")
    print(f"  长(l): {l:.2f}m (前后方向)")
    print(f"  宽(w): {w:.2f}m (左右方向)")
    print(f"  高(h): {h:.2f}m (上下方向)")
    print()
    print("bbox 旋转角:")
    print(f"  phi (yaw):   {phi:.4f} rad = {np.degrees(phi):.1f}°")
    print(f"  theta (pitch): {theta:.4f} rad = {np.degrees(theta):.1f}°")
    print(f"  psi (roll):    {psi:.4f} rad = {np.degrees(psi):.1f}°")
    print()
    
    # 旋转矩阵
    rotation = R.from_euler('ZYX', [phi, theta, psi])
    rot_matrix = rotation.as_matrix()
    
    print("旋转矩阵:")
    print(rot_matrix)
    print()
    
    # 测试车头方向
    # 在用户坐标系中，-l方向是车头
    # 局部坐标系中车头方向向量是 [-1, 0, 0]
    head_direction_local = np.array([-1, 0, 0])
    head_direction_world = rot_matrix @ head_direction_local
    
    print("车头方向 (局部 -> 世界):")
    print(f"  局部: {head_direction_local}")
    print(f"  世界: {head_direction_world}")
    print(f"  世界xy平面角度: {np.degrees(np.arctan2(head_direction_world[1], head_direction_world[0])):.1f}°")
    print()
    
    # 8个角点
    half_l, half_w, half_h = l/2, w/2, h/2
    corners_local = np.array([
        [-half_l, -half_w, -half_h],  # 左前下
        [-half_l, -half_w,  half_h],  # 左前上
        [-half_l,  half_w, -half_h],  # 右前下
        [-half_l,  half_w,  half_h],  # 右前上
        [ half_l, -half_w, -half_h],  # 左后下
        [ half_l, -half_w,  half_h],  # 左后上
        [ half_l,  half_w, -half_h],  # 右后下
        [ half_l,  half_w,  half_h],  # 右后上
    ])
    
    # 旋转后的角点
    corners_world = (rot_matrix @ corners_local.T).T
    
    print("角点 (局部 -> 旋转后):")
    labels = ['左前下', '左前上', '右前下', '右前上', '左后下', '左后上', '右后下', '右后上']
    for i, (local, world, label) in enumerate(zip(corners_local, corners_world, labels)):
        print(f"  {label}: {local} -> {world}")
    
    print()
    print("=" * 50)
    
    # 验证scipy和手动计算是否一致
    print("\n验证 scipy 和手动计算:")
    
    # 手动计算 ZYX 旋转矩阵
    cz, sz = np.cos(phi), np.sin(phi)
    cy, sy = np.cos(theta), np.sin(theta)
    cx, sx = np.cos(psi), np.sin(psi)
    
    # R = Rz @ Ry @ Rx
    R_manual = np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy,   cy*sx,            cy*cx]
    ])
    
    print(f"scipy矩阵与手动计算差异: {np.max(np.abs(rot_matrix - R_manual)):.2e}")
    
    if np.max(np.abs(rot_matrix - R_manual)) < 1e-10:
        print("✓ 一致")
    else:
        print("✗ 不一致!")


if __name__ == '__main__':
    test_rotation()

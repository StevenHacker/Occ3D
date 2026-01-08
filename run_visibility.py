#!/usr/bin/env python3
# -*-coding:utf-8 -*-
"""
3D标注框可见性计算脚本
======================

【使用方法】
python run_visibility.py --clip_path /path/to/clip --visualize

【参数说明】
--clip_path     : 数据clip路径
--visualize     : 是否生成可视化图像
--output_json   : 是否输出可见性结果JSON
--style         : 可视化样式 (full/score/status/compact)

【示例】
# 仅计算可见性
python run_visibility.py --clip_path /data/clip_001

# 计算并可视化
python run_visibility.py --clip_path /data/clip_001 --visualize

# 输出JSON结果
python run_visibility.py --clip_path /data/clip_001 --output_json
"""

import os
import sys
import argparse
import json
import glob
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# 导入可见性计算接口
from visibility_api import (
    compute_frame_visibility,
    compute_bbox_visibility,
    classify_visibility,
    visualize_visibility,
    HAS_VISUALIZER
)

# cv2 用于可视化
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# pypcd 用于读取点云
try:
    from pypcd import pypcd
    HAS_PYPCD = True
except ImportError:
    HAS_PYPCD = False
    print("[提示] pypcd未安装，将使用备用方法读取pcd文件")


# ============================================================
# 工具函数 (与您原代码兼容)
# ============================================================

def xyzquat2mat(xyzquat):
    """四元数转变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(xyzquat[3:]).as_matrix()
    T[:3, 3] = xyzquat[:3]
    return T


def build_ego_to_lidar_transform(translation, rotation):
    """构建ego到lidar的变换矩阵"""
    t = np.array(translation, dtype=np.float32)
    rot = np.array(rotation, dtype=np.float32)
    
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = t
    
    T_inv = np.linalg.inv(T)
    return T_inv


def read_camera_param(params_json, camera_name):
    """读取相机参数"""
    with open(params_json, 'r', encoding='utf-8') as f:
        json_info = json.load(f)
        info = json_info['sensors']['Cameras']
        for sensor in info:
            if sensor['name'] == camera_name:
                dist = np.array(sensor['intrinsic']['D'], dtype=np.float64)
                mtx = np.array(sensor['intrinsic']['K'], dtype=np.float64)
                quat = np.array(sensor['extrinsic']['to_lidar_main'], dtype=np.float64)
                extr = xyzquat2mat(quat)
                w = sensor['width']
                h = sensor['height']
                return dist, mtx, extr, w, h
    return None, None, None, None, None


def filter_origin_car(pts):
    """过滤采集车周围的点"""
    suv_length = 5.0
    suv_width = 3.0
    suv_height = 3.0
    
    car_x_min, car_x_max = -suv_length / 2, suv_length / 2
    car_y_min, car_y_max = -suv_width / 2, suv_width / 2
    car_z_min, car_z_max = -suv_height / 2, suv_height / 2
    
    car_mask = (
        (pts[:, 0] >= car_x_min) & (pts[:, 0] <= car_x_max) &
        (pts[:, 1] >= car_y_min) & (pts[:, 1] <= car_y_max) &
        (pts[:, 2] >= car_z_min) & (pts[:, 2] <= car_z_max)
    )
    valid_mask = np.logical_not(car_mask)
    return pts[valid_mask]


def read_point_cloud_raw(pcd_path):
    """
    纯Python读取PCD文件（备用方法，不依赖pypcd）
    支持ascii和binary格式
    """
    with open(pcd_path, 'rb') as f:
        # 读取头部
        header = {}
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('DATA'):
                data_type = line.split()[-1]
                break
            if ' ' in line:
                key, value = line.split(' ', 1)
                header[key] = value
        
        # 获取点数和字段
        num_points = int(header.get('POINTS', 0))
        fields = header.get('FIELDS', 'x y z').split()
        
        # 找到xyz的索引
        try:
            x_idx = fields.index('x')
            y_idx = fields.index('y')
            z_idx = fields.index('z')
        except ValueError:
            x_idx, y_idx, z_idx = 0, 1, 2
        
        if data_type == 'ascii':
            # ASCII格式
            points = []
            for _ in range(num_points):
                line = f.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    values = line.split()
                    if len(values) > max(x_idx, y_idx, z_idx):
                        x = float(values[x_idx])
                        y = float(values[y_idx])
                        z = float(values[z_idx])
                        points.append([x, y, z])
            return np.array(points, dtype=np.float64)
        
        else:
            # Binary格式
            sizes = [int(s) for s in header.get('SIZE', '4 4 4').split()]
            types = header.get('TYPE', 'F F F').split()
            
            # 构建dtype
            dtype_map = {'F': 'f', 'I': 'i', 'U': 'I'}
            dtype_list = []
            for i, (field, size, typ) in enumerate(zip(fields, sizes, types)):
                np_type = dtype_map.get(typ, 'f') + str(size)
                dtype_list.append((field, np_type))
            
            dt = np.dtype(dtype_list)
            data = np.frombuffer(f.read(), dtype=dt, count=num_points)
            
            points = np.column_stack([data['x'], data['y'], data['z']])
            return points.astype(np.float64)


def read_point_cloud(pcd_path):
    """读取点云文件（自动选择方法）"""
    # 优先使用pypcd（如果可用且稳定）
    if HAS_PYPCD:
        try:
            pypcd.pcd_type_to_numpy_type[('I', 1)] = np.int8
            pcd_obj = pypcd.PointCloud.from_path(pcd_path)
            pcd_data = pcd_obj.pc_data
            pcd_pts = np.column_stack([pcd_data['x'], pcd_data['y'], pcd_data['z']])
            return pcd_pts
        except Exception as e:
            print(f"[警告] pypcd读取失败，使用备用方法: {e}")
    
    # 备用方法：纯Python解析
    return read_point_cloud_raw(pcd_path)


def transform_bbox_to_lidar(bbox, T_ego_to_lidar):
    """将bbox从ego坐标系转换到lidar坐标系"""
    # 强制转float，兼容字符串类型
    point_homo = np.array([
        float(bbox['position']['x']),
        float(bbox['position']['y']),
        float(bbox['position']['z']),
        1.0
    ], dtype=np.float64)
    
    point_lidar_homo = np.dot(T_ego_to_lidar, point_homo)
    
    bbox_lidar = bbox.copy()
    bbox_lidar['position'] = {
        'x': float(point_lidar_homo[0]),
        'y': float(point_lidar_homo[1]),
        'z': float(point_lidar_homo[2])
    }
    
    # 确保size和orientation也是数字类型
    if 'size' in bbox_lidar:
        bbox_lidar['size'] = [float(s) for s in bbox_lidar['size']]
    
    if 'orientation' in bbox_lidar:
        bbox_lidar['orientation'] = {
            k: float(v) for k, v in bbox_lidar['orientation'].items()
        }
    
    return bbox_lidar


# ============================================================
# 主处理函数
# ============================================================

def process_frame(
    label_json: str,
    clip_path: str,
    visualize: bool = False,
    output_json: bool = False,
    style: str = 'full',
    verbose: bool = False
) -> dict:
    """
    处理单帧数据
    
    Args:
        label_json: 标注JSON文件路径
        clip_path: clip数据路径
        visualize: 是否生成可视化图像
        output_json: 是否返回详细结果
        style: 可视化样式
        verbose: 是否打印详细信息
    
    Returns:
        帧的可见性结果
    """
    frame_name = os.path.basename(label_json)
    
    if verbose:
        print(f"  [1/4] 读取标注: {frame_name}")
    
    # 读取标注数据
    with open(label_json, 'r', encoding='utf-8') as fp:
        all_data = json.load(fp)
    
    # 路径
    pcd_path = os.path.join(
        os.path.dirname(clip_path),
        all_data['frame_info']['lidar_object_info']['lidar_path']
    )
    params_json = os.path.join(clip_path, 'sensor_datas', 'info.json')
    
    # 读取点云
    if verbose:
        print(f"  [2/4] 读取点云: {os.path.basename(pcd_path)}")
    pcd_pts = read_point_cloud(pcd_path)
    if verbose:
        print(f"        点云数量: {len(pcd_pts)}")
    
    # 获取3D标注
    label_3D = all_data['frame_info']['lidar_object_info']['lidar_object_info']
    if verbose:
        print(f"        标注框数量: {len(label_3D)}")
    
    # 坐标转换
    lidar2ego_translation = all_data['sensor_info']['lidar_info']['lidar_main']['lidar2ego_translation']
    lidar2ego_rotation = all_data['sensor_info']['lidar_info']['lidar_main']['lidar2ego_rotation']
    T_ego_to_lidar = build_ego_to_lidar_transform(lidar2ego_translation, lidar2ego_rotation)
    T_ego_to_lidar = np.asarray(T_ego_to_lidar, dtype=np.float64)
    
    # 转换所有bbox到lidar坐标系
    label_3d_lidar = []
    for bbox in label_3D:
        bbox_lidar = transform_bbox_to_lidar(bbox, T_ego_to_lidar)
        label_3d_lidar.append(bbox_lidar)
    
    # 过滤采集车
    filter_pcd_pts = filter_origin_car(pcd_pts)
    
    # ========== 计算可见性 ==========
    if verbose:
        print(f"  [3/4] 计算可见性 (首次运行需编译，请稀等30-60秒)...")
    
    import time
    t0 = time.time()
    visibility_results = compute_frame_visibility(label_3d_lidar, filter_pcd_pts)
    t1 = time.time()
    
    if verbose:
        print(f"        完成! 耗时: {t1-t0:.1f}s")
    print(f"\n[{frame_name}] 可见性结果:")
    for track_id, info in visibility_results.items():
        print(f"  {track_id}: {info['score']:.0%} ({info['status']})")
    
    # ========== 可视化 (可选) ==========
    if visualize and HAS_CV2 and HAS_VISUALIZER:
        if verbose:
            print(f"  [4/4] 生成可视化...")
        camera_list = all_data['frame_info']['camera_object_info'].keys()
        
        # 创建可视化输出目录
        vis_path = os.path.join(clip_path, 'visibility_vis')
        os.makedirs(vis_path, exist_ok=True)
        
        for camera in camera_list:
            # 读取相机参数
            distortion, intrinsic, extrinsic, width, height = \
                read_camera_param(params_json, camera)
            
            if intrinsic is None:
                continue
            
            # 读取图像
            camera_image_path = os.path.join(
                clip_path,
                all_data['sensor_info']['camera_info'][camera]['path']
            )
            
            if not os.path.exists(camera_image_path):
                continue
            
            img = cv2.imread(camera_image_path)
            if img is None:
                continue
            
            # 绑制可见性
            img = visualize_visibility(
                img,
                label_3d_lidar,
                visibility_results,
                extrinsic,
                intrinsic,
                distortion,
                width,
                height,
                style=style,
                draw_wireframe=True,
                add_legend=True,
                filter_camera=camera
            )
            
            # 保存
            output_name = os.path.basename(camera_image_path).replace('.jpg', '_vis.jpg')
            camera_vis_path = os.path.join(vis_path, camera)
            os.makedirs(camera_vis_path, exist_ok=True)
            output_path = os.path.join(camera_vis_path, output_name)
            cv2.imwrite(output_path, img)
        
        print(f"  可视化已保存到: {vis_path}")
    
    # 返回结果
    frame_result = {
        'frame': frame_name,
        'visibility': visibility_results
    }
    
    return frame_result


def process_clip(
    clip_path: str,
    visualize: bool = False,
    output_json: bool = False,
    style: str = 'full',
    max_frames: int = None
) -> dict:
    """
    处理整个clip
    
    Args:
        clip_path: clip数据路径
        visualize: 是否生成可视化图像
        output_json: 是否输出JSON结果
        style: 可视化样式
        max_frames: 最大处理帧数 (None表示全部)
    
    Returns:
        所有帧的可见性结果
    """
    # 获取所有标注文件
    label_path = os.path.join(clip_path, 'labels')
    label_list = sorted(glob.glob(os.path.join(label_path, "*.json")))
    
    if len(label_list) == 0:
        print(f"[错误] 未找到标注文件: {label_path}")
        return {}
    
    print(f"找到 {len(label_list)} 帧标注")
    
    # 限制帧数
    if max_frames is not None:
        label_list = label_list[:max_frames]
    
    # 清理旧的可视化目录
    if visualize:
        vis_path = os.path.join(clip_path, 'visibility_vis')
        if os.path.exists(vis_path):
            shutil.rmtree(vis_path)
    
    # 处理每一帧
    all_results = {}
    
    # 第一帧详细打印（Numba编译）
    first_frame = True
    
    for label_json in tqdm(label_list, desc="处理帧"):
        try:
            verbose = first_frame  # 第一帧显示详细信息
            frame_result = process_frame(
                label_json, clip_path, visualize, output_json, style, verbose
            )
            all_results[frame_result['frame']] = frame_result['visibility']
            first_frame = False
        except Exception as e:
            print(f"[错误] 处理 {label_json} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存JSON结果
    if output_json:
        output_path = os.path.join(clip_path, 'visibility_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            # 转换为可序列化格式
            serializable_results = {}
            for frame, vis in all_results.items():
                serializable_results[frame] = {
                    track_id: {
                        'score': info['score'],
                        'status': info['status'],
                        'n_samples': info['n_samples'],
                        'n_blocked': info['n_blocked'],
                        'n_visible': info['n_visible']
                    }
                    for track_id, info in vis.items()
                }
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")
    
    # 统计
    print("\n" + "=" * 50)
    print("可见性统计")
    print("=" * 50)
    
    status_counts = {'VISIBLE': 0, 'PARTIAL': 0, 'OCCLUDED': 0, 'BLOCKED': 0}
    total = 0
    
    for frame, vis in all_results.items():
        for track_id, info in vis.items():
            status = info['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            total += 1
    
    print(f"总目标数: {total}")
    for status, count in status_counts.items():
        pct = count / total * 100 if total > 0 else 0
        print(f"  {status}: {count} ({pct:.1f}%)")
    print("=" * 50)
    
    return all_results


# ============================================================
# 命令行接口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='3D标注框可见性计算',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 仅计算可见性
  python run_visibility.py --clip_path /data/clip_001
  
  # 计算并可视化
  python run_visibility.py --clip_path /data/clip_001 --visualize
  
  # 输出JSON结果
  python run_visibility.py --clip_path /data/clip_001 --output_json
  
  # 只处理前5帧
  python run_visibility.py --clip_path /data/clip_001 --max_frames 5 --visualize
        """
    )
    
    parser.add_argument(
        '--clip_path', type=str, required=True,
        help='数据clip路径'
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='是否生成可视化图像'
    )
    parser.add_argument(
        '--output_json', action='store_true',
        help='是否输出可见性结果JSON'
    )
    parser.add_argument(
        '--style', type=str, default='full',
        choices=['full', 'score', 'status', 'compact'],
        help='可视化样式 (default: full)'
    )
    parser.add_argument(
        '--max_frames', type=int, default=None,
        help='最大处理帧数'
    )
    
    args = parser.parse_args()
    
    # 检查路径
    if not os.path.exists(args.clip_path):
        print(f"[错误] 路径不存在: {args.clip_path}")
        sys.exit(1)
    
    # 检查依赖
    if args.visualize and not HAS_CV2:
        print("[警告] cv2未安装，无法进行可视化")
        args.visualize = False
    
    # 处理
    print(f"处理clip: {args.clip_path}")
    print(f"可视化: {args.visualize}")
    print(f"输出JSON: {args.output_json}")
    
    results = process_clip(
        args.clip_path,
        visualize=args.visualize,
        output_json=args.output_json,
        style=args.style,
        max_frames=args.max_frames
    )


if __name__ == '__main__':
    main()

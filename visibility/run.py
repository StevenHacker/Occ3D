#!/usr/bin/env python3
# -*-coding:utf-8 -*-
"""
可见性计算主程序
================

使用方法:
    python run.py --clip_path /path/to/clip --visualize
    或
    python -m visibility.run --clip_path /path/to/clip --visualize
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

# 支持直接运行和模块运行两种方式
if __name__ == '__main__' and __package__ is None:
    # 直接运行时，添加父目录到路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import visibility.config as config
    from visibility.core import compute_frame_visibility
    try:
        from visibility.visualizer import draw_visibility, HAS_CV2
    except ImportError:
        HAS_CV2 = False
        draw_visibility = None
else:
    # 作为模块运行
    from . import config
    from .core import compute_frame_visibility
    try:
        from .visualizer import draw_visibility, HAS_CV2
    except ImportError:
        HAS_CV2 = False
        draw_visibility = None

try:
    import cv2
except ImportError:
    cv2 = None


# ============================================================
# 数据读取工具
# ============================================================

def xyzquat2mat(xyzquat):
    """四元数转变换矩阵"""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(xyzquat[3:]).as_matrix()
    T[:3, 3] = xyzquat[:3]
    return T


def read_camera_param(params_json, camera_name):
    """读取相机参数"""
    with open(params_json, 'r', encoding='utf-8') as f:
        json_info = json.load(f)
        for sensor in json_info['sensors']['Cameras']:
            if sensor['name'] == camera_name:
                return {
                    'distortion': np.array(sensor['intrinsic']['D'], dtype=np.float64),
                    'intrinsic': np.array(sensor['intrinsic']['K'], dtype=np.float64),
                    'extrinsic': xyzquat2mat(np.array(sensor['extrinsic']['to_lidar_main'], dtype=np.float64)),
                    'width': sensor['width'],
                    'height': sensor['height']
                }
    return None


def read_pcd(pcd_path):
    """读取PCD文件"""
    with open(pcd_path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('DATA'):
                data_type = line.split()[-1]
                break
            if ' ' in line:
                key, value = line.split(' ', 1)
                header[key] = value
        
        num_points = int(header.get('POINTS', 0))
        fields = header.get('FIELDS', 'x y z').split()
        
        try:
            x_idx, y_idx, z_idx = fields.index('x'), fields.index('y'), fields.index('z')
        except ValueError:
            x_idx, y_idx, z_idx = 0, 1, 2
        
        if data_type == 'ascii':
            points = []
            for _ in range(num_points):
                line = f.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    values = line.split()
                    if len(values) > max(x_idx, y_idx, z_idx):
                        points.append([float(values[x_idx]), float(values[y_idx]), float(values[z_idx])])
            return np.array(points, dtype=np.float64)
        else:
            sizes = [int(s) for s in header.get('SIZE', '4 4 4').split()]
            types = header.get('TYPE', 'F F F').split()
            dtype_map = {'F': 'f', 'I': 'i', 'U': 'I'}
            dtype_list = [(field, dtype_map.get(typ, 'f') + str(size)) 
                          for field, size, typ in zip(fields, sizes, types)]
            data = np.frombuffer(f.read(), dtype=np.dtype(dtype_list), count=num_points)
            return np.column_stack([data['x'], data['y'], data['z']]).astype(np.float64)


def filter_ego_car(pts):
    """过滤采集车"""
    half_l = config.EGO_CAR_LENGTH / 2
    half_w = config.EGO_CAR_WIDTH / 2
    half_h = config.EGO_CAR_HEIGHT / 2
    
    mask = (
        (pts[:, 0] >= -half_l) & (pts[:, 0] <= half_l) &
        (pts[:, 1] >= -half_w) & (pts[:, 1] <= half_w) &
        (pts[:, 2] >= -half_h) & (pts[:, 2] <= half_h)
    )
    return pts[~mask]


def build_ego_to_lidar(translation, rotation):
    """构建ego到lidar变换"""
    T = np.eye(4)
    T[:3, :3] = np.array(rotation, dtype=np.float32)
    T[:3, 3] = np.array(translation, dtype=np.float32)
    return np.linalg.inv(T)


def transform_bbox(bbox, T):
    """
    转换bbox到lidar坐标系
    
    注意：如果T包含旋转，需要同时转换朝向角
    """
    # 转换位置
    point = np.array([
        float(bbox['position']['x']),
        float(bbox['position']['y']),
        float(bbox['position']['z']),
        1.0
    ], dtype=np.float64)
    
    point_lidar = T @ point
    
    bbox_new = bbox.copy()
    bbox_new['position'] = {
        'x': float(point_lidar[0]),
        'y': float(point_lidar[1]),
        'z': float(point_lidar[2])
    }
    
    # 转换尺寸为float
    if 'size' in bbox_new:
        bbox_new['size'] = [float(s) for s in bbox_new['size']]
    
    # 转换朝向角
    if 'orientation' in bbox_new:
        ori = bbox_new['orientation']
        phi = float(ori.get('phi', 0))
        theta = float(ori.get('theta', 0))
        psi = float(ori.get('psi', 0))
        
        # 检查T是否包含旋转（非单位矩阵）
        R_transform = T[:3, :3]
        if not np.allclose(R_transform, np.eye(3), atol=1e-6):
            # 将bbox的旋转与变换矩阵的旋转组合
            # bbox在ego坐标系中的旋转
            R_bbox_ego = R.from_euler('ZYX', [phi, theta, psi]).as_matrix()
            # 组合旋转：先bbox旋转，再坐标系变换
            R_bbox_lidar = R_transform @ R_bbox_ego
            # 提取新的欧拉角
            new_euler = R.from_matrix(R_bbox_lidar).as_euler('ZYX')
            phi, theta, psi = new_euler[0], new_euler[1], new_euler[2]
        
        bbox_new['orientation'] = {
            'phi': float(phi),
            'theta': float(theta),
            'psi': float(psi)
        }
    
    return bbox_new


# ============================================================
# 主处理函数
# ============================================================

def process_frame(label_json, clip_path, visualize=False, verbose=False):
    """处理单帧"""
    frame_name = os.path.basename(label_json)
    
    # 读取数据
    with open(label_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pcd_path = os.path.join(os.path.dirname(clip_path), 
                            data['frame_info']['lidar_object_info']['lidar_path'])
    
    if verbose:
        print(f"  读取点云: {os.path.basename(pcd_path)}")
    
    points = read_pcd(pcd_path)
    points = filter_ego_car(points)
    
    if verbose:
        print(f"  点云数量: {len(points)}")
    
    # 获取标注
    label_3d = data['frame_info']['lidar_object_info']['lidar_object_info']
    
    # 坐标转换
    lidar_info = data['sensor_info']['lidar_info']['lidar_main']
    T = build_ego_to_lidar(lidar_info['lidar2ego_translation'], lidar_info['lidar2ego_rotation'])
    label_3d_lidar = [transform_bbox(b, T) for b in label_3d]
    
    if verbose:
        print(f"  标注框数量: {len(label_3d_lidar)}")
        print(f"  计算可见性...")
    
    # 计算可见性
    results = compute_frame_visibility(label_3d_lidar, points)
    
    # 打印结果
    print(f"\n[{frame_name}]")
    for tid, info in results.items():
        status = info['status']
        if status in ['ERROR', 'PARSE_ERROR', 'SAMPLE_ERROR']:
            print(f"  {tid}: ❌ {status}")
        else:
            print(f"  {tid}: {info['score']:.0%} ({status})")
    
    # 可视化
    if visualize and HAS_CV2 and cv2 is not None:
        params_json = os.path.join(clip_path, 'sensor_datas', 'info.json')
        camera_list = data['frame_info']['camera_object_info'].keys()
        
        vis_path = os.path.join(clip_path, 'visibility_vis')
        os.makedirs(vis_path, exist_ok=True)
        
        for camera in camera_list:
            cam_params = read_camera_param(params_json, camera)
            if cam_params is None:
                continue
            
            img_path = os.path.join(clip_path, data['sensor_info']['camera_info'][camera]['path'])
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img = draw_visibility(
                img, label_3d_lidar, results,
                cam_params['extrinsic'],
                cam_params['intrinsic'],
                cam_params['width'],
                cam_params['height'],
                filter_camera=camera
            )
            
            out_dir = os.path.join(vis_path, camera)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, os.path.basename(img_path).replace('.jpg', '_vis.jpg'))
            cv2.imwrite(out_path, img)
    
    return {'frame': frame_name, 'results': results}


def process_clip(clip_path, visualize=False, output_json=False, max_frames=None):
    """处理整个clip"""
    label_path = os.path.join(clip_path, 'labels')
    label_list = sorted(glob.glob(os.path.join(label_path, "*.json")))
    
    if not label_list:
        print(f"[错误] 未找到标注: {label_path}")
        return {}
    
    print(f"找到 {len(label_list)} 帧")
    
    if max_frames:
        label_list = label_list[:max_frames]
    
    # 清理旧可视化
    if visualize:
        vis_path = os.path.join(clip_path, 'visibility_vis')
        if os.path.exists(vis_path):
            shutil.rmtree(vis_path)
    
    # 处理
    all_results = {}
    for i, label_json in enumerate(tqdm(label_list, desc="处理帧")):
        try:
            result = process_frame(label_json, clip_path, visualize, verbose=(i==0))
            all_results[result['frame']] = result['results']
        except Exception as e:
            print(f"[错误] {label_json}: {e}")
    
    # 保存JSON
    if output_json:
        out_path = os.path.join(clip_path, 'visibility_results.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n结果保存到: {out_path}")
    
    # 统计
    print("\n" + "=" * 40)
    print("统计")
    print("=" * 40)
    counts = {'VISIBLE': 0, 'PARTIAL': 0, 'OCCLUDED': 0, 'BLOCKED': 0, 'ERROR': 0}
    total = 0
    for frame_results in all_results.values():
        for info in frame_results.values():
            status = info.get('status', 'ERROR')
            if status not in counts:
                status = 'ERROR'
            counts[status] += 1
            total += 1
    
    print(f"总目标: {total}")
    for status, count in counts.items():
        if count > 0:
            print(f"  {status}: {count} ({count/total*100:.1f}%)")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='可见性计算')
    parser.add_argument('--clip_path', type=str, default=None, help='数据路径')
    parser.add_argument('--visualize', action='store_true', help='生成可视化')
    parser.add_argument('--output_json', action='store_true', help='输出JSON')
    parser.add_argument('--max_frames', type=int, default=None, help='最大帧数')
    parser.add_argument('--print_config', action='store_true', help='打印配置')
    
    args = parser.parse_args()
    
    if args.print_config:
        config.print_config()
        return
    
    if args.clip_path is None:
        parser.error("请提供 --clip_path 参数")
    
    if not os.path.exists(args.clip_path):
        print(f"[错误] 路径不存在: {args.clip_path}")
        sys.exit(1)
    
    print(f"处理: {args.clip_path}")
    print(f"可视化: {args.visualize}")
    
    process_clip(args.clip_path, args.visualize, args.output_json, args.max_frames)


if __name__ == '__main__':
    main()

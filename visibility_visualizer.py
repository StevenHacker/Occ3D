#!/usr/bin/env python3
# -*-coding:utf-8 -*-
"""
3D标注框可见性可视化工具
========================

【功能】
- 将3D标注框中心点投影到各相机视角
- 在图像上标注可见性分数和状态
- 支持多种可视化样式

【使用方法】
```python
from visibility_visualizer import VisibilityVisualizer

visualizer = VisibilityVisualizer()
img = visualizer.draw_visibility_on_image(
    img, label_3d_list, visibility_results, 
    extrinsic, intrinsic, distortion, width, height
)
cv2.imwrite('output.jpg', img)
```

【依赖安装】
如果遇到cv2导入错误，请尝试：
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
pip install opencv-python-headless==4.5.5.64
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union

# cv2 可选导入
try:
    import cv2
    HAS_CV2 = True
except ImportError as e:
    HAS_CV2 = False
    CV2_ERROR = str(e)
    print(f"[警告] OpenCV导入失败: {e}")
    print("[提示] 请尝试: pip install opencv-python-headless==4.5.5.64")


def _check_cv2():
    """检查cv2是否可用"""
    if not HAS_CV2:
        raise ImportError(
            f"OpenCV (cv2) 导入失败: {CV2_ERROR}\n"
            "请尝试以下命令修复:\n"
            "  pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y\n"
            "  pip install opencv-python-headless==4.5.5.64"
        )


class VisibilityVisualizer:
    """可见性可视化器"""
    
    # 状态对应的颜色 (BGR格式)
    STATUS_COLORS = {
        'VISIBLE':  (0, 255, 0),    # 绿色
        'PARTIAL':  (0, 255, 255),  # 黄色
        'OCCLUDED': (0, 165, 255),  # 橙色
        'BLOCKED':  (0, 0, 255),    # 红色
        'UNKNOWN':  (128, 128, 128) # 灰色
    }
    
    # 状态对应的中文
    STATUS_CN = {
        'VISIBLE':  '可见',
        'PARTIAL':  '部分',
        'OCCLUDED': '遮挡',
        'BLOCKED':  '不可见',
        'UNKNOWN':  '未知'
    }
    
    def __init__(self, font_scale: float = 0.6, thickness: int = 2):
        """
        初始化可视化器
        
        Args:
            font_scale: 字体大小
            thickness: 线条粗细
        """
        self.font_scale = font_scale
        self.thickness = thickness
        if HAS_CV2:
            self.font = cv2.FONT_HERSHEY_SIMPLEX
        else:
            self.font = None
    
    def project_point_to_image(
        self,
        point_3d: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        distortion: Optional[np.ndarray] = None,
        width: int = 1920,
        height: int = 1080
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        将3D点投影到图像平面
        
        【注意】此函数不依赖cv2，可以单独使用
        
        Args:
            point_3d: (3,) 3D点坐标 (lidar坐标系)
            extrinsic: (4, 4) 外参矩阵 (lidar到相机)
            intrinsic: (3, 3) 内参矩阵
            distortion: (5,) 或 (4,) 畸变系数，可选
            width: 图像宽度
            height: 图像高度
        
        Returns:
            pixel: (u, v) 像素坐标，如果不在图像内返回None
            depth: 深度值
        """
        # 转换为齐次坐标
        point_homo = np.append(point_3d, 1.0)
        
        # 变换到相机坐标系
        point_cam = np.dot(np.linalg.inv(extrinsic), point_homo)[:3]
        
        # 检查是否在相机前方
        depth = point_cam[2]
        if depth <= 0:
            return None, depth
        
        # 投影到图像平面
        if HAS_CV2 and distortion is not None and len(distortion) > 0 and np.any(distortion != 0):
            # 使用OpenCV的投影函数处理畸变
            rvec = np.zeros(3)
            tvec = np.zeros(3)
            
            # projectPoints需要的是相机坐标系下的点
            projected, _ = cv2.projectPoints(
                point_cam.reshape(1, 3),
                rvec, tvec,
                intrinsic, distortion
            )
            u, v = projected[0, 0]
        else:
            # 无畸变或无cv2，直接投影 (针孔模型)
            point_norm = point_cam[:2] / point_cam[2]
            u = intrinsic[0, 0] * point_norm[0] + intrinsic[0, 2]
            v = intrinsic[1, 1] * point_norm[1] + intrinsic[1, 2]
        
        # 检查是否在图像范围内
        if 0 <= u < width and 0 <= v < height:
            return (int(u), int(v)), depth
        else:
            return None, depth
    
    def project_bbox_center(
        self,
        bbox_dict: Dict,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        distortion: Optional[np.ndarray] = None,
        width: int = 1920,
        height: int = 1080
    ) -> Tuple[Optional[Tuple[int, int]], float]:
        """
        将3D标注框中心点投影到图像
        
        Args:
            bbox_dict: 3D标注框字典
            其他参数同 project_point_to_image
        
        Returns:
            pixel: (u, v) 像素坐标
            depth: 深度值
        """
        center = np.array([
            float(bbox_dict['position']['x']),
            float(bbox_dict['position']['y']),
            float(bbox_dict['position']['z'])
        ])
        return self.project_point_to_image(
            center, extrinsic, intrinsic, distortion, width, height
        )
    
    def project_bbox_corners(
        self,
        bbox_dict: Dict,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        distortion: Optional[np.ndarray] = None,
        width: int = 1920,
        height: int = 1080
    ) -> List[Tuple[int, int]]:
        """
        将3D标注框8个角点投影到图像
        
        【坐标系约定】(与用户原代码一致)
        - l(长度)方向: -l是前(车头), +l是后(车尾)
        - w(宽度)方向: -w是左, +w是右
        - h(高度)方向: -h是下, +h是上
        
        Returns:
            corners_2d: 投影后的角点列表（仅包含在图像内的点）
        """
        from scipy.spatial.transform import Rotation as R
        
        # 解析bbox (强制转float，兼容字符串)
        cx = float(bbox_dict['position']['x'])
        cy = float(bbox_dict['position']['y'])
        cz = float(bbox_dict['position']['z'])
        
        l = float(bbox_dict['size'][2])  # 长
        w = float(bbox_dict['size'][0])  # 宽
        h = float(bbox_dict['size'][1])  # 高
        
        phi = float(bbox_dict['orientation']['phi'])
        theta = float(bbox_dict['orientation']['theta'])
        psi = float(bbox_dict['orientation']['psi'])
        
        # 局部坐标系下的8个角点 (与用户原代码get_box_point一致)
        # 坐标系: -l=前, +l=后, -w=左, +w=右, -h=下, +h=上
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
        
        # 旋转矩阵 (ZYX顺序，与用户原代码一致)
        rotation = R.from_euler('ZYX', [phi, theta, psi])
        rot_matrix = rotation.as_matrix()
        
        # 转换到世界坐标系: world = R @ local + center
        # 对于(N,3)的点矩阵，等价于 local @ R^T + center
        corners_world = np.dot(corners_local, rot_matrix.T) + np.array([cx, cy, cz])
        
        # 投影到图像
        corners_2d = []
        for corner in corners_world:
            pixel, depth = self.project_point_to_image(
                corner, extrinsic, intrinsic, distortion, width, height
            )
            if pixel is not None:
                corners_2d.append(pixel)
        
        return corners_2d
    
    def draw_visibility_label(
        self,
        img: np.ndarray,
        position: Tuple[int, int],
        track_id: str,
        score: float,
        status: str,
        style: str = 'full'
    ) -> np.ndarray:
        """
        在图像上绘制可见性标签
        
        Args:
            img: 输入图像
            position: (u, v) 标签位置
            track_id: 目标ID
            score: 可见性分数
            status: 状态
            style: 样式
                - 'full': 完整信息 (ID + 分数 + 状态)
                - 'score': 仅分数
                - 'status': 仅状态
                - 'compact': 紧凑 (分数 + 状态图标)
        
        Returns:
            绘制后的图像
        """
        _check_cv2()
        
        color = self.STATUS_COLORS.get(status, self.STATUS_COLORS['UNKNOWN'])
        u, v = position
        
        if style == 'full':
            # 完整信息：ID、分数、状态
            text = f"ID:{track_id} V:{score:.0%} [{status}]"
        elif style == 'score':
            # 仅分数
            text = f"{score:.0%}"
        elif style == 'status':
            # 仅状态（中文）
            text = self.STATUS_CN.get(status, status)
        elif style == 'compact':
            # 紧凑：分数 + 状态缩写
            status_short = status[0]  # 取首字母
            text = f"{score:.0%}{status_short}"
        else:
            text = f"{score:.2f}"
        
        # 计算文本大小
        (text_w, text_h), baseline = cv2.getTextSize(
            text, self.font, self.font_scale, self.thickness
        )
        
        # 绘制背景矩形
        padding = 4
        bg_rect = [
            (u - padding, v - text_h - padding - baseline),
            (u + text_w + padding, v + padding)
        ]
        
        # 确保在图像范围内
        bg_rect[0] = (max(0, bg_rect[0][0]), max(0, bg_rect[0][1]))
        bg_rect[1] = (min(img.shape[1], bg_rect[1][0]), min(img.shape[0], bg_rect[1][1]))
        
        # 半透明背景
        overlay = img.copy()
        cv2.rectangle(overlay, bg_rect[0], bg_rect[1], (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        # 绘制文本
        cv2.putText(
            img, text,
            (u, v),
            self.font, self.font_scale, color, self.thickness, cv2.LINE_AA
        )
        
        # 绘制中心点标记
        cv2.circle(img, (u, v + 5), 5, color, -1)
        cv2.circle(img, (u, v + 5), 5, (255, 255, 255), 1)
        
        return img
    
    def draw_bbox_wireframe(
        self,
        img: np.ndarray,
        corners_2d: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        thickness: int = 2
    ) -> np.ndarray:
        """
        绘制3D框的线框投影
        
        Args:
            img: 输入图像
            corners_2d: 投影后的角点
            color: 颜色
            thickness: 线条粗细
        
        Returns:
            绘制后的图像
        """
        _check_cv2()
        
        if len(corners_2d) < 2:
            return img
        
        # 计算凸包并绘制
        points = np.array(corners_2d, dtype=np.int32)
        hull = cv2.convexHull(points)
        cv2.polylines(img, [hull], True, color, thickness)
        
        return img
    
    def draw_visibility_on_image(
        self,
        img: np.ndarray,
        label_3d_list: List[Dict],
        visibility_results: Dict[str, Dict],
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        distortion: Optional[np.ndarray] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        style: str = 'full',
        draw_wireframe: bool = True,
        filter_camera: Optional[str] = None
    ) -> np.ndarray:
        """
        在图像上绘制所有标注框的可见性信息
        
        【主要接口函数】
        
        Args:
            img: 输入图像 (BGR格式)
            label_3d_list: 3D标注框列表
            visibility_results: 可见性计算结果 (来自compute_frame_visibility)
            extrinsic: (4, 4) 外参矩阵
            intrinsic: (3, 3) 内参矩阵
            distortion: 畸变系数
            width: 图像宽度 (默认从img获取)
            height: 图像高度 (默认从img获取)
            style: 标签样式 ('full', 'score', 'status', 'compact')
            draw_wireframe: 是否绘制3D框线框
            filter_camera: 过滤相机名称 (仅绘制包含此相机的框)
        
        Returns:
            绘制后的图像
        """
        _check_cv2()
        
        if width is None:
            width = img.shape[1]
        if height is None:
            height = img.shape[0]
        
        img = img.copy()
        
        for bbox_dict in label_3d_list:
            track_id = str(bbox_dict.get('track_id', 'unknown'))
            
            # 过滤相机
            if filter_camera is not None:
                camera_ids = bbox_dict.get('camera_ids', [])
                if filter_camera not in camera_ids:
                    continue
            
            # 获取可见性结果
            if track_id in visibility_results:
                info = visibility_results[track_id]
                score = info.get('score', 0)
                status = info.get('status', 'UNKNOWN')
            else:
                score = 0
                status = 'UNKNOWN'
            
            color = self.STATUS_COLORS.get(status, self.STATUS_COLORS['UNKNOWN'])
            
            # 投影中心点
            pixel, depth = self.project_bbox_center(
                bbox_dict, extrinsic, intrinsic, distortion, width, height
            )
            
            if pixel is None:
                continue
            
            # 绘制线框
            if draw_wireframe:
                corners_2d = self.project_bbox_corners(
                    bbox_dict, extrinsic, intrinsic, distortion, width, height
                )
                if len(corners_2d) >= 3:
                    img = self.draw_bbox_wireframe(img, corners_2d, color, self.thickness)
            
            # 绘制可见性标签
            img = self.draw_visibility_label(
                img, pixel, track_id, score, status, style
            )
        
        return img
    
    def create_visibility_legend(
        self,
        img: np.ndarray,
        position: str = 'top-left'
    ) -> np.ndarray:
        """
        在图像上添加图例
        
        Args:
            img: 输入图像
            position: 图例位置 ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        
        Returns:
            添加图例后的图像
        """
        _check_cv2()
        
        img = img.copy()
        
        legend_items = [
            ('VISIBLE',  '可见 (≥70%)'),
            ('PARTIAL',  '部分 (30-70%)'),
            ('OCCLUDED', '遮挡 (5-30%)'),
            ('BLOCKED',  '不可见 (<5%)'),
        ]
        
        # 计算图例大小
        line_height = 25
        legend_height = len(legend_items) * line_height + 20
        legend_width = 180
        
        # 确定位置
        margin = 10
        if position == 'top-left':
            x, y = margin, margin
        elif position == 'top-right':
            x, y = img.shape[1] - legend_width - margin, margin
        elif position == 'bottom-left':
            x, y = margin, img.shape[0] - legend_height - margin
        else:  # bottom-right
            x, y = img.shape[1] - legend_width - margin, img.shape[0] - legend_height - margin
        
        # 绘制背景
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + legend_width, y + legend_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # 绘制边框
        cv2.rectangle(img, (x, y), (x + legend_width, y + legend_height), (255, 255, 255), 1)
        
        # 绘制图例项
        for i, (status, label) in enumerate(legend_items):
            color = self.STATUS_COLORS[status]
            item_y = y + 15 + i * line_height
            
            # 颜色方块
            cv2.rectangle(img, (x + 10, item_y - 10), (x + 25, item_y + 5), color, -1)
            
            # 文本
            cv2.putText(
                img, label,
                (x + 35, item_y),
                self.font, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )
        
        return img


# ============================================================
# 不依赖cv2的纯投影功能
# ============================================================

def project_bboxes_to_image(
    label_3d_list: List[Dict],
    visibility_results: Dict[str, Dict],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    distortion: Optional[np.ndarray] = None,
    width: int = 1920,
    height: int = 1080,
    filter_camera: Optional[str] = None
) -> List[Dict]:
    """
    将3D标注框投影到图像平面（不依赖cv2）
    
    【用途】获取投影坐标后，可用任意方式绑定（PIL、matplotlib等）
    
    Args:
        label_3d_list: 3D标注框列表
        visibility_results: 可见性结果
        extrinsic: 外参矩阵
        intrinsic: 内参矩阵
        distortion: 畸变系数
        width: 图像宽度
        height: 图像高度
        filter_camera: 过滤相机
    
    Returns:
        投影结果列表:
        [
            {
                'track_id': 'xxx',
                'pixel': (u, v),        # 中心点像素坐标
                'depth': 15.3,          # 深度
                'score': 0.85,          # 可见性分数
                'status': 'VISIBLE',    # 状态
                'in_image': True        # 是否在图像内
            },
            ...
        ]
    """
    visualizer = VisibilityVisualizer()
    results = []
    
    for bbox_dict in label_3d_list:
        track_id = str(bbox_dict.get('track_id', 'unknown'))
        
        # 过滤相机
        if filter_camera is not None:
            camera_ids = bbox_dict.get('camera_ids', [])
            if filter_camera not in camera_ids:
                continue
        
        # 获取可见性
        if track_id in visibility_results:
            info = visibility_results[track_id]
            score = info.get('score', 0)
            status = info.get('status', 'UNKNOWN')
        else:
            score = 0
            status = 'UNKNOWN'
        
        # 投影
        pixel, depth = visualizer.project_bbox_center(
            bbox_dict, extrinsic, intrinsic, distortion, width, height
        )
        
        results.append({
            'track_id': track_id,
            'pixel': pixel,
            'depth': depth,
            'score': score,
            'status': status,
            'in_image': pixel is not None
        })
    
    return results


def print_visibility_summary(
    label_3d_list: List[Dict],
    visibility_results: Dict[str, Dict]
) -> None:
    """
    打印可见性统计摘要（不依赖cv2）
    
    Args:
        label_3d_list: 3D标注框列表
        visibility_results: 可见性结果
    """
    print("\n" + "=" * 50)
    print("可见性统计摘要")
    print("=" * 50)
    
    status_counts = {'VISIBLE': 0, 'PARTIAL': 0, 'OCCLUDED': 0, 'BLOCKED': 0, 'UNKNOWN': 0}
    
    for bbox_dict in label_3d_list:
        track_id = str(bbox_dict.get('track_id', 'unknown'))
        if track_id in visibility_results:
            info = visibility_results[track_id]
            status = info.get('status', 'UNKNOWN')
            score = info.get('score', 0)
            status_counts[status] = status_counts.get(status, 0) + 1
            print(f"  {track_id}: {score:.0%} ({status})")
        else:
            status_counts['UNKNOWN'] += 1
            print(f"  {track_id}: 未计算")
    
    print("-" * 50)
    print(f"总计: {len(label_3d_list)} 个目标")
    print(f"  可见(VISIBLE):   {status_counts['VISIBLE']}")
    print(f"  部分(PARTIAL):   {status_counts['PARTIAL']}")
    print(f"  遮挡(OCCLUDED):  {status_counts['OCCLUDED']}")
    print(f"  不可见(BLOCKED): {status_counts['BLOCKED']}")
    print("=" * 50 + "\n")


# ============================================================
# 便捷函数 (需要cv2)
# ============================================================

def visualize_frame_visibility(
    img: np.ndarray,
    label_3d_list: List[Dict],
    visibility_results: Dict[str, Dict],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    distortion: Optional[np.ndarray] = None,
    style: str = 'full',
    add_legend: bool = True
) -> np.ndarray:
    """
    一键可视化帧的可见性结果
    
    Args:
        img: 输入图像
        label_3d_list: 3D标注框列表
        visibility_results: 可见性结果
        extrinsic: 外参矩阵
        intrinsic: 内参矩阵
        distortion: 畸变系数
        style: 标签样式
        add_legend: 是否添加图例
    
    Returns:
        可视化后的图像
    """
    visualizer = VisibilityVisualizer()
    
    img = visualizer.draw_visibility_on_image(
        img, label_3d_list, visibility_results,
        extrinsic, intrinsic, distortion,
        style=style
    )
    
    if add_legend:
        img = visualizer.create_visibility_legend(img)
    
    return img


# ============================================================
# 与您代码的集成示例
# ============================================================

def integrate_with_process_frame():
    """
    展示如何在您的 process_frame 函数中集成可视化
    
    【使用方法】
    将以下代码片段添加到您的 process_frame 函数中
    """
    example_code = '''
# ========== 在 process_frame 函数中添加 ==========

from visibility_api import compute_frame_visibility
from visibility_visualizer import VisibilityVisualizer

def process_frame(label_json, target_clip_path):
    # ... 您原有的数据加载代码 ...
    
    # 计算可见性
    visibility_results = compute_frame_visibility(label_3d_lidar, filter_pcd_pts)
    
    # 初始化可视化器
    visualizer = VisibilityVisualizer()
    
    # 分视角处理
    for camera in camera_list:
        # 读取参数
        distortion, intrinsic, extrinsic, width, height = \\
            read_camera_param(params_json, camera)
        
        # 读取图像
        img = cv2.imread(camera_image_path)
        
        # 绘制可见性
        img = visualizer.draw_visibility_on_image(
            img, 
            label_3d_lidar, 
            visibility_results,
            extrinsic, 
            intrinsic, 
            distortion,
            width, 
            height,
            style='full',           # 'full', 'score', 'status', 'compact'
            draw_wireframe=True,    # 是否绘制3D框线框
            filter_camera=camera    # 只显示在该相机视野内的框
        )
        
        # 添加图例
        img = visualizer.create_visibility_legend(img, position='top-left')
        
        # 保存
        cv2.imwrite(camera_image_vis_path, img)
'''
    print(example_code)


# ============================================================
# 测试代码
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("可见性可视化工具测试")
    print("=" * 60)
    
    # 创建模拟数据
    np.random.seed(42)
    
    # 模拟相机参数
    intrinsic = np.array([
        [1000, 0, 960],
        [0, 1000, 540],
        [0, 0, 1]
    ], dtype=np.float64)
    
    extrinsic = np.eye(4, dtype=np.float64)
    # 假设相机在lidar前方0.5m，上方0.5m
    extrinsic[0, 3] = 0.5
    extrinsic[2, 3] = -0.5
    
    distortion = np.zeros(5)
    
    # 模拟3D标注框
    label_3d_list = [
        {
            'position': {'x': 15.0, 'y': 2.0, 'z': 0.5},
            'size': [2.0, 1.5, 4.5],
            'orientation': {'phi': 0.1, 'theta': 0.0, 'psi': 0.0},
            'track_id': '001',
            'camera_ids': ['front']
        },
        {
            'position': {'x': 25.0, 'y': -3.0, 'z': 0.5},
            'size': [2.0, 1.5, 4.5],
            'orientation': {'phi': -0.1, 'theta': 0.0, 'psi': 0.0},
            'track_id': '002',
            'camera_ids': ['front']
        },
        {
            'position': {'x': 40.0, 'y': 0.0, 'z': 0.5},
            'size': [2.0, 1.5, 4.5],
            'orientation': {'phi': 0.0, 'theta': 0.0, 'psi': 0.0},
            'track_id': '003',
            'camera_ids': ['front']
        },
        {
            'position': {'x': 10.0, 'y': -5.0, 'z': 0.5},
            'size': [2.0, 1.5, 4.5],
            'orientation': {'phi': 0.3, 'theta': 0.0, 'psi': 0.0},
            'track_id': '004',
            'camera_ids': ['front']
        },
    ]
    
    # 模拟可见性结果
    visibility_results = {
        '001': {'score': 0.95, 'status': 'VISIBLE'},
        '002': {'score': 0.45, 'status': 'PARTIAL'},
        '003': {'score': 0.15, 'status': 'OCCLUDED'},
        '004': {'score': 0.02, 'status': 'BLOCKED'},
    }
    
    # ========== 测试1: 不依赖cv2的投影功能 ==========
    print("\n【测试1】纯投影功能 (不依赖cv2)")
    print("-" * 40)
    
    projections = project_bboxes_to_image(
        label_3d_list, visibility_results,
        extrinsic, intrinsic, distortion,
        width=1920, height=1080
    )
    
    for proj in projections:
        if proj['in_image']:
            print(f"  {proj['track_id']}: 像素({proj['pixel'][0]}, {proj['pixel'][1]}), "
                  f"深度={proj['depth']:.1f}m, 可见性={proj['score']:.0%} ({proj['status']})")
        else:
            print(f"  {proj['track_id']}: 不在图像内")
    
    # ========== 测试2: 打印统计摘要 ==========
    print_visibility_summary(label_3d_list, visibility_results)
    
    # ========== 测试3: cv2可视化 (如果可用) ==========
    if HAS_CV2:
        print("\n【测试2】cv2可视化")
        print("-" * 40)
        
        # 模拟图像
        img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        img[:] = (50, 50, 50)  # 深灰色背景
        
        visualizer = VisibilityVisualizer()
        
        img = visualizer.draw_visibility_on_image(
            img, label_3d_list, visibility_results,
            extrinsic, intrinsic, distortion,
            style='full',
            draw_wireframe=True
        )
        
        img = visualizer.create_visibility_legend(img, position='top-left')
        
        # 保存
        output_path = '/workspace/visibility_demo.jpg'
        cv2.imwrite(output_path, img)
        print(f"演示图像已保存到: {output_path}")
    else:
        print("\n【测试2】cv2可视化 - 跳过 (cv2不可用)")
        print("-" * 40)
        print("提示: 安装cv2后可使用完整可视化功能")
        print("  pip install opencv-python-headless==4.5.5.64")
    
    # 显示集成示例
    print("\n" + "=" * 60)
    print("集成示例代码:")
    print("=" * 60)
    integrate_with_process_frame()

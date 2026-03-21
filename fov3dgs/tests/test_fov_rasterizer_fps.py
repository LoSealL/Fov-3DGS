#!/usr/bin/env python
"""
单元测试：测试 diff_gaussian_rasterization_fov_pcheck_obb 的 GaussianRasterizer
使用静态构造的数据，不依赖外部文件
"""

import math
import os
import sys
import time

import torch
from loguru import logger

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diff_gaussian_rasterization_fov_pcheck_obb import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)


def create_mock_camera(image_height=822, image_width=1237, fov_degrees=90):
    """创建模拟相机参数

    默认使用 bicycle 数据集 images_4 的分辨率：1237x822
    """
    fov_rad = math.radians(fov_degrees)
    tanfovx = math.tan(fov_rad * 0.5)
    tanfovy = math.tan(fov_rad * 0.5 * image_height / image_width)

    # 简化的视图矩阵（单位矩阵）
    viewmatrix = torch.eye(4, dtype=torch.float32, device="cuda")
    viewmatrix[2, 3] = 2.0  # 相机沿 Z 轴后移 2 米

    # 投影矩阵
    projmatrix = viewmatrix.clone()

    # 相机中心位置
    camera_center = torch.tensor([0.0, 0.0, -2.0], dtype=torch.float32, device="cuda")

    return {
        "image_height": image_height,
        "image_width": image_width,
        "tanfovx": tanfovx,
        "tanfovy": tanfovy,
        "viewmatrix": viewmatrix,
        "projmatrix": projmatrix,
        "camera_center": camera_center,
    }


def create_mock_gaussians(num_points=10000, sh_degree=3):
    """创建模拟高斯点数据"""
    device = "cuda"

    # 随机生成 3D 位置（在相机前方 2 米附近）
    means3D = torch.randn(num_points, 3, dtype=torch.float32, device=device) * 0.5
    means3D[:, 2] += 2.0  # 沿 Z 轴偏移

    # 缩放（随机大小）
    scales = torch.rand(num_points, 3, dtype=torch.float32, device=device) * 0.1 + 0.01

    # 旋转（四元数，归一化）
    rotations = torch.randn(num_points, 4, dtype=torch.float32, device=device)
    rotations = torch.nn.functional.normalize(rotations, dim=1)

    # 不透明度（降低以避免过亮）
    opacities = torch.rand(num_points, 1, dtype=torch.float32, device=device) * 0.3 + 0.1

    # SH 系数（简化，只使用 DC 分量，值较小）
    num_sh_coeffs = 1
    shs_rest = torch.randn(num_points, num_sh_coeffs, 3, dtype=torch.float32, device=device) * 0.1

    # 多层数据（用于 FOV 感知渲染）
    num_layers = 4
    shs_dcs = torch.randn(num_points, num_layers, 3, dtype=torch.float32, device=device)
    highest_levels = torch.randint(0, num_layers, (num_points, 1), dtype=torch.float32, device=device)

    return {
        "means3D": means3D,
        "scales": scales,
        "rotations": rotations,
        "opacities": opacities,
        "shs_rest": shs_rest,
        "shs_dcs": shs_dcs,
        "highest_levels": highest_levels,
    }


def create_raster_settings(camera):
    """创建光栅化设置"""
    return GaussianRasterizationSettings(
        image_height=camera["image_height"],
        image_width=camera["image_width"],
        tanfovx=camera["tanfovx"],
        tanfovy=camera["tanfovy"],
        bg=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=camera["viewmatrix"],
        projmatrix=camera["projmatrix"],
        sh_degree=3,
        campos=camera["camera_center"],
        prefiltered=False,
        debug=False,
    )


def test_rasterizer_forward():
    """测试光栅化器前向传播"""
    logger.info("=" * 60)
    logger.info("测试 1: GaussianRasterizer 前向传播")
    logger.info("=" * 60)

    # 创建模拟数据
    camera = create_mock_camera(image_height=512, image_width=512)
    gaussians = create_mock_gaussians(num_points=5000)
    raster_settings = create_raster_settings(camera)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 创建 means2D（屏幕空间位置，初始化为 0）
    means2D = torch.zeros_like(gaussians["means3D"], requires_grad=True)

    # 测试不同的 gaze 位置
    gaze_positions = [
        torch.tensor([0.5, 0.5], dtype=torch.float32, device="cuda"),
        torch.tensor([0.25, 0.25], dtype=torch.float32, device="cuda"),
        torch.tensor([0.75, 0.75], dtype=torch.float32, device="cuda"),
    ]

    for i, gaze in enumerate(gaze_positions):
        logger.info(f"\n测试 gaze 位置：{gaze.cpu().tolist()}")

        # 前向传播
        rendered_image, radii = rasterizer(
            means3D=gaussians["means3D"],
            means2D=means2D,
            shs_rest=gaussians["shs_rest"],
            colors_precomp=None,
            opacities=gaussians["opacities"],
            scales=gaussians["scales"],
            rotations=gaussians["rotations"],
            cov3D_precomp=None,
            shs_dcs=gaussians["shs_dcs"],
            highest_levels=gaussians["highest_levels"],
            gazeArray=gaze,
            alpha=0.05,
            blending=True,
        )

        logger.info(f"  输出图像形状：{rendered_image.shape}")
        logger.info(f"  输出值范围：[{rendered_image.min():.4f}, {rendered_image.max():.4f}]")
        logger.info(f"  可见高斯点数：{(radii > 0).sum().item()}")

        # 验证输出
        assert rendered_image.shape == (3, 512, 512), "输出图像形状错误"
        assert torch.isfinite(rendered_image).all(), "输出包含非有限值"
        assert rendered_image.min() >= 0, "输出值不应小于 0"

    logger.info("\n✓ 前向传播测试通过")
    return True


def test_rasterizer_fps():
    """测试光栅化器 FPS 性能"""
    logger.info("=" * 60)
    logger.info("测试 2: GaussianRasterizer FPS 性能测试")
    logger.info("=" * 60)

    # 创建模拟数据（使用 bicycle 数据集的真实配置）
    camera = create_mock_camera(image_height=1080, image_width=1920)  # images_4 分辨率
    gaussians = create_mock_gaussians(num_points=50000)  # ~600 万点（与 bicycle 实际数量相当）
    raster_settings = create_raster_settings(camera)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = torch.zeros_like(gaussians["means3D"], requires_grad=True)
    gaze = torch.tensor([0.5, 0.5], dtype=torch.float32, device="cuda")

    # 预热
    logger.info("\n预热阶段 (10 次迭代)...")
    for _ in range(10):
        rendered_image, radii = rasterizer(
            means3D=gaussians["means3D"],
            means2D=means2D,
            shs_rest=gaussians["shs_rest"],
            colors_precomp=None,
            opacities=gaussians["opacities"],
            scales=gaussians["scales"],
            rotations=gaussians["rotations"],
            cov3D_precomp=None,
            shs_dcs=gaussians["shs_dcs"],
            highest_levels=gaussians["highest_levels"],
            gazeArray=gaze,
            alpha=0.05,
            blending=True,
        )
        torch.cuda.synchronize()

    # FPS 测试
    num_frames = 100
    logger.info(f"\n性能测试阶段 ({num_frames} 次迭代)...")

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    total_time = 0
    fps_list = []

    for i in range(num_frames):
        starter.record()
        rendered_image, radii = rasterizer(
            means3D=gaussians["means3D"],
            means2D=means2D,
            shs_rest=gaussians["shs_rest"],
            colors_precomp=None,
            opacities=gaussians["opacities"],
            scales=gaussians["scales"],
            rotations=gaussians["rotations"],
            cov3D_precomp=None,
            shs_dcs=gaussians["shs_dcs"],
            highest_levels=gaussians["highest_levels"],
            gazeArray=gaze,
            alpha=0.05,
            blending=True,
        )
        ender.record()
        torch.cuda.synchronize()

        frame_time = starter.elapsed_time(ender)
        fps = 1000.0 / frame_time if frame_time > 0 else float("inf")
        fps_list.append(fps)
        total_time += frame_time

        if (i + 1) % 20 == 0:
            logger.info(f"  帧 {i + 1}/{num_frames}: {frame_time:.2f}ms, FPS: {fps:.1f}")

    avg_fps = num_frames / (total_time / 1000)
    min_fps = min(fps_list)
    max_fps = max(fps_list)

    logger.info("\n" + "=" * 60)
    logger.info(f"性能测试结果:")
    logger.info(f"  分辨率：{camera['image_width']}x{camera['image_height']} (bicycle images_4)")
    logger.info(f"  高斯点数：{gaussians['means3D'].shape[0]:,} ({gaussians['means3D'].shape[0]/1e6:.1f}M)")
    logger.info(f"  平均 FPS: {avg_fps:.1f}")
    logger.info(f"  最小 FPS: {min_fps:.1f}")
    logger.info(f"  最大 FPS: {max_fps:.1f}")
    logger.info(f"  平均帧时间：{total_time / num_frames:.2f}ms")
    logger.info("=" * 60)

    return avg_fps


def test_rasterizer_blending_vs_non_blending():
    """测试 blending 和非 blending 模式的差异"""
    logger.info("=" * 60)
    logger.info("测试 3: Blending vs Non-Blending 模式对比")
    logger.info("=" * 60)

    camera = create_mock_camera(image_height=512, image_width=512)
    gaussians = create_mock_gaussians(num_points=5000)
    raster_settings = create_raster_settings(camera)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = torch.zeros_like(gaussians["means3D"], requires_grad=True)
    gaze = torch.tensor([0.5, 0.5], dtype=torch.float32, device="cuda")

    # Blending 模式
    logger.info("\nBlending 模式...")
    rendered_blending, radii_blending = rasterizer(
        means3D=gaussians["means3D"],
        means2D=means2D,
        shs_rest=gaussians["shs_rest"],
        colors_precomp=None,
        opacities=gaussians["opacities"],
        scales=gaussians["scales"],
        rotations=gaussians["rotations"],
        cov3D_precomp=None,
        shs_dcs=gaussians["shs_dcs"],
        highest_levels=gaussians["highest_levels"],
        gazeArray=gaze,
        alpha=0.05,
        blending=True,
    )

    # Non-Blending 模式
    logger.info("Non-Blending 模式...")
    rendered_normal, radii_normal = rasterizer(
        means3D=gaussians["means3D"],
        means2D=means2D,
        shs_rest=gaussians["shs_rest"],
        colors_precomp=None,
        opacities=gaussians["opacities"],
        scales=gaussians["scales"],
        rotations=gaussians["rotations"],
        cov3D_precomp=None,
        shs_dcs=gaussians["shs_dcs"],
        highest_levels=gaussians["highest_levels"],
        gazeArray=gaze,
        alpha=0.05,
        blending=False,
    )

    logger.info(f"\nBlending 模式:")
    logger.info(f"  图像均值：{rendered_blending.mean():.4f}")
    logger.info(f"  可见点数：{(radii_blending > 0).sum().item()}")

    logger.info(f"\nNon-Blending 模式:")
    logger.info(f"  图像均值：{rendered_normal.mean():.4f}")
    logger.info(f"  可见点数：{(radii_normal > 0).sum().item()}")

    logger.info(f"\n差异:")
    logger.info(f"  图像差异均值：{(rendered_blending - rendered_normal).abs().mean():.6f}")

    logger.info("\n✓ Blending 模式对比测试通过")
    return True


def test_different_gaze_positions():
    """测试不同注视点位置的影响"""
    logger.info("=" * 60)
    logger.info("测试 4: 不同注视点位置对比")
    logger.info("=" * 60)

    camera = create_mock_camera(image_height=512, image_width=512)
    gaussians = create_mock_gaussians(num_points=10000)
    raster_settings = create_raster_settings(camera)
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = torch.zeros_like(gaussians["means3D"], requires_grad=True)

    # 9 个注视点位置
    gaze_positions = [
        (0.25, 0.25),
        (0.25, 0.50),
        (0.25, 0.75),
        (0.50, 0.25),
        (0.50, 0.50),
        (0.50, 0.75),
        (0.75, 0.25),
        (0.75, 0.50),
        (0.75, 0.75),
    ]

    results = []
    for gx, gy in gaze_positions:
        gaze = torch.tensor([gx, gy], dtype=torch.float32, device="cuda")

        rendered_image, radii = rasterizer(
            means3D=gaussians["means3D"],
            means2D=means2D,
            shs_rest=gaussians["shs_rest"],
            colors_precomp=None,
            opacities=gaussians["opacities"],
            scales=gaussians["scales"],
            rotations=gaussians["rotations"],
            cov3D_precomp=None,
            shs_dcs=gaussians["shs_dcs"],
            highest_levels=gaussians["highest_levels"],
            gazeArray=gaze,
            alpha=0.05,
            blending=True,
        )

        results.append(
            {
                "gaze": (gx, gy),
                "mean": rendered_image.mean().item(),
                "std": rendered_image.std().item(),
                "visible": (radii > 0).sum().item(),
            }
        )

        logger.info(f"  Gaze ({gx:.2f}, {gy:.2f}): mean={results[-1]['mean']:.4f}, visible={results[-1]['visible']}")

    logger.info("\n✓ 不同注视点位置测试通过")
    return results


def main():
    """运行所有测试"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO",
    )

    logger.info("FOV Gaussian Rasterizer 单元测试")
    logger.info("=" * 60)

    # 检查 CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA 不可用！")
        return False

    logger.info(f"CUDA 设备：{torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA 版本：{torch.version.cuda}")
    logger.info("")

    try:
        # 测试 1: 前向传播
        test_rasterizer_forward()
        logger.info("")

        # 测试 2: FPS 性能
        avg_fps = test_rasterizer_fps()
        logger.info("")

        # 测试 3: Blending 对比
        test_rasterizer_blending_vs_non_blending()
        logger.info("")

        # 测试 4: 不同注视点
        test_different_gaze_positions()
        logger.info("")

        logger.info("=" * 60)
        logger.info("所有测试通过！✓")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"测试失败：{e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

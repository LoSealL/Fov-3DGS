#!/usr/bin/env python
"""测试 render_compose_gazes_fps.py 的核心功能 - 使用简单模型测试 FPS"""

import os

import torch
from loguru import logger

from gaussian_renderer_fov import GaussianModel, render
from scene import Scene


def test_fps_render(scene_dir, model_dir, iteration):
    """使用简单模型测试 FPS 渲染"""
    logger.info(f"Testing FPS render with model: {model_dir}")

    with torch.no_grad():
        # Load model - directly set attributes without argparse
        from argparse import Namespace

        gaussians = GaussianModel(3)  # sh_degree=3

        # Create a minimal args object
        args = Namespace(
            model_path=model_dir,
            source_path=scene_dir,
            images="images",
            eval=True,
            resolution=-1,
            white_background=False,
            data_device="cuda",
            sh_degree=3,
        )

        scene = Scene(
            args, gaussians, load_iteration=iteration, shuffle=False, fps_mode=True
        )

        # Load point cloud
        ply_path = os.path.join(
            model_dir, f"point_cloud/iteration_{iteration}/point_cloud.ply"
        )
        gaussians.load_ply(ply_path)
        logger.info(f"Loaded {gaussians.get_xyz.shape[0]} points")

        # Setup background
        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Get test cameras
        views = scene.getTestCameras()
        logger.info(f"Test cameras: {len(views)}")

        # Gaze samples
        gaze_samples = [(0.25 * i, 0.25 * j) for i in range(1, 4) for j in range(1, 4)]
        logger.info(f"Gaze samples: {gaze_samples}")

        # Warmup
        starter, ender = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        view = views[0]
        gazeArray = torch.tensor([0.5, 0.5]).float().cuda()

        logger.info("Warming up...")
        for i in range(3):
            rendering = render(
                view,
                gaussians,
                background,
                alpha=0.05,
                gazeArray=gazeArray,
                blending=False,
                starter=starter,
                ender=ender,
            )["render"]
            torch.cuda.synchronize()

        # Test one gaze
        logger.info("\nTesting gaze (0.5, 0.5)...")
        time = 0
        for i in range(5):
            rendering = render(
                view,
                gaussians,
                background,
                alpha=0.05,
                gazeArray=gazeArray,
                blending=False,
                starter=starter,
                ender=ender,
            )["render"]
            torch.cuda.synchronize()
            time += starter.elapsed_time(ender)
            logger.info(f"  Frame {i + 1}: {starter.elapsed_time(ender):.2f}ms")

        avg_time = time / 5
        fps = 5 / (time / 1000)
        logger.info(f"\nAverage time: {avg_time:.2f}ms")
        logger.info(f"FPS: {fps:.2f}")

        logger.info("\nTest completed successfully!")
        return fps


if __name__ == "__main__":
    import sys

    # Default test with ms_d model
    scene_dir = r"D:\Works\third-party\Fov-3DGS\dataset\flowers"
    model_dir = r"D:\Works\third-party\Fov-3DGS\dataset\flowers\ms_d"
    iteration = 30000

    if len(sys.argv) > 1:
        scene_dir = sys.argv[1]
    if len(sys.argv) > 2:
        model_dir = sys.argv[2]
    if len(sys.argv) > 3:
        iteration = int(sys.argv[3])

    test_fps_render(scene_dir, model_dir, iteration)

# use argparser
import argparse
import json
import os
import subprocess
import sys

from loguru import logger


# Example usage within a larger script context
def main():
    # take arguments, scene, target_loss, initial_pretrain_ply, max_pooling_size from command line, layernum from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, help="Scene path")
    # target loss
    parser.add_argument(
        "--relax_ratio",
        type=float,
        help="Target loss relaxation ratio (100% + / - relax_ratio)",
    )
    # path of pretrain trial
    parser.add_argument("--path_to_pretrain_trial", type=str, default=None)
    # pruning settings
    parser.add_argument("--pruning_metric", type=str, default="max_comp_efficiency")
    # FOV parameters
    parser.add_argument(
        "--iterations", type=int, help="Budget for pruning, total iteration number"
    )

    parser.add_argument("--monitor_val", action="store_true", default=False)

    parser.add_argument("--images_set", type=str, default="")

    parser.add_argument("--use_scale_decay", action="store_true", default=False)

    args = parser.parse_args()

    if args.monitor_val:
        args.original_train_loss = f"{args.path_to_pretrain_trial}/test_results.json"
    else:
        args.original_train_loss = f"{args.path_to_pretrain_trial}/train_results.json"

    args.pretrained_pc = f"{args.path_to_pretrain_trial}/point_cloud/iteration_{args.iterations}/point_cloud.ply"

    # check if the original training loss txt exists
    if not os.path.exists(args.original_train_loss):
        if args.monitor_val:
            cmd = f"python render.py -s {args.scene} -m {args.path_to_pretrain_trial}  --eval --iteration {args.iterations} --skip_train"
        else:
            cmd = f"python render.py -s {args.scene} -m {args.path_to_pretrain_trial}  --eval --iteration {args.iterations} --skip_test"
        logger.info(
            "Original training loss json not found, running render.py to generate images."
        )
        logger.info(cmd)
        subprocess.run(cmd, shell=True)
        logger.info(
            "Original training loss json not found, running hvs_metric to generate it."
        )
        if args.monitor_val:
            cmd = f"python hvs_metrics.py -m {args.path_to_pretrain_trial} -s test"
        else:
            cmd = f"python hvs_metrics.py -m {args.path_to_pretrain_trial} -s train"
        subprocess.run(cmd, shell=True)

    # find the "Average L1 HVS Loss" in the text
    with open(args.original_train_loss, "r") as file:
        text = file.read()
        data = json.loads(text)
    hvs_uniform_value = data[f"ours_{args.iterations}"]["HVS Uniform"]
    ssim = data[f"ours_{args.iterations}"]["SSIM"]
    psnr = data[f"ours_{args.iterations}"]["PSNR"]

    # print
    logger.info(f"Average HVS Loss: {hvs_uniform_value}")
    logger.info(f"Average SSIM: {ssim}")

    args.target_hvs = float(hvs_uniform_value) * (1 + args.relax_ratio)
    args.target_ssim = float(ssim) * (1 - args.relax_ratio)
    args.target_psnr = float(psnr) * (1 - args.relax_ratio)

    # prune_iterations = round(args.iterations * 0.9)
    # adaptation_iters = args.iterations - prune_iterations
    prune_iterations = 1
    adaptation_iters = 1

    base_command = [
        "",
        "prune.py",
        "-s",
        args.scene,
        "-i",
        args.images_set,
        "-m",
        args.scene,
        "--eval",
        "--pretrain_ply",
        args.pretrained_pc,
        "--pooling_size",
        "1",
        "--target_hvs",
        str(args.target_hvs),
        "--target_ssim",
        str(args.target_ssim),
        "--target_psnr",
        str(args.target_psnr),
        "--pruning_iters",
        str(prune_iterations),
        "--final_adaptation_iters",
        str(adaptation_iters),
        "--trial_name",
        f"mon2_montest={args.monitor_val}_sd={args.use_scale_decay}_{args.pruning_metric}_{args.relax_ratio}",
        "--position_lr_init_scale",
        "0.1",
        "--metric",
        args.pruning_metric,
    ]
    additional_flags = []
    if args.monitor_val:
        additional_flags.append("--monitor_val")

    if args.use_scale_decay:
        additional_flags.append("--use_scale_decay")

    # Complete command with additional flags
    command = base_command + additional_flags
    logger.info(f"Running command: {' '.join(command)}")
    result = subprocess.check_output(command, executable=sys.executable)
    logger.info(result.decode("utf-8"))


if __name__ == "__main__":
    main()

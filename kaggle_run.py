# kaggle_run.py
import argparse
import os
import sys
import subprocess

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    env = os.environ.copy()
    
    # Set PYTHONPATH to include project root and task directories
    project_root = os.path.dirname(os.path.abspath(__file__))
    detection_dir = os.path.join(project_root, "detection")
    segmentation_dir = os.path.join(project_root, "segmentation")
    
    existing_pythonpath = env.get("PYTHONPATH", "")
    new_pythonpaths = [project_root, detection_dir, segmentation_dir]
    if existing_pythonpath:
        new_pythonpaths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(new_pythonpaths)
    
    # Required for Kaggle T4 (no NVLink / InfiniBand) — prevents NCCL hangs
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env
    )
    
    # Stream output in real-time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    rc = process.poll()
    return rc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle execution entrypoint helper")
    parser.add_argument("--action", type=str, required=True,
                        choices=["train_clf", "eval_clf", "train_det", "test_det", "train_seg", "test_seg", "robust_clf"],
                        help="Action to perform")
    parser.add_argument("--model", type=str, default="lsnet_t", help="Model name")
    parser.add_argument("--config", type=str, help="Path to MMCV/MMDet/MMSeg config file")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume training from")
    parser.add_argument("--extra-args", type=str, default="", help="Extra arguments to pass to the script")
    
    args, unknown = parser.parse_known_args()
    
    cmd_parts = []
    
    if args.action == "train_clf":
        cmd_parts = [f"torchrun --nproc_per_node=2 main.py --model {args.model}"]
        if args.resume:
            cmd_parts.append(f"--resume {args.resume}")
        if args.extra_args:
            cmd_parts.append(args.extra_args)
            
    elif args.action == "eval_clf":
        cmd_parts = [f"torchrun --nproc_per_node=2 main.py --eval --model {args.model}"]
        if args.checkpoint:
            cmd_parts.append(f"--resume {args.checkpoint}")
        if args.extra_args:
            cmd_parts.append(args.extra_args)
            
    elif args.action == "robust_clf":
        cmd_parts = [f"torchrun --nproc_per_node=2 main.py --eval --model {args.model}"]
        if args.checkpoint:
            cmd_parts.append(f"--resume {args.checkpoint}")
        if args.extra_args:
            cmd_parts.append(args.extra_args)

    elif args.action == "train_det":
        if not args.config:
            print("Error: --config is required for train_det")
            sys.exit(1)
        cmd_parts = [f"python detection/train.py {args.config}"]
        if args.resume:
            cmd_parts.append(f"--resume-from {args.resume}")
        if args.extra_args:
            cmd_parts.append(args.extra_args)
            
    elif args.action == "test_det":
        if not args.config or not args.checkpoint:
            print("Error: --config and --checkpoint are required for test_det")
            sys.exit(1)
        cmd_parts = [f"python detection/test.py {args.config} {args.checkpoint}"]
        if args.extra_args:
            cmd_parts.append(args.extra_args)
            
    elif args.action == "train_seg":
        if not args.config:
            print("Error: --config is required for train_seg")
            sys.exit(1)
        cmd_parts = [f"python segmentation/tools/train.py {args.config}"]
        if args.resume:
            cmd_parts.append(f"--resume-from {args.resume}")
        if args.extra_args:
            cmd_parts.append(args.extra_args)
            
    elif args.action == "test_seg":
        if not args.config or not args.checkpoint:
            print("Error: --config and --checkpoint are required for test_seg")
            sys.exit(1)
        cmd_parts = [f"python segmentation/tools/test.py {args.config} {args.checkpoint}"]
        if args.extra_args:
            cmd_parts.append(args.extra_args)

    if unknown:
        cmd_parts.append(" ".join(unknown))
        
    full_cmd = " ".join(cmd_parts)
    rc = run_cmd(full_cmd)
    sys.exit(rc)

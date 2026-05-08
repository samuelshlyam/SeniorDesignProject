import subprocess
import os
import torch

MODELS = ["Heston", "Bates", "Bergomi", "rBergomi"]
DATE_RANGES = ["2010-2012", "2013-2015", "2016-2019", "2020-2022"]

def chunk(lst, n):
    """Split a list into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i,m):(i+1)*k + min(i+1,m)] for i in range(n)]

num_gpus = torch.cuda.device_count()
print(f"Detected {num_gpus} GPU(s)")

if num_gpus == 0:
    # No GPU, just run normally
    subprocess.run(["python", "train.py"])

elif num_gpus == 1:
    # Run everything on the single GPU
    subprocess.run(["python", "train.py"])

elif num_gpus == 2:
    # Split models across 2 GPUs: 2 models each
    splits = chunk(MODELS, 2)
    runs = [{"gpu": i, "models": splits[i]} for i in range(2)]

elif num_gpus >= 4:
    # One model per GPU
    runs = [{"gpu": i, "models": [MODELS[i]]} for i in range(4)]

if num_gpus >= 2:
    processes = []
    for run in runs:
        cmd = ["python", "train.py", "--models"] + run["models"]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(run["gpu"])}
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
        print(f"Launched PID {p.pid} on GPU {run['gpu']}: models={run['models']}")

    for p in processes:
        p.wait()

print("All training runs complete.")
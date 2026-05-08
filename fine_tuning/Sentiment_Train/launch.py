import subprocess
import os
import torch

MODELS = ["Heston", "Bates", "Bergomi", "rBergomi"]
DATE_RANGES = ["2010-2012", "2013-2015", "2016-2018", "2020-2022"]

def chunk(lst, n):
    """Split a list into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i,m):(i+1)*k + min(i+1,m)] for i in range(n)]

num_gpus = torch.cuda.device_count()
print(f"Detected {num_gpus} GPU(s)")

if num_gpus == 0:
    subprocess.run(["python", "train.py"])

elif num_gpus == 1:
    subprocess.run(["python", "train.py"])

elif num_gpus == 2:
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

        log_file = open(f"training_gpu{run['gpu']}_{run['models'][0]}.log", "w")
        p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
        processes.append((p, log_file))
        print(f"Launched PID {p.pid} on GPU {run['gpu']}: models={run['models']}")

    for p, log_file in processes:
        p.wait()
        log_file.close()

print("All training runs complete.")

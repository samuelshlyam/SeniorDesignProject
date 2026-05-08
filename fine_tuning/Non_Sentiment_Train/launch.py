import subprocess
import os
import torch
from pathlib import Path

# Get Fine_Tuning_Files if they don't already exist
FINE_TUNING_DIR = Path(__file__).parent.parent / "Fine_Tuning_Files"

if not FINE_TUNING_DIR.exists():
    import zipfile
    GDRIVE_FILE_ID = "1hMUE2asSAVtb_JO9--i3UmpJ_jxFN_wu"  # from the shareable link
    zip_path = FINE_TUNING_DIR.parent / "Fine_Tuning_Files.zip"

    print("Fine_Tuning_Files not found — downloading from Google Drive...")
    subprocess.run(["pip", "install", "-q", "gdown"], check=True)
    subprocess.run(["python", "-m", "gdown", "--id", GDRIVE_FILE_ID, "-O", str(zip_path)], check=True)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(FINE_TUNING_DIR.parent)
    zip_path.unlink()
    print("Done.")

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
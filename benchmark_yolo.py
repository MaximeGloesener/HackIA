import torch 
from ultralytics import YOLO
import numpy as np 
import time 
from tqdm import tqdm 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# base model full precision non pruned
model_weight = 'models/best.pt'

base_model = YOLO(model_weight, task="detect")

# export to tensorrt
# base_model.export(format="engine", device=device, half=True, imgsz=224)

# quantized model
engine_file = "models/bestfp16_224.engine"
quantized_model = YOLO(engine_file, task="detect")


# input data 
input_data = np.random.rand(224, 224, 3).astype(np.float32) 

# Warmup runs for gpu
elapsed = 0.0
for _ in range(3):
    start_time = time.time()
    for _ in range(20):
        quantized_model(input_data, imgsz=224, verbose=False)
    elapsed = time.time() - start_time

eps = 1e-3
# Compute number of runs as higher of min_time or num_timed_runs
num_runs = max(round(60 / (elapsed + eps) * 20), 100 * 50)

# Timed runs
run_times = []
for _ in tqdm(range(num_runs)):
    results = base_model(input_data, imgsz=224, verbose=False)
    run_times.append(results[0].speed["inference"])  # Convert to milliseconds

# Compute statistics
run_times = np.array(run_times)
mean_time = np.mean(run_times)
std_time = np.std(run_times)
print(f"Mean inference time: {mean_time:.2f} ms")
print(f'FPS GPU: {1000/mean_time:.2f}')

import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import subprocess
import threading
import time
import os

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

batch_size = 64
num_classes = 1000
inputs = torch.randn(batch_size, 3, 224, 224).to(device)
labels = torch.randint(0, num_classes, (batch_size,)).to(device)

# GPU Logging
log_file = "gpu_usage_log.txt"
keep_logging = True

def log_gpu_usage(interval=1):
    with open(log_file, "w") as f:
        while keep_logging:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,power.draw",
                 "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            f.write(f"{time.time():.2f}, {output.stdout.strip()}\n")
            time.sleep(interval)

# Start GPU logging in the background
log_thread = threading.Thread(target=log_gpu_usage, daemon=True)
log_thread.start()

# Warm-up
for _ in range(5):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Benchmark
start_time = time.time()
for _ in range(20):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
end_time = time.time()

# Stop logging
keep_logging = False
log_thread.join()

# Output
total_time = end_time - start_time
total_images = batch_size * 20
print(f"\nBenchmark completed!")
print(f"Time taken: {total_time:.2f} seconds")
print(f"Throughput: {total_images / total_time:.2f} images/second")
print(f"GPU log saved to: {os.path.abspath(log_file)}")

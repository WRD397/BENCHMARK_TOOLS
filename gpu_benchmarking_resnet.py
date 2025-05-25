import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import subprocess
import threading
import time
import os


def setup_logger(fileHandling:bool=False, logFileName:str=None):
    """
    fileHandling : deault is False, if True, it demands to have a logFileName as well to store the logs in the file instead of on streaamhandler.
    logFileName : default is None. if fileHandling is True, need to pass the file name here. It will create a folder in the curr dir. named log and store the log on the file passed here.

    returns logger.
    """

    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    customFormatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(name)s :: [%(processName)s|%(threadName)s] :: %(message)s')

    if fileHandling : fileHandler = logging.FileHandler(f'./log/{logFileName}')
    else : 
        if logger.hasHandlers() : logger.handlers.clear()
        fileHandler = logging.StreamHandler()

    fileHandler.setFormatter(customFormatter)
    logger.addHandler(fileHandler)

    return logger

logger = setup_logger(fileHandling=True, logFileName='torch_gpu_benchmarking.log')


# Setup
logger.info(f'\n\n========= device info. ==========')
logger.info(f'cuda available : {torch.cuda.is_available()}')
logger.info(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = models.resnet18().to(device)
## if you want to put more stress on GPU memory utilization, use bigger models
model = models.resnet50().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#batch_size = 32
batch_size = 64
#batch_size = 128 ## if you want to put more stress on GPU memory utilization, use higher batch sizes like 128, 512...
num_classes = 1000

resolution = 224
inputs = torch.randn(batch_size, 3, resolution, resolution).to(device)
## if you want to put more stress on GPU memory utilization, use higher resolution images
## inputs = torch.randn(batch_size, 3, 512, 512).to(device)

labels = torch.randint(0, num_classes, (batch_size,)).to(device)

# GPU Logging
log_file = "gpu_usage_log_resnet.txt"
keep_logging = True
logger.info("benchmark logging start.")
logger.info(f"GPU Benchmark log will be saved to: {os.path.abspath(log_file)}")

logger.info(f'MODEL : {model}')
logger.info(f'RESOLUTION : {resolution}')
logger.info(f'BATCH SIZE : {batch_size}')

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
logger.info("Warm up starts")
try :
    for _ in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
except Exception as e:
    logger.error(f'********* warm up could not be completed due to below error.')
    logger.error(e)
    torch.cuda.empty_cache()
    raise Exception(f'Error.')

logger.info("Memory usage after warm-up:")
logger.info(f'{torch.cuda.memory_allocated() / 1024 ** 2}, MB allocated')
logger.info(f'{torch.cuda.memory_reserved() / 1024 ** 2}, MB reserved')

# Benchmark
logger.info("Benchmarking starts")
try :
    start_time = time.time()
    for _ in range(20):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    end_time = time.time()
except Exception as e:
    logger.error(f'********* benchmark could not be completed due to below error.')
    logger.error(e)
    torch.cuda.empty_cache()
    raise Exception(f'Error.')

# Log memory after training step
logger.info("Memory usage after complete benchmarking:")
logger.info(f'{torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB allocated')
logger.info(f'{torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB reserved')

# Stop logging
keep_logging = False
log_thread.join()
logger.info("benchmark logging stops.")

# Output
total_time = end_time - start_time
total_images = batch_size * 20
logger.info(f"\nBenchmark completed!")
logger.info(f"Time taken: {total_time:.2f} seconds")
logger.info(f"Throughput: {total_images / total_time:.2f} images/second")
logger.info(f"GPU Benchmark log saved to: {os.path.abspath(log_file)}")

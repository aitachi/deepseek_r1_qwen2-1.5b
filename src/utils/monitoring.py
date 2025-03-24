import psutil
import GPUtil
import torch
import time
import threading

def print_resource_usage():
    """打印当前系统资源使用情况（CPU、内存、GPU）。"""
    # CPU 使用率
    cpu_percent = psutil.cpu_percent()
    # 内存使用情况
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / (1024 ** 3)
    memory_total_gb = memory.total / (1024 ** 3)

    # GPU 信息
    gpu_info = ""
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                gpu_info += f"\nGPU {i}: {gpu.name}"
                gpu_info += f"\n  - 显存占用: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil * 100:.1f}%)"
                gpu_info += f"\n  - GPU利用率: {gpu.load * 100:.1f}%"
        except:
            gpu_info = "\nGPU信息获取失败"
    else:
        gpu_info = "\n未检测到GPU"

    print(f"系统资源使用情况:")
    print(f"CPU使用率: {cpu_percent}%")
    print(f"内存使用: {memory_used_gb:.2f}GB / {memory_total_gb:.2f}GB ({memory.percent}%)")
    print(gpu_info)
    print("-" * 50)


def resource_monitor_thread(interval=60, stop_event=None):
    """定期监控资源使用情况的线程函数。"""
    while not stop_event.is_set():
        print_resource_usage()
        time.sleep(interval)


def check_cuda():
    """验证CUDA可用性并打印设备信息。"""
    print("CUDA是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA版本:", torch.version.cuda)
        print("可用的GPU数量:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
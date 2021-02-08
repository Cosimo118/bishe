import pynvml
import os
import time

pynvml.nvmlInit()
# 这里的0是GPU id
gpu_list = [0,4,5]
ratio = 1024**2
while 1:
    for item in gpu_list:
        time.sleep(1)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(item))
        total = meminfo.total/ratio#以兆M为单位就需要除以1024**2
        used = meminfo.used/ratio
        free = meminfo.free/ratio
        print("total: ", total)
        print("used: ", used)
        print("free: ", free)
        if free > 6000:
            print("-------start------------")
            os.system(f'CUDA_VISIBLE_DEVICES={item} python train.py -s 1 -e 10000')
            print("--------finish--------------")
            break
        else:
            print(f"---------{item} is not free----------------")
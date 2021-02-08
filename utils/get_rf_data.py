from scipy.io import loadmat
import numpy as np
from PIL import Image
import os

for root, dirs, files in os.walk(r"C:\Users\ASUS\Desktop\rf_data", topdown=False):
    i = 0
    for name in files:
        filename = os.path.join(root, name)
        m = loadmat(filename)
        # 32*n
        length = len(m['scat'])
        # print(length)
        # rf_data = np.array(m['scat'])
        # new_length = (len(m['scat'])//16+1)*16

        # temp_array = np.zeros((new_length,32))
        # temp_array[0:length,:] = rf_data

        # rf_data_resize = temp_array.reshape(-1,512)

        # leng = rf_data_resize.shape[0]
        # temp_arr = np.zeros((leng*4,512))

        # temp_arr[0:leng,:] = rf_data_resize
        # temp_arr[leng:2*leng,:] = rf_data_resize
        # temp_arr[2*leng:3*leng,:] = rf_data_resize
        # temp_arr[3*leng:4*leng,:] = rf_data_resize

        # output = np.zeros((512,512))
        # output = temp_arr[0:512,:]
        # print(output.shape)

        # np.save(r'C:\Users\ASUS\Desktop\vscode_workspace\U-net\Pytorch-UNet\data\rf_datas\\'+str(i),output)
        # i = i+1


        # 这里的逻辑是先给原始rf数据做padding，然后折成n*512大小的矩阵，再重复堆叠直到达到512*512的大小



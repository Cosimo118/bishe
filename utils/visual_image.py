import h5py  #导入工具包  
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# #HDF5的写入：  
# imgData = np.zeros((30,3,128,256))  
# f = h5py.File('HDF5_FILE.h5','w')   #创建一个h5文件，文件指针是f  
# f['data'] = imgData                 #将数据写入文件的主键data下面  
# f['labels'] = range(100)            #将数据写入文件的主键labels下面  
# f.close()                           #关闭文件  

#HDF5的读取：  
f = h5py.File(r'C:\Users\ASUS\Downloads\archive_to_download\archive_to_download\reconstructed_image\experiments\contrast_speckle\contrast_speckle_expe_img_from_rf.hdf5','r')   #打开h5文件  
US = f['US']
US_G = US['US_DATASET0000']
data = US_G['data']
a = data['real'][:]                    #取出主键为data的所有的键值  
print(a.shape)
b = np.array(a)
b=b.transpose((2,1,0))
img = Image.fromarray(b,'CMYK')
img.show()
f.close()
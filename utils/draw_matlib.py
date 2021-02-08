import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

rf_data = np.load(r'../data/rf_image_data/1000.npy')
rf_data = np.array(rf_data)

image = Image.open(r'../data/image_data/1000.png')
image = np.array(image)

gt_mask = Image.open(r'../data/masks_512/1000.png')
gt_mask = np.array(gt_mask)

rf_image = Image.open(r'../test.png')
rf_image = np.array(rf_image)

mask = Image.open(r'../mask_test.png')
mask = np.array(mask)

plt.subplot(151)
plt.imshow(rf_data,cmap='gray')
plt.axis('off')
plt.title('rf_channel_data',fontsize='x-small')

plt.subplot(152)
plt.imshow(image,cmap='gray')
plt.axis('off')
plt.title('das_image',fontsize='x-small')

plt.subplot(153)
plt.imshow(rf_image,cmap='gray')
plt.axis('off')
plt.title('dnn_image',fontsize='x-small')


plt.subplot(154)
plt.imshow(gt_mask,cmap='gray')
plt.axis('off')
plt.title('true_segmentaion',fontsize='x-small')


plt.subplot(155)
plt.imshow(mask,cmap='gray')
plt.axis('off')
plt.title('dnn_segmentation',fontsize='x-small')

plt.savefig('test.jpg')
plt.show()
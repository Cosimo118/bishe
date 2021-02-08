import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from skimage import draw
import skimage.io

plt.rcParams['figure.figsize'] = (5.12, 5.12)
for r in range(10,41,1):
    img = np.zeros((512,512),dtype=np.uint8)
    rr, cc =draw.ellipse(256, 256, r/50.0*256, r/50.0*256)
    img[rr, cc] =255
    for z in range(15,21,1):
        skimage.io.imsave(r'C:\Users\ASUS\Desktop\masks\mask_'+str(r)+'_zstart'+str(z)+'.png',img)
# 

#for r in range(10,42,2):
# fig = plt.figure()
# ax = fig.subplots()
# rectangle = ax.patch
# rectangle.set(facecolor='black',    # 淡绿色，苍绿色
#             alpha=1
#             )
# circle = Ellipse(xy=(0, 0),    # 圆心坐标
#             width=r/20.0,    # 半径
#             height = r/5.0             
#             )

# ax.add_patch(p=circle)
# circle.set(fc='white',    # facecolor
#         ec='white',    # edgecolor,
#         alpha=0.6,
#         #lw=1,    # line widht
#         )

# # 调整坐标轴刻度范围
# ax.set(xlim=(-5, 5),
#     ylim=(-5, 5),
#     #aspect='equal'
#     )
# #image_r34_zstart35
# # for z in range(15,70,5):
# #     plt.savefig(r'C:\Users\ASUS\Desktop\mask\mask_'+str(r)+'_zstart'+str(z)+'.png')
# plt.show()
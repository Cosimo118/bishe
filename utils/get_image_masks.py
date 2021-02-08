import os

for root, dirs, files in os.walk(r"/home/cike/workspace/bishe/U-net/Pytorch-UNet/data/image_data", topdown=False):
    i = 0
    for name in files:
        filename = os.path.join(root, name)
        cmd = r'cp {} /home/cike/workspace/bishe/U-net/Pytorch-UNet/data/image_data/{}.png'.format(filename,str(i))
        os.system(cmd)
        i =i+1
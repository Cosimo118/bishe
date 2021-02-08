import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# img = Image.open(r'../data/image_data/0.png')
img = np.load(r'../data/rf_image_data/0.npy')
img = np.array(img)
np.set_printoptions(threshold=np.inf)
print(np.max(img))
print(np.min(img))


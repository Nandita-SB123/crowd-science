from PIL import Image
import numpy as np

img = Image.open("harrods.png").convert('L')
img_array = np.array(img)
binary = np.where(img_array > 200, 255, 0).astype(np.uint8)
Image.fromarray(binary).convert('RGB').save("harrods_floor4_walls.png")
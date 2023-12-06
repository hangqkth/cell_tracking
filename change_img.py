import matplotlib.pyplot as plt
import os

file_root = '3 min aquisition_1_C03_05'
save_root = '3 min aquisition_1_C03_05_jpg'
img_files = [os.path.join(file_root, img) for img in os.listdir(file_root)]
save_files = [os.path.join(save_root, img) for img in os.listdir(file_root)]
for i in range(len(img_files)):
    cell_img = plt.imread(img_files[i])
    # print(img_files[i][:-3])
    plt.imsave(save_files[i][:-3]+'.jpg', cell_img)

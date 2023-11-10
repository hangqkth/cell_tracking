import numpy as np
import scipy.io as scio
import numpy
import matplotlib.pyplot as plt
import os


# def load_mat(idx, mat_name):
#     return scio.loadmat(file_list[idx])[mat_name].astype(np.float64)

mat_file1 = '3 min aquisition_1_C03_11_001.mat'
mat_file2 = '3 min aquisition_1_C03_14_001.mat'
a = scio.loadmat(mat_file2)
label_keys = a.keys()
feature_name = a['featureNames']
feature1 = a['features1']
axis_ratio1 = feature1[:, 1]
center_dist1 = feature1[:, 6]
t_list = a['t']
# print(a.keys())
# print(a['t'])


def get_xy(axis_ratio, center_dist, img_shape, img):
    print(axis_ratio, center_dist, img_shape)
    center_loc = [img_shape[0] // 2, img_shape[1] // 2]
    plt.plot(center_loc[1], center_loc[0], marker='v', color="white")
    # plt.imshow(img)

    print(center_loc)
    x_offset = int(np.cos(axis_ratio) * center_dist)
    y_offset = int(np.sin(axis_ratio) * center_dist)
    print(x_offset, y_offset)
    target_loc = [center_loc[0] + x_offset, center_loc[1] + y_offset]
    print(target_loc)
    return target_loc


img = plt.imread('3 min aquisition_1_C03_11/3 min aquisition_1_C03_11_t001.TIF')
img = img[:, :, 0]
img_shape = [img.shape[0], img.shape[1]]
cell_loc = get_xy(axis_ratio1[0], center_dist1[0], img_shape, img)

plt.plot(cell_loc[1], cell_loc[0], marker='*', color="red")
plt.imshow(img)
plt.show()


# plt.imshow(img[:, :, 3])
# plt.show()
# print(img.shape)
# print(img.shape)

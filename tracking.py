import pickle
import numpy as np
from hungarian import hungarian_ac
from kalman_two_frame import kf_one_step


def read_list_from_file(file_path):

    with open(file_path, 'rb') as file:
        lst = pickle.load(file)
    return lst


def get_centers_from_bounding_boxes(bounding_boxes):
    # Create figure and axes
    centers = []
    # Add bounding boxes to the image
    for box in bounding_boxes:
        x_min, x_max, y_min, y_max = box[0], box[1], box[2], box[3]
        x_center = (x_max + x_min) // 2  # x is the dimension 1 of the img array
        y_center = (y_max + y_min) // 2  # y is the dimension 0 of the img array

        centers.append([x_center, y_center])
    return centers


def get_centers(seq_root, detections):
    # img_list = [os.path.join(seq_root, f) for f in os.listdir(seq_root)]
    all_cell_centers = []
    for i in range(len(detections)):
        # img = plt.imread(img_list[i])
        # print(img.shape)
        if len(detection[i]) > 0:
            centers_per_img = get_centers_from_bounding_boxes(detections[i])
            all_cell_centers.append(centers_per_img)
    return all_cell_centers


def kf_prediction(x_list, y_list, cov_list, ac_idx):
    x_list_next = []
    cov_list_next = []
    for c in range(len(ac_idx)):
        x_new, cov_new = kf_one_step(x_list[ac_idx[c][0]], y_list[ac_idx[c][1]], cov_list[c])
        x_list_next.append(x_new)
        cov_list_next.append(cov_new)
    return x_list_next, cov_list_next


def adjust_associate(x, y, origin_ac, cov_list):

    l_x, l_y = len(x), len(y)
    matched_x = [match[0] for match in origin_ac]
    matched_y = [match[1] for match in origin_ac]
    if l_x > l_y:  # YOLO miss some cells, copy unmatched x_m for y_n
        y_new, new_ac = y, origin_ac
        c = 0
        for i in range(l_x):
            if i not in matched_x:
                y_new.append(x[i])
                new_ac.append((i, l_y+c))
                c += 1
        return y_new, new_ac

    elif l_x < l_y:  # some cell is divided, find parent for the newborn cells
        x_new, new_ac, cov_new = x, origin_ac, cov_list
        c = 0
        for i in range(l_y):
            if i not in matched_y:  # new cell is y[i]
                distances = [np.linalg.norm(np.array(x[j])-np.array(y[i])) for j in range(l_x)]
                x_new.append(x[np.argmin(distances)])  # nearest neighbour to find parent
                cov_new.append(cov_list[np.argmin(distances)])
                new_ac.append((l_x+c, i))
                c += 1
        return x_new, new_ac, cov_new


def track_cell_centers(all_centers):
    # for every frame in the image sequence
    all_x = []
    all_association = []
    x_k, cov_list = None, None
    for f in range(len(all_centers)-1):
        y_k = all_centers[f]  # current frame observation

        # initialization
        if f == 0:
            x_k = y_k  # initialized prediction, no need association
            # predict next frame based on last prediction and current observation
            match_idx = hungarian_ac(np.array(x_k), np.array(y_k))
            x_k_next, cov_list = kf_prediction(x_k, y_k, [np.zeros((2, 2)) for n in range(len(x_k))], match_idx)
            x_k = x_k_next  # update x_k for next loop
        else:
            # association
            match_idx = hungarian_ac(np.array(x_k), np.array(y_k))
            if len(x_k) < len(y_k):
                x_k, match_idx, cov_list = adjust_associate(x_k, y_k, match_idx, cov_list)
            elif len(x_k) > len(y_k):
                y_k, match_idx = adjust_associate(x_k, y_k, match_idx, cov_list)
            x_k_next, cov_list = kf_prediction(x_k, y_k, cov_list, match_idx)
            x_k = x_k_next
            print(x_k)


if __name__ == "__main__":
    detection = read_list_from_file('runs/detect/3 min aquisition_1_C03_11.pkl')
    # print(detection)
    all_centers = get_centers('3 min aquisition_1_C03_11', detection)
    track_cell_centers(all_centers)



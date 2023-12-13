import pickle
import time
import copy
import numpy as np
from hungarian import hungarian_ac
from kalman_two_frame import kf_one_step
import matplotlib.pyplot as plt


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


def get_centers(detections):
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
    new_x_dict, new_cov_dict = {}, {}
    cov_list_next = []
    for c in range(len(ac_idx)):
        x_new, cov_new = kf_one_step(x_list[ac_idx[c][0]], y_list[ac_idx[c][1]], cov_list[c])
        new_x_dict[str(ac_idx[c][1])] = x_new
        new_cov_dict[str(ac_idx[c][1])] = cov_new
        # x_list_next.append(x_new)
        # cov_list_next.append(cov_new)
    for i in range(len(x_list)):
        x_list_next.append(new_x_dict[str(i)])
        cov_list_next.append(new_cov_dict[str(i)])
    return x_list_next, cov_list_next


def adjust_associate(x, y, origin_ac, cov_list):

    l_x, l_y = len(x), len(y)
    matched_x = [match[0] for match in origin_ac]
    matched_y = [match[1] for match in origin_ac]
    if l_x > l_y:  # YOLO miss some cells, copy unmatched x_m for y_n
        y_new, new_ac = y.copy(), origin_ac.copy()
        c = 0
        for i in range(l_x):
            if i not in matched_x:
                y_new.append(x[i])
                new_ac.append((i, l_y+c))
                c += 1

        return y_new, new_ac

    elif l_x < l_y:  # some cell is divided, find parent for the newborn cells
        # print(x)
        # print(y)
        # print(origin_ac)
        x_new, new_ac, cov_new = x.copy(), origin_ac.copy(), cov_list.copy()
        c = 0
        for i in range(l_y):
            if i not in matched_y:  # new cell is y[i]
                distances = [np.linalg.norm(np.array(x[j])-np.array(y[i])) for j in range(l_x)]
                x_new.append(x[np.argmin(distances)])  # nearest neighbour to find parent
                cov_new.append(cov_list[np.argmin(distances)])
                new_ac.append((l_x+c, i))
                c += 1
                # print(new_ac)
                # print(x_new)
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
            x_k = y_k.copy()  # initialized prediction, no need association
            all_x.append(x_k)
            # predict next frame based on last prediction and current observation
            match_idx = hungarian_ac(np.array(x_k), np.array(y_k))
            x_k_next, cov_list = kf_prediction(x_k, y_k, [np.zeros((2, 2)) for n in range(len(x_k))], match_idx)
            x_k = x_k_next.copy()  # update x_k for next loop
        else:
            # association
            match_idx = hungarian_ac(np.array(x_k).copy(), np.array(y_k).copy())

            # adjust if length does not match
            if len(x_k) < len(y_k):
                x_k, match_idx, cov_list = adjust_associate(x_k.copy(), y_k.copy(), match_idx.copy(), cov_list.copy())
            elif len(x_k) > len(y_k):
                y_k, match_idx = adjust_associate(x_k.copy(), y_k.copy(), match_idx.copy(), cov_list.copy())


            # # adjust for too long distance
            # distance_list = []
            # for c in range(len(match_idx)):
            #     distance_list.append(np.linalg.norm(np.array(x_k[match_idx[c][0]])-np.array(y_k[match_idx[c][1]]), 2))

            # if max(distance_list) > 10:
            #     if len(distance_list) > 1:
            #         max_index = distance_list.index(max(distance_list))
            #         modified_lst = distance_list[:max_index] + distance_list[max_index + 1:]
            #         second_max_index = modified_lst.index(max(modified_lst))
            #         if second_max_index >= max_index:
            #             second_max_index += 1
            #
            #         if distance_list[second_max_index] > 5:
            #             print("found one")
            #             temp1, temp2 = match_idx[max_index][1], match_idx[second_max_index][1]
            #             temp3, temp4 = match_idx[max_index][0], match_idx[second_max_index][0]
            #             match_idx[max_index] = (temp3, temp2)
            #             match_idx[second_max_index] = (temp4, temp1)
            # print(f)
            # print(x_k)
            # print(y_k)
            # print(match_idx)
            all_x.append(x_k)
            x_k_next, cov_list = kf_prediction(x_k.copy(), y_k.copy(), cov_list.copy(), match_idx.copy())
            x_k = x_k_next.copy()

        # print("x_k to saved:", x_k)
        # print("match_idx to saved:", match_idx)
        # all_x.append(x_k)
        all_association.append(match_idx)
    # print(len(all_centers))
    # print(len(all_x))
    # print(len(all_association))

    return [all_centers, all_x, all_association]


def draw_trajectory(info):
    all_centers, all_x, all_association = info[0], info[1], info[2]
    trajectories = [[all_x[0][i]] for i in range(len(all_x[0]))]
    points = {}
    # points[str(i)] means the i_th element in all_x[f] is saved in point_dict[str(i)]_th list in trajectories
    for i in range(len(all_x[0])):
        points[str(i)] = i
    points = {'0': 0}

    start_frame = [0]
    # print(len(all_x), len(all_association))
    for f in range(len(all_association)-1):
        # print(len(all_x[f+1]), len(all_association[f]), len(points))
        # print(all_x[f+1])
        # print(all_association[f])

        current_x = all_x[f+1].copy()
        current_match = all_association[f].copy()
        # print(f+1)
        # print(current_x)
        # print(current_match)

        # if len(all_x[f+1]) > len(points):
        #     trajectories.append([])
        #     start_frame.append(f)
        #     points.append(len(trajectories)-1)
        #
        new_points = {}
        last_idx_list, next_idx_list = [], []
        for m in range(len(current_match)):
            one_match = current_match[m]
            last_idx = one_match[0]
            next_idx = one_match[1]
            last_idx_list.append(last_idx)
            next_idx_list.append(next_idx)
            trajectories[points[str(last_idx)]].append(current_x[next_idx])
            new_points[str(next_idx)] = points[str(last_idx)]

        if len(current_match) < len(current_x):
            # print(next_idx_list)
            start_frame.append(f+1)
            for i in range(len(current_x)):
                if i not in next_idx_list:
                    trajectories.append([current_x[i]])
                    new_points[str(i)] = len(trajectories) - 1

        points = new_points.copy()

    # Plot the trajectory

    plt.figure(figsize=(18, 12))  # Adjust the figure size as needed
    plt.imshow(plt.imread('3 min aquisition_1_C03_14/3 min aquisition_1_C03_14_t465.TIF'))
    # trajectories_new = [lst for lst in trajectories if len(lst) >= 100]
    # trajectories = trajectories_new
    for l in range(len(trajectories)):
        print(len(trajectories[l]))
        # Extract x and y coordinates separately
        x_coords = [coord[0] for coord in trajectories[l]]
        y_coords = [coord[1] for coord in trajectories[l]]
        # colors = ['blue', 'magenta', 'green', 'orange', 'purple', 'orange', 'black', 'yellow']
        # for c in range(len(x_coords)-1):
        #     x, y = x_coords[c], y_coords[c]
        #     x1, y1 = x_coords[c+1], y_coords[c+1]
        #
        #     # if abs(x-x1)+abs(y-y1) < 150:
        #     plt.plot([x, x1], [y, y1], marker='o', linestyle='-', c=colors[l], markersize=5)
        plt.plot(x_coords, y_coords, marker='o', linestyle='-', )

        # plt.scatter(x_coords, y_coords, marker='o', linestyle='-', )

        plt.title('Trajectory Plot', fontsize=24)
        plt.xlabel('X-axis', fontsize=18)
        plt.ylabel('Y-axis', fontsize=18)
        plt.grid(True)

    plt.legend(["cell "+str(l+1) for l in range(len(trajectories))], fontsize=18)
    plt.xlim([0, 1388])
    plt.ylim([0, 1040])
    plt.xticks(fontsize=18)  # Change x-axis tick labels font size
    plt.yticks(fontsize=18)
    plt.show()

    plt.figure(figsize=(18, 12))
    plt.imshow(plt.imread('3 min aquisition_1_C03_14/3 min aquisition_1_C03_14_t465.TIF'))
    for i in range(len(all_centers)):
        for p in range(len(all_centers[i])):
            plt.scatter(all_centers[i][p][0], all_centers[i][p][1], marker='o', c='b')
    plt.title('Observation Plot', fontsize=24)
    plt.xlabel('X-axis', fontsize=18)
    plt.ylabel('Y-axis', fontsize=18)
    plt.grid(True)
    plt.xlim([0, 1388])
    plt.ylim([0, 1040])
    plt.xticks(fontsize=18)  # Change x-axis tick labels font size
    plt.yticks(fontsize=18)
    plt.show()
    # print(start_frame)
    # print(len(trajectories[0]))
    for t in range(len(trajectories)):
        print(len(trajectories[t]))


    return trajectories, start_frame


def plot_tracking(traces, start_f):
    activated_cells = []
    # for f in range(len(traces[0])):
    for f in range(450):
        if len(activated_cells) < len(traces):
            if f == start_f[len(activated_cells)]:
                activated_cells.append(len(activated_cells))
        # print(activated_cells)
        plt.figure(figsize=(12, 9))
        plt.imshow(plt.imread('3 min aquisition_1_C03_11/3 min aquisition_1_C03_11_t'+str(f+1).zfill(3)+'.TIF'))
        for c in activated_cells:
            x1, y1 = traces[c][f-start_f[c]-1][0], traces[c][f-start_f[c]-1][1]
            plt.scatter(x1, y1, s=60)
            plt.xlim([0, 1388])
            plt.ylim([0, 1040])
            plt.grid(True)
            # if f > 20:
            # print(f)
            # print(x1, y1)
        plt.legend(["cell " + str(c+1) for c in activated_cells])
        plt.title('t='+str(f+1))
        plt.savefig('runs/tracks/'+str(f)+'.png')
        plt.close()

    # print(activated_cells)


if __name__ == "__main__":
    detection = read_list_from_file('runs/detect/3 min aquisition_1_C03_11.pkl')
    # print(detection)
    all_centers = get_centers(detection)
    tracking_info = track_cell_centers(all_centers)

    traces, start_f = draw_trajectory(tracking_info)


    plot_tracking(traces, start_f)




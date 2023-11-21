import matplotlib.pyplot as plt
import matplotlib.patches as patches
from play_yolo import read_list_from_file
import cv2
import os


def adjust_bounding_boxes(x_min, x_max, y_min, y_max, w_t, h_t):
    # Calculate the original width and height of each bounding box
    original_width = x_max - x_min
    original_height = y_max - y_min

    # Calculate the center of the bounding boxes
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Calculate the new x_min, x_max, y_min, y_max based on the target width and height
    new_x_min = center_x - (w_t / 2)
    new_x_max = center_x + (w_t / 2)
    new_y_min = center_y - (h_t / 2)
    new_y_max = center_y + (h_t / 2)

    return new_x_min, new_x_max, new_y_min, new_y_max


def plot_bounding_boxes(image, bounding_boxes, root, idx):
    # Add bounding boxes to the image
    # width_list, height_list = [], []
    max_w = 90
    max_h = 50
    for b in range(len(bounding_boxes)):
        box = bounding_boxes[b]
        x_min, x_max, y_min, y_max = box[0], box[1], box[2], box[3]
        x_min, x_max, y_min, y_max = adjust_bounding_boxes(x_min, x_max, y_min, y_max, max_w, max_h)
        # print(x_max-x_min)
        # print(y_max-y_min)
        # print(int(x_min), int(x_max), int(y_min), int(y_max))
        # img_w, img_h = image.shape[0], image.shape[1]
        # width_list.append(x_max-x_min)
        # height_list.append(y_max-y_min)

        # plt.imshow(image[int(y_min):int(y_max), int(x_min):int(x_max), :])
        # plt.show()
        cropped_cell = image.copy()[int(y_min):int(y_max), int(x_min):int(x_max), :]
        # plt.imshow(cropped_cell)
        # plt.show()
        # print(root + str(idx) + str(b) + '.png')
        cv2.imwrite(root + str(idx) + '_' + str(b) + '.png', cropped_cell)
        # plt.imsave(root + str(idx) + str(b) + '.png', cropped_cell)



    # Show the plot
    # print(max(width_list))
    # print(min(height_list))
    # return max(width_list), min(height_list)
    # plt.savefig(root+str(idx)+'.png')


def show_detection_on_seq_data(seq_root, detections):
    img_list = [os.path.join(seq_root, f) for f in os.listdir(seq_root)]
    os.mkdir('runs/crops/'+seq_root)
    # all_w, all_h = [], []
    for i in range(len(img_list)-1):
    # for i in range(5):
        img = plt.imread(img_list[i])
        # print(img.shape)
        if len(detection[i]) > 0:
            # w, h = plot_bounding_boxes(img, detections[i], 'runs/plots/'+seq_root+'/', i)
            plot_bounding_boxes(img, detections[i], 'runs/crops/' + seq_root + '/', i)
    #         all_w.append(w)
    #         all_h.append(h)
    # print(sorted(all_w), sorted(all_h))


if __name__ == "__main__":
    detection = read_list_from_file('runs/detect/3 min aquisition_1_C03_11.pkl')
    # print(detection)
    show_detection_on_seq_data('3 min aquisition_1_C03_11', detection)
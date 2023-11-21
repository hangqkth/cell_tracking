import torch
import scipy.io as scio
import os
import numpy as np
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s',  _verbose=False, autoshape=False)
# print(model)


def convert_to_yolo_format(annotation_list, output_file):
    yolo_annotations = ""

    for annotation in annotation_list:
        class_id = annotation['class']  # Class index
        x_min, y_min, x_max, y_max = annotation['bbox']  # Bounding box coordinates

        # Calculate bounding box coordinates relative to the image size
        image_width = 1388  # Replace with your image width
        image_height = 1040  # Replace with your image height

        x_center = (x_min + x_max) / (2 * image_width)
        y_center = (y_min + y_max) / (2 * image_height)
        box_width = (x_max - x_min) / image_width
        box_height = (y_max - y_min) / image_height

        yolo_annotations += f"{class_id} {x_center} {y_center} {box_width} {box_height}\n"

    # Write YOLO annotations to a file
    with open(output_file, 'w') as f:
        f.write(yolo_annotations)


# # Example input list of annotations
# annotations = [
#     {'class': 0, 'bbox': (100, 150, 300, 400)},  # Example annotation 1
#     {'class': 1, 'bbox': (50, 80, 200, 250)},  # Example annotation 2
#     # Add more annotations as needed
# ]
#
# # Output file path for YOLO annotations
# output_file_path = 'output_annotations.txt'  # Replace with your desired output file path
#
# # Convert the input annotations to YOLO format and save to a file
# convert_to_yolo_format(annotations, output_file_path)


def load_mat(mat_name):
    return scio.loadmat(mat_name)


bound_mat = load_mat('./baxter_label/C03_11_bound.mat')['all_bound'][:, :-1, :]

# os.mkdir('yolo_label/C03_11_001')

for i in range(bound_mat.shape[1]):
    label = bound_mat[:, i, :]
    for c in range(label.shape[1]):
        box = label[:, c].copy()
        # print(cell_box)
        if np.sum(box) != 0:
            print('yes')
            print(np.sum(box))
            x_min = box[0, c]  # x-coordinate of the upper-left corner
            y_min = box[1, c]  # y-coordinate of the upper-left corner
            w = box[2, c]  # width of the bounding box
            h = box[3, c]  # height of the bounding box
            x_max = x_min + w


        # print(sum(cell_box))

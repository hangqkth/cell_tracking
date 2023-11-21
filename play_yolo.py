import cv2
import torch
from tqdm import tqdm
import os
import numpy as np
import pickle


def save_list_to_file(lst, file_path):
    """
    Save a Python list to a file using pickle.

    Parameters:
    - lst: The list to be saved.
    - file_path: The path to the file where the list will be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(lst, file)


def read_list_from_file(file_path):
    """
    Read a Python list from a file using pickle.

    Parameters:
    - file_path: The path to the file containing the saved list.

    Returns:
    - The list read from the file.
    """
    with open(file_path, 'rb') as file:
        lst = pickle.load(file)
    return lst


# Function to remove the background and keep only the circular well
def remove_background(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the edges of the grayscale image
    # (This image is just to understand how the Hough Circle Transform works.)
    edges = cv2.Canny(gray, threshold1=80, threshold2=180)

    # Use the Hough Circle Transform to detect the circular well
    circles = cv2.HoughCircles(
                  gray,
                  cv2.HOUGH_GRADIENT,
                  dp=1,
                  minDist=100,
                  param1=180,
                  param2=30,
                  minRadius=100,
                  maxRadius=np.round(gray.shape[0]/2).astype('int')
                )

    # Ensure that circles were detected
    if circles is not None:
        # Round the coordinates and radius to integers
        circles = np.round(circles[0, :]).astype('int')

        # Create a mask with a black background
        mask = np.zeros_like(gray)

        # Draw the detected circle on the mask
        for (x, y, r) in circles:
            cv2.circle(mask, (x, y), r, 255, -1)

        # Use the mask to extract the circular well
        result = cv2.bitwise_and(image, image, mask=mask)

        # # Save results
        # cv2.imwrite("output.TIF", result)
        # cv2.imwrite("edges.TIF", edges)
    else:
        print("No circular well was detected in the image.")
        result = None
    return result


def run_yolo_on_seq_imgs(seq_root, yolo_model):
    img_list = [os.path.join(seq_root, f) for f in os.listdir(seq_root)]
    detection_list = []
    for i in tqdm(range(len(img_list))[1:]):
        img = cv2.imread(img_list[i])
        img = remove_background(img)
        results = yolo_model([img], size=640)  # batch of images
        crops = results.crop(save=True)
        result_tensor = results.pandas().xyxy[0]
        result_list = read_result(result_tensor)
        selected = select_object(result_list)
        detection_list.append(selected)
    save_list_to_file(detection_list, 'runs/detect/'+seq_root+'.pkl')


def read_result(result_tensor):
    detection = []
    for i in range(len(result_tensor['xmin'])):
        detection.append([result_tensor['xmin'][i], result_tensor['xmax'][i],
                          result_tensor['ymin'][i], result_tensor['ymax'][i], ])
    return detection


def select_object(result_list):
    max_cell_width = 200
    selected = []
    for i in range(len(result_list)):
        if abs(result_list[i][0] - result_list[i][1]) < max_cell_width and \
                abs(result_list[i][2] - result_list[i][3]) < max_cell_width:
            selected.append(result_list[i])
    return selected


if __name__ == "__main__":
    # Model

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s',  _verbose=False)



    model.conf = 0.5  # NMS confidence threshold
    model.iou = 0.1  # NMS IoU threshold
    # agnostic = False  # NMS class-agnostic
    # multi_label = False  # NMS multiple labels per box
    # classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    # max_det = 1000  # maximum number of detections per image
    # amp = False  # Automatic Mixed Precision (AMP) inference

    run_yolo_on_seq_imgs('3 min aquisition_1_C03_11', yolo_model=model)
    # detection = read_list_from_file('runs/detect/3 min aquisition_1_C03_11.pkl')
    # print(detection)

    # plot_bounding_boxes(test_img, selected)

    # #      xmin    ymin    xmax   ymax  confidence  class    name
    # # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
    # # 1  433.50  433.50   517.5  714.5    0.687988     27     tie
    # # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
    # # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie

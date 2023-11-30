import pickle
from hungarian import associate

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



if __name__ == "__main__":
    detection = read_list_from_file('runs/detect/3 min aquisition_1_C03_11.pkl')
    # print(detection)
    all_centers = get_centers('3 min aquisition_1_C03_11', detection)
    associate(all_centers, 'hungarian')

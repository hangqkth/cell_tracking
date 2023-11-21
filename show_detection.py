import matplotlib.pyplot as plt
import matplotlib.patches as patches
from play_yolo import read_list_from_file
import os


def plot_bounding_boxes(image, bounding_boxes, root, idx):
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image, interpolation='none')
    # Add bounding boxes to the image
    for box in bounding_boxes:
        x_min, x_max, y_min, y_max = box[0], box[1], box[2], box[3]
        width = x_max - x_min
        height = y_max - y_min
        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    # Show the plot
    plt.savefig(root+str(idx)+'.png')


def show_detection_on_seq_data(seq_root, detections):
    img_list = [os.path.join(seq_root, f) for f in os.listdir(seq_root)]
    os.mkdir('runs/plots/'+seq_root)
    for i in range(len(img_list)):
        img = plt.imread(img_list[i])
        # print(img.shape)
        if len(detection[i]) > 0:
            plot_bounding_boxes(img, detections[i], 'runs/plots/'+seq_root+'/', i)


if __name__ == "__main__":
    detection = read_list_from_file('runs/detect/3 min aquisition_1_C03_14.pkl')
    # print(detection)
    show_detection_on_seq_data('3 min aquisition_1_C03_14', detection)

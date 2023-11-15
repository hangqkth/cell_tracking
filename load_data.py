import scipy.io as scio
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def load_mat(mat_name):
    return scio.loadmat(mat_name)


def plot_pos_and_box_on_img(img, pos_array, box_array, root, idx):
    # Display the image
    plt.figure('0', figsize=(img.shape[1] / 100, img.shape[0] / 100))
    plt.imshow(img)
    plt.axis('off')
    # Coordinates and dimensions of the bounding box
    for c in range(pos_array.shape[-1]):
        if pos_array[0, c] + pos_array[1, c] > 0:
            x = box_array[0, c]  # x-coordinate of the upper-left corner
            y = box_array[1, c]  # y-coordinate of the upper-left corner
            w = box_array[2, c]  # width of the bounding box
            h = box_array[3, c]  # height of the bounding box

            # Create a Rectangle patch
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')

            # Add the bounding box to the plot
            plt.gca().add_patch(rect)

    # Show the plot with the bounding box
    # plt.show()
    plt.savefig('runs/ba_plots/'+root+'/'+str(idx)+'.png', bbox_inches='tight', pad_inches=0)
    plt.close('0')


def show_label_on_seq_imgs(seq_root, pos, box):
    img_list = [os.path.join(seq_root, f) for f in os.listdir(seq_root)]

    os.mkdir('runs/ba_plots/' + seq_root)
    for i in range(len(img_list)):
    # for i in range(5):
        img = plt.imread(img_list[i])
        plot_pos_and_box_on_img(img, pos[:, i, :], box[:, i, :], seq_root, i)


if __name__ == "__main__":
    pos_mat = load_mat('./baxter_label/C03_11_pos.mat')['all_pos'][:, :-1, :]
    bound_mat = load_mat('./baxter_label/C03_11_bound.mat')['all_bound'][:, :-1, :]
    show_label_on_seq_imgs('3 min aquisition_1_C03_11', pos_mat, bound_mat)

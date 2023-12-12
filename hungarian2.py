import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import os
import pickle
def hungarian_association():
    # cost_matrix = np.zeros((len(prev_cells), len(current_cells)))
    #
    # for i in range(len(prev_cells)):
    #     for j in range(len(current_cells)):
    #         cost_matrix[i, j] = np.linalg.norm(prev_cells[i] - current_cells[j])

    cost_matrix = [
        [1, 2, 8],
        [4, 8, 1],
        [3, 1, 1]]

    # Using linear_sum_assignment to solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    associations = list(zip(row_ind, col_ind))
    print(associations)
    return associations

# [(0, 0), (1, 2), (2, 1)]
hungarian_association()

#
# def nearest_neighbor_association(prev_cells, current_cells):
#     associations = []
#
#     for i in range(len(prev_cells)):
#         min_distance = float('inf')
#         best_match_index = None
#
#         for j in range(len(current_cells)):
#             distance = np.linalg.norm(prev_cells[i] - current_cells[j])
#
#             if distance < min_distance:
#                 min_distance = distance
#                 best_match_index = j
#
#         associations.append((i, best_match_index))
#
#     return associations
#
#
# # to show the data association results for one whole time series cell images
# # input = 'cells', the list of every cell center location in images
# # input = 'type', the data association type,  "Nearest" for nearest neighbour or "Hungarian"
# def associate(cells, type = "Hungarian"):
#     for i in range(len(cells)-1):
#         prev_cells = np.array(cells[i])
#         current_cells = np.array(cells[i + 1])
#
#         if type == "Nearest":
#             association = nearest_neighbor_association(prev_cells, current_cells)
#         else:
#             association = hungarian_association(prev_cells, current_cells)
#
#
#         plt.clf()
#         plt.scatter(prev_cells[:, 0], prev_cells[:, 1], color='blue', label='Previous Image')
#         plt.scatter(current_cells[:, 0], current_cells[:, 1], color='red', label='Current Image')
#
#         for prev_idx, current_idx in association:
#             plt.plot([prev_cells[prev_idx, 0], current_cells[current_idx, 0]],
#                      [prev_cells[prev_idx, 1], current_cells[current_idx, 1]], color='green')
#
#
#         plt.legend()
#         plt.xlim(0, 1388)
#         plt.ylim(0, 1040)
#         plt.xlabel('X-coordinate')
#         plt.ylabel('Y-coordinate')
#         plt.title('Cell Association')
#
#         if type == 'Nearest':
#             output_folder = 'frames_nearest'
#         else:
#             output_folder = 'frames_hungarian'
#
#         os.makedirs(output_folder, exist_ok=True)
#         # plt.savefig(os.path.join(output_folder, f'frame_{i}.png'))
#         #
#
#         plt.show()
#
# # Example usage:
# # Assume prev_cells and current_cells are lists of 2D coordinates of cells in the previous and current images
# # Replace these lists with your actual data
# if __name__ == "__main__":
#
#     with open('./runs/detect/3 min aquisition_1_C03_11.pkl', 'rb') as file:
#         centers = pickle.load(file)
#
#     # associate(centers, "Nearest")
#
#
#     for i in range(len(centers)-1):
#         prev_cells = np.array(centers[i])
#         current_cells = np.array(centers[i + 1])
#         association = hungarian_association(prev_cells, current_cells)
#         print(association)
## Cell tracking 

### Data: image sequences containing fixed camara view of microscope cell photos

Put the data folder directly under the project root like this: cell_tracking\3 min aquisition_1_C03_11

### The algorithm is based on YOLOv5, Kalman Filter. The Hungarian Algorithm is used for data association

To implement cell tracking, first run play_yolo.py, then YOLO model will find the position of cells and save them as .pkl file
Then, run tracking.py, it will track the cell position.

### Estimate cell state
To train and test the CNN model for cell state estimation, run train_shape.py
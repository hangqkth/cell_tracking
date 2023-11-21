import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.utils.data as data
from sklearn.utils import shuffle
import torch.nn as nn
import torch
import torch.nn.functional as F
import time
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


def img2array(root):
    classes = ['g', 'm', 's']
    classes_path = [os.path.join(root, c) for c in classes]
    array_data_list = []
    array_label_list = []
    for c in range(len(classes_path)):
        im_files = [os.path.join(classes_path[c], im) for im in os.listdir(classes_path[c])]
        all_im = []
        for i in im_files:
            im = plt.imread(i)[:, :, :3]
            im = np.transpose(im, (2, 0, 1))
            all_im.append(im)
        all_im = np.stack(all_im, axis=0)
        array_data_list.append(all_im)
        array_label_list.append(np.ones((all_im.shape[0], 1))*c)
    data_array = np.concatenate(array_data_list, axis=0)
    label_array = np.concatenate(array_label_list, axis=0)
    return data_array, label_array


class CellShapeData(data.Dataset):
    def __init__(self, data_array, label_array):
        self.data = data_array
        self.label = label_array

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        return self.data[item, ], self.label[item, 0]


class CellShapeNet(nn.Module):
    def __init__(self, c_in=3):
        super(CellShapeNet, self).__init__()
        self.deep_feature = 64*11*9  # 6336
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16))
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(2, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64))
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(2, 3)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64))

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.deep_feature, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU())
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x_in):
        x = self.cnn1(x_in)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = torch.reshape(x, [x.shape[0], -1])

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train_and_val(train_loader, val_loader, lr, model, epochs, criterion, device):
    best_loss_test = float("inf")
    best_acc_test = 0
    str_labels = ['G', 'M', 'S']
    for epoch in range(epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # learning rate, 0.001
        batch = 0
        "start training"
        for data, label in train_loader:
            batch += 1
            model.train()  # train mode
            data = data.to(device=device, dtype=torch.float32)  # numpy to tensor
            pred = model(data)  # forward propagation
            optimizer.zero_grad()  # initialize optimizer
            loss = criterion(pred, label.to(device=device, dtype=torch.long))
            loss.backward()  # backward propagation
            optimizer.step()  # update model parameters
            if batch % 10 == 0:
                print("\rTrain Epoch: {:d} | Train loss: {:.4f} | Batch : {}/{}".format(epoch + 1, loss, batch, len(train_loader)))

            "start testing"
            if batch % 10 == 0:
                loss_sum = 0
                pred_list, true_list = [], []
                for data, label in val_loader:
                    model.eval()  # evaluating mode
                    with torch.no_grad():  # no gradient
                        data = data.to(device=device, dtype=torch.float32)
                        pred = model(data)
                        loss_test = criterion(pred, label.to(device=device, dtype=torch.long))
                        loss_sum += loss_test
                        pred = pred.cpu().detach().numpy()
                        pred = np.argmax(pred, axis=1)
                        pred_list += pred.tolist()
                        true_list += label.numpy().tolist()
                loss_avg = loss_sum / len(val_loader)
                acc = accuracy_score(true_list, pred_list)
                if acc >= best_acc_test:
                    best_loss_test = loss_avg
                    best_acc_test = acc
                    disp = ConfusionMatrixDisplay.from_predictions(y_true=true_list, y_pred=pred_list,
                                                                   cmap=plt.cm.Blues,
                                                                   xticks_rotation="vertical",
                                                                   display_labels=str_labels)
                    plt.tight_layout()
                    plt.show()
                    # torch.save(model.state_dict(), './saved_model/cell_net.pth')
                print("\rTest Epoch: {:d} | Test loss: {:.4f} | Test Accuracy: {:.4%} | Best evaluation loss: {:.6f}".format(epoch + 1, loss_avg, acc, best_loss_test))
                time.sleep(0.1)



if __name__ == "__main__":
    data_array, label_array = img2array('./gms')
    shuffled_dataset, shuffled_labels = shuffle(data_array, label_array, random_state=42)
    train_data, train_label = shuffled_dataset[:900, ], shuffled_labels[:900, ]
    val_data, val_label = shuffled_dataset[900:, ], shuffled_labels[900:, ]
    train_loader = data.DataLoader(dataset=CellShapeData(train_data, train_label), batch_size=32, shuffle=True)
    val_loader = data.DataLoader(dataset=CellShapeData(val_data, val_label), batch_size=32, shuffle=True)
    model = CellShapeNet()

    criterion = nn.CrossEntropyLoss()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #
    # model_param = torch.load('./saved_model/garbage_net.pth')  # keep on training from the existing model
    # model.load_state_dict(model_param)
    #
    model.to(device)
    print("start training")
    train_and_val(train_loader, val_loader, 1e-3, model, 50, criterion, device)
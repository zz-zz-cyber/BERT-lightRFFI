import torch
import torch.nn.functional as F
from model import Bert, Classification, ResNet, distillation_loss, EffNetV2, AlBert
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from dataset import KdDataset
import matplotlib.pyplot as plt
import time
import os
from scipy.io import loadmat

cfg_mobilenet = [
    # k, t, c, SE, HS, s
    [3, 1, 16, 1, 0, 2],
    [3, 4.5, 24, 0, 0, 2],
    [3, 3.67, 24, 0, 0, 1],
    [5, 4, 40, 1, 1, 2],
    [5, 6, 40, 1, 1, 1],
    [5, 6, 40, 1, 1, 1],
    [5, 3, 48, 1, 1, 1],
    [5, 3, 48, 1, 1, 1],
    [5, 6, 96, 1, 1, 2],
    [5, 6, 96, 1, 1, 1],
    [5, 6, 96, 1, 1, 1],
]

cfg_efficientnet = [
        # t, c, n, s, SE
        [1, 24, 2, 1, 0],
        [4, 48, 4, 2, 0],
        [4, 64, 4, 2, 0],
        [4, 128, 6, 2, 1],
        [6, 160, 9, 1, 1],
        [6, 256, 15, 2, 1],
    ]

class KD:
    def __init__(self, train_datapath, val_datapath):
        self.epoch_num = 100
        self.classes = 25
        self.t_model = Bert()
        self.s_model = ResNet()
        self.clmodel = Classification(512, self.classes)
        self.lr = 0.0001
        self.batchsize = 32
        # self.train_dataset = KdDataset(train_datapath)
        # self.train_dataloader = DataLoader(self.train_dataset, self.batchsize, shuffle=True)
        # self.val_dataset = KdDataset(val_datapath)
        # self.val_dataloader = DataLoader(self.val_dataset, self.batchsize, shuffle=False)
        self.res_optimizer = Adam(self.s_model.parameters(), lr=self.lr)
        self.class_optimizer = Adam(self.clmodel.parameters(), lr=self.lr)
        self.criterion = distillation_loss
        self.temperature = 7
        self.alpha = 0.3
        self.acc = []
        self.train_loss = []
        self.val_loss = []
        self.t_modelpath = '/home/liuyi/MLRFF/res/best/bert.pth'
        self.s_modelpath = '/home/liuyi/MLRFF/res/best/student.pth'
        self.cl_modelpath = '/home/liuyi/MLRFF/res/classification_model.pth'
        self.losspath = '/home/liuyi/MLRFF/res/albert_s.png'

    def train(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t_model.load_state_dict(torch.load(self.t_modelpath))
        self.t_model.to(device)
        self.s_model.to(device)
        self.s_model.to(torch.double)
        self.t_model.to(torch.double)
        train_epoch_loss = []
        for epoch in range(self.epoch_num):
            self.s_model.train()
            self.t_model.eval()
            for idx, (t_data, s_data, label) in enumerate(self.train_dataloader):
                t_data = t_data.to(device)
                s_data = s_data.to(device)
                t_feature = self.t_model(t_data)
                s_feature = self.s_model(s_data)
                loss = F.mse_loss(s_feature, t_feature[:, 0, :])
                self.res_optimizer.zero_grad()
                loss.backward()
                self.res_optimizer.step()
                train_epoch_loss.append(loss.item())
            self.train_loss.append(np.mean(train_epoch_loss))

            self.s_model.eval()
            val_epoch_loss = []
            with torch.no_grad():
                for idx, (t_data, s_data, label) in enumerate(self.val_dataloader):
                    t_data = t_data.to(device)
                    s_data = s_data.to(device)
                    t_feature = self.t_model(t_data)
                    s_feature = self.s_model(s_data)
                    loss = F.mse_loss(s_feature, t_feature[:, 0, :])
                    val_epoch_loss.append(loss.item())
                self.val_loss.append(np.mean(val_epoch_loss))
            print("Epoch {}: train_loss = {}, val_loss = {}".format(epoch, self.train_loss[-1], self.val_loss[-1]))

        plt.figure(figsize=(6, 4))
        plt.plot(self.train_loss, '-o', label="train_loss")
        plt.plot(self.val_loss, '-o', label="val_loss")
        plt.title("epochs_loss")
        plt.legend()
        plt.savefig(self.losspath)
        plt.close()
        torch.save(self.s_model.state_dict(), self.s_modelpath)

    def classification(self):
        acc = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clmodel.to(device)
        self.s_model.load_state_dict(torch.load(self.s_modelpath))
        self.s_model.to(device)
        self.s_model.to(torch.double)
        self.clmodel.to(torch.double)
        self.s_model.eval()
        for epoch in range(500):
            # 训练
            self.clmodel.train()
            train_epoch_loss = []
            for idx, (data, aug_data, label) in enumerate(self.val_dataloader):
                data = data.to(device)
                aug_data = aug_data.to(device)
                label = label.to(device)
                output1 = self.s_model(aug_data)
                output = self.clmodel(output1)
                self.class_optimizer.zero_grad()
                loss = F.cross_entropy(output, label)
                loss.backward()
                self.class_optimizer.step()
                train_epoch_loss.append(loss.item())
            self.train_loss.append(np.mean(train_epoch_loss))
            # 验证
            self.clmodel.eval()
            val_epoch_loss = []
            epoch_acc = 0.0
            best_acc = 0.0
            conf_matrix = np.zeros((self.classes, self.classes), dtype=float)
            with torch.no_grad():
                for idx, (data, aug_data, label) in enumerate(self.train_dataloader):
                    data = data.to(device)
                    aug_data = aug_data.to(device)
                    label = label.to(device)
                    output1 = self.s_model(aug_data)
                    output = self.clmodel(output1)
                    predict_y = torch.max(output, dim=1)[1]
                    epoch_acc += torch.eq(predict_y, label).sum().item()
                    loss = F.cross_entropy(output, label)
                    val_epoch_loss.append(loss.item())
                    predict_y_np = predict_y.cpu().numpy()
                    labels_np = label.cpu().numpy()
                    for i in range(len(predict_y_np)):
                        conf_matrix[labels_np[i], predict_y_np[i]] += 1
            epoch_acc = epoch_acc / len(self.train_dataset)
            self.val_loss.append(np.mean(val_epoch_loss))
            acc.append(epoch_acc)
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(self.clmodel.state_dict(), '/home/liuyi/MLRFF/res/albert_s.pth')
                # torch.save(self.s_model.state_dict(), '/home/liuyi/MLRFF/res/best/test_student.pth')
                row_sums = conf_matrix.sum(axis=1)
                conf_matrix_prob = conf_matrix / row_sums[:, np.newaxis]
                res_matrix = (conf_matrix_prob * 100).astype(int)
                np.save('/home/liuyi/MLRFF/res/albert_s.npy', res_matrix)
                fig, ax = plt.subplots(figsize=(15, 15))
                cax = ax.matshow(res_matrix, cmap='Blues', interpolation='nearest', vmin=0, vmax=res_matrix.max())
                for i in range(res_matrix.shape[0]):
                    for j in range(res_matrix.shape[1]):
                        text_color = 'white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black'
                        plt.text(j, i, str(res_matrix[i, j]), ha='center', va='center', color=text_color)
                plt.title('Confusion Matrix')
                # plt.colorbar()
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.savefig('/home/liuyi/MLRFF/res/s_CM_albert.png')
                plt.close()
            print("Epoch {}: train_loss = {}, val_loss = {}, acc = {}".format(epoch, self.train_loss[-1],
                                                                              self.val_loss[-1], acc[-1]))
            plt.figure(figsize=(12, 4))
            plt.subplot(121)
            plt.plot(self.train_loss, '-o', label="train_loss")
            plt.plot(self.val_loss, '-o', label="val_loss")
            plt.title("epochs_loss")
            plt.legend()
            plt.subplot(122)
            plt.plot(acc)
            plt.title("Acc")
            plt.savefig('/home/liuyi/MLRFF/res/class_res_albert.png')
            plt.close()

    def identification(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t_model.load_state_dict(torch.load(self.t_modelpath))
        self.t_model.to(device)
        self.t_model.to(torch.double)
        self.t_model.eval()
        self.s_model.load_state_dict(torch.load(self.s_modelpath))
        self.s_model.to(device)
        self.s_model.to(torch.double)
        self.s_model.eval()
        self.clmodel.load_state_dict(torch.load(self.cl_modelpath))
        self.clmodel.to(device)
        self.clmodel.to(torch.double)
        self.clmodel.eval()

        CLS = torch.tensor([0 for _ in range(128)]).unsqueeze(0)
        data_path = '/home/liuyi/MLRFF/data/val/device1/data10.mat'
        sample_data = loadmat(data_path)
        I = np.transpose(sample_data['I'])[0]
        Q = np.transpose(sample_data['Q'])[0]
        sample = np.empty((I.size + Q.size), dtype=I.dtype)
        sample[0::2] = I
        sample[1::2] = Q
        data = torch.tensor(sample)
        data = data.resize(64, 128)
        data = torch.cat((CLS, data), dim=0)
        data = data.to(device)

        t_start = time.time()
        output_t = self.t_model(data)
        output1 = self.clmodel(output_t)
        t_end = time.time()
        t_duration = t_end - t_start
        print(t_duration)

        I = torch.tensor(sample_data['I'])
        Q = torch.tensor(sample_data['Q'])
        sample2 = torch.cat([I.unsqueeze(0), Q.unsqueeze(0)], dim=0)
        data = sample2.squeeze(-1)
        data = data.unsqueeze(0)
        data = data.to(device)

        s_start = time.time()
        output_s = self.s_model(data)
        output2 = self.clmodel(output_s)
        s_end = time.time()
        s_duration = s_end - s_start
        print(s_duration)


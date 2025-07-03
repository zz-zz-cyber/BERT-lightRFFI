import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from model import Bert, contrastiveloss, Classification, AlBert
from dataset import BertDataset, ClassDataset, Test2Dataset, NLDataset, Test3Dataset
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F
from XLNet import XLNet
from GPT2 import GPT2Model

class BertTrainer:
    def __init__(self, train_datapath, val_datapath):
        self.epoch_num = 51
        self.classes = 25
        # self.model = Bert()
        # self.model = XLNet(n_token=128, n_layer=6, n_head=4, d_head=8,
        #                                 d_inner=512, d_model=512,
        #                                 dropout=0, dropatt=0, bi_data=False, attn_type="bi",
        #                                 clamp_len=-1, same_length=False,
        #                                 reuse_len=256, mem_len=384).to("cuda")
        # self.model = GPT2Model()
        self.model = AlBert()
        self.clmodel = Classification(512, 25)
        self.lr = 0.0001
        self.batchsize = 32
        self.train_dataset = Test2Dataset(train_datapath)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batchsize, shuffle=True)
        self.val_dataset = Test2Dataset(val_datapath)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batchsize, shuffle=False)
        self.bert_optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.class_optimizer = Adam(self.clmodel.parameters(), lr=0.0007)
        self.criterion = contrastiveloss
        self.train_loss = []
        self.val_loss = []
        self.save_modelpath = '/home/liuyi/MLRFF/res/albert.pth'
        self.save_figpath = '/home/liuyi/MLRFF/res/loss_albert.png'

    def train(self):
        start_time = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.to(torch.double)
        for epoch in range(self.epoch_num):
            # 训练
            self.model.train()
            train_epoch_loss = []
            for idx, (x, y, label) in enumerate(self.train_dataloader):
                x = x.to(device)
                y = y.to(device)
                output1 = self.model(x)
                output2 = self.model(y)
                self.bert_optimizer.zero_grad()
                loss = self.criterion(output1[:, 0, :], output2[:, 0, :])
                # loss = self.criterion(output1, output2)
                loss.backward()
                self.bert_optimizer.step()
                train_epoch_loss.append(loss.item())
            self.train_loss.append(np.mean(train_epoch_loss))
            # 验证
            self.model.eval()
            val_epoch_loss = []
            with torch.no_grad():
                for idx, (x, y, label) in enumerate(self.val_dataloader):
                    x = x.to(device)
                    y = y.to(device)
                    output1 = self.model(x)
                    output2 = self.model(y)
                    loss = self.criterion(output1[:, 0, :], output2[:, 0, :])
                    # loss = self.criterion(output1, output2)
                    val_epoch_loss.append(loss.item())
            self.val_loss.append(np.mean(val_epoch_loss))
            print("Epoch {}: train_loss = {}, val_loss = {}".format(epoch, self.train_loss[-1], self.val_loss[-1]))
        torch.save(self.model.state_dict(), self.save_modelpath)
        plt.figure(figsize=(6, 4))
        plt.plot(self.train_loss, '-o', label="train loss")
        plt.plot(self.val_loss, '-o', label="val loss")
        plt.title("epochs loss")
        plt.legend()
        plt.savefig(self.save_figpath)
        plt.close()
        end_time = time.time()
        print("Train Time: {}s".format(end_time - start_time))

    def classification(self):
        acc = []
        best_acc = 0.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clmodel.to(device)
        self.clmodel.to(torch.double)
        self.model.load_state_dict(torch.load(self.save_modelpath))
        self.model.to(device)
        self.model.to(torch.double)
        self.model.eval()
        for epoch in range(100):
            # 训练
            self.clmodel.train()
            train_epoch_loss = []
            for idx, (data, aug_data, label) in enumerate(self.train_dataloader):
                data = data.to(device)
                aug_data = aug_data.to(device)
                label = label.to(device)
                output1 = self.model(aug_data)
                output2 = self.clmodel(output1[:, 0, :])
                # output2 = self.clmodel(output1)
                self.bert_optimizer.zero_grad()
                loss = F.cross_entropy(output2, label)
                loss.backward()
                self.bert_optimizer.step()
                train_epoch_loss.append(loss.item())
            self.train_loss.append(np.mean(train_epoch_loss))
            # 验证
            self.clmodel.eval()
            val_epoch_loss = []
            epoch_acc = 0.0
            conf_matrix = np.zeros((self.classes, self.classes), dtype=float)
            with torch.no_grad():
                for idx, (data, aug_data, label) in enumerate(self.val_dataloader):
                    data = data.to(device)
                    aug_data = aug_data.to(device)
                    label = label.to(device)
                    output1 = self.model(aug_data)
                    output2 = self.clmodel(output1[:, 0, :])
                    # output2 = self.clmodel(output1)
                    predict_y = torch.max(output2, dim=1)[1]
                    epoch_acc += torch.eq(predict_y, label).sum().item()
                    loss = F.cross_entropy(output2, label)
                    val_epoch_loss.append(loss.item())
                    predict_y_np = predict_y.cpu().numpy()
                    labels_np = label.cpu().numpy()
                    for i in range(len(predict_y_np)):
                        conf_matrix[labels_np[i], predict_y_np[i]] += 1
            epoch_acc = epoch_acc / len(self.val_dataset)
            self.val_loss.append(np.mean(val_epoch_loss))
            acc.append(epoch_acc)
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(self.clmodel.state_dict(), '/home/liuyi/MLRFF/res/classification_albert.pth')
                torch.save(self.model.state_dict(), '/home/liuyi/MLRFF/res/best/albert.pth')
                row_sums = conf_matrix.sum(axis=1)
                conf_matrix_prob = conf_matrix / row_sums[:, np.newaxis]
                res_matrix = (conf_matrix_prob * 100).astype(int)
                fig, ax = plt.subplots(figsize=(15, 15))
                cax = ax.matshow(res_matrix, cmap='Blues', interpolation='nearest', vmin=0, vmax=res_matrix.max())
                np.save('/home/liuyi/MLRFF/res/albert2class.npy', res_matrix)
                for i in range(res_matrix.shape[0]):
                    for j in range(res_matrix.shape[1]):
                        text_color = 'white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black'
                        plt.text(j, i, str(res_matrix[i, j]), ha='center', va='center', color=text_color)
                plt.title('Confusion Matrix')
                # plt.colorbar()
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.savefig('/home/liuyi/MLRFF/res/ConfusionMatrix_albert.png')
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

        print("Best accury : {}".format(best_acc))

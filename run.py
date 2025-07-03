from train import BertTrainer
import torch
from test_Res import ResTest
from KDtrain import KD

train_datapath = '/home/liuyi/MLRFF/data/train'
val_datapath = '/home/liuyi/MLRFF/data/val'
# BertTrainer(train_datapath, val_datapath).train()
# BertTrainer(train_datapath, val_datapath).classification()
# KD(train_datapath, val_datapath).train()
# KD(train_datapath, val_datapath).classification()
KD(train_datapath, val_datapath).identification()
#BertTest(train_datapath, val_datapath).train()
#ResTest(train_datapath, val_datapath).train()
#ResTest(train_datapath, val_datapath).classification()
#print(torch.cuda.is_available())

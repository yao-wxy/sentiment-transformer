class Trainer():
    def __init__(self,model,optimizer,criterion,train_loader,val_loader,device):
        self.model=model
        self.optimizer=optimizer
        self.criterion=criterion
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.device=device
        self.best_val_loss=float('inf')
        self.count=0


    def train_for_one_epoch(self):
        # 训练一轮的逻辑
        self.model.train()
        train_loss=0.0
        train_num=0
        for x,y in self.train_loader:   # x代表每一批的样本，y代表一批样本的标签，是一个张量，shape是（样本数，标签）
            x=x.to(self.device)
            y=y.to(self.device)
            self.optimizer.zero_grad()
            logits=self.model(x)
            loss=self.criterion(logits,y)
            loss.backward()
            self.optimizer.step()
            bs=y.size(0)
            train_loss+=loss.item() *bs
            train_num+=bs
        avg_loss=train_loss/train_num
        return avg_loss

    def validation(self):
        # 验证集的逻辑
        self.model.eval()
        # 进入推理模式
        total_loss=0.0
        total_num=0
        with torch.no_grad():
            for x_val, y_val in self.val_loader:
                x_val=x_val.to(self.device)
                y_val=y_val.to(self.device)
                logits=self.model(x_val)
                val_loss=self.criterion(logits,y_val)   # 代表一批loss的平均值
                bs=y_val.size(0)    # 代表取y_val形状的第一维，样本数
                total_loss+= bs*val_loss.item()
                total_num+=bs
        avg_loss=total_loss/total_num    # 平均损失值
        return avg_loss

    def early_stopped(self, val_loss, patience):
        if self.best_val_loss-val_loss>0.001:
            self.best_val_loss=val_loss
            self.count=0

            os.makedirs('../cache', exist_ok=True)
            torch.save({
                'model_state_dict':self.model.state_dict(),
                'best_val_loss':self.best_val_loss
            }, '../cache/best_model.pt')

        else:
            self.count+=1

        print(f'当前验证集最小损失值是：{self.best_val_loss}\n')

        if self.count>=patience:
            return True     # 早停

        return False    # 没有遇到早停时，返回本轮结果

import pickle
import torch.cuda
from datasets.mixed_dataset import Cleaner
from models.emotional_classifier import Emotion_Classifier
import torch
import torch.nn as nn
import os
import random
import numpy as np

# 1.读取和清洗数据
# 2.训练模型
# 3.保存模型
# 4.用户交互

SAVE = False
DO_TRAIN = False
# 是否显示调试结果
VERBOSE_MODE = False

# 设置随机种子，固定随机性。

seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

path=r"C:\Users\yqy\Desktop\情感类数据集"
# 保存常用参数，每一步函数的返回结果
cleaner=Cleaner()
full_texts=cleaner.read_and_clean_data(path)
texts,labels=cleaner.label_cleaner(full_texts)

os.makedirs('../cache', exist_ok=True)
if SAVE:
    vocab = cleaner.build_vocab(texts)  # 生成一个词表
    os.makedirs('../cache', exist_ok=True)
    # 在函数里面已经保存过了，无需再次保存。

else:
    # 如果需要直接读取词表:
    with open('../cache/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

# 读取当前词表大小
vocab_size = len(vocab)
if VERBOSE_MODE:
    print(vocab_size)

encoded_texts=cleaner.encode_texts(texts,vocab)
padded_texts=cleaner.padding(encoded_texts)
train_data,val_data=cleaner.classifier_train_and_validation(labels,padded_texts)
train_loader,val_loader=cleaner.MyDataLoader(train_data=train_data,val_data=val_data,batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Emotion_Classifier(vocab_size=vocab_size, num_heads=8, num_classes=2, d_model=128,max_len=40).to(device)

# 定义要用的训练参数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay=2e-3)
trainer = Trainer(model,optimizer,criterion,train_loader,val_loader,device)

# 进行训练，开始。
epochs = 200
if DO_TRAIN:
    for epoch in range(1,epochs+1):
        # 输出每一轮的训练结果
        train_loss = trainer.train_for_one_epoch()
        val_loss = trainer.validation()
        # 判断是否早停
        judge = trainer.early_stopped(val_loss,patience=6)
        if judge == False:
            print(f'训练轮数：{epoch},训练损失：{train_loss:.4f};'
              f',验证损失：{val_loss:.4f}')

        else:
            print('早停了，不能训练了。\n\n\n')
            break

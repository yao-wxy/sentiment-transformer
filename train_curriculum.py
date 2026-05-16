from datasets.curriculum_dataset import Clean_My_Data, Classifier_My_Data, Data_to_Torch
from models.emotional_classifier import Emotion_Classifier
import os
import torch
import torch.nn as nn
import pickle
import random
import numpy as np


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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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

            os.makedirs('../new_cache', exist_ok=True)
            torch.save({
                'model_state_dict':self.model.state_dict(),
                'best_val_loss':self.best_val_loss
            }, '../new_cache/best_model.pt')

        else:
            self.count+=1

        print(f'当前验证集最小损失值是：{self.best_val_loss}')

        if self.count>=patience:
            return True     # 早停

        return False    # 没有遇到早停时，返回本轮结果


# -----------------------------------各种调试开关控制----------------------------------------------------------------------
# 1) 决定是否重写保存词表
SAVE = False

# 2) 判断是否开始训练
DO_TRAIN = False

# ----------------------------------使用的参数调用----------------------------------------------------------------------
# 设置随机种子，固定随机性。
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 罗列所有使用的参数
path = r'C:\Users\yqy\Desktop\情感类数据集'

cleaner = Clean_My_Data()
simple_data, complex_data = cleaner.prepare_data(path)
cleaned_simple_data = cleaner.clean_data(simple_data)
cleaned_complex_data = cleaner.clean_data(complex_data)  # 有标签数据

classifier = Classifier_My_Data()
all_contents = classifier.get_all_contents_from_complex_neg_files(path)
hard_contents = classifier.get_hard_contents(path)

val_data = classifier.get_val_data(cleaned_simple_data, cleaned_complex_data, path)
train_simple, train_complex = classifier.get_train_data(val_data, cleaned_simple_data,
                                                        cleaned_complex_data)  # 切分出的训练集
train_stage1 = classifier.stage_1(train_simple, train_complex)
train_stage2 = classifier.stage_2(train_simple, train_complex, all_contents, hard_contents, n=1)
train_stage3 = classifier.stage_3(train_simple, train_complex)  # 三阶段各自的训练集数据

torch_data = Data_to_Torch()
cleaned_data = cleaned_simple_data + cleaned_complex_data

if SAVE:
    vocab = torch_data.build_vocab(cleaned_data)
    os.makedirs('../new_cache', exist_ok=True)
    # 在函数里面已经保存过了，无需再次保存。

else:
    # 如果需要直接读取词表:
    with open('../new_cache/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

# 读取当前词表大小
vocab_size = len(vocab)

MAX_LENGTH = 40  # 最大长度
encoded_data = torch_data.encoded(cleaned_data, vocab)
padded_data = torch_data.padding(encoded_data, MAX_LENGTH)  # 编码补零后的全部数据

encoded_val = torch_data.encoded(val_data, vocab)
padded_val = torch_data.padding(encoded_val, MAX_LENGTH)  # 编码补零后的验证集数据

# 把每个阶段的训练集数据编码补零以后，分批处理。
encoded_stage1 = torch_data.encoded(train_stage1, vocab)
padded_stage1 = torch_data.padding(encoded_stage1, MAX_LENGTH)  # 补零后的第一阶段训练集数据
train_loader1, val_loader1 = torch_data.MyDataLoader(padded_stage1, padded_val, batch_size=64)

encoded_stage2 = torch_data.encoded(train_stage2, vocab)
padded_stage2 = torch_data.padding(encoded_stage2, MAX_LENGTH)  # 补零后的第二阶段训练集数据
train_loader2, val_loader2 = torch_data.MyDataLoader(padded_stage2, padded_val, batch_size=64)

encoded_stage3 = torch_data.encoded(train_stage3, vocab)
padded_stage3 = torch_data.padding(encoded_stage3, MAX_LENGTH)  # 补零后的第三阶段训练集数据
train_loader3, val_loader3 = torch_data.MyDataLoader(padded_stage3, padded_val, batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Emotion_Classifier(vocab_size=vocab_size, num_heads=8, num_classes=2, d_model=128,
                           max_len=MAX_LENGTH).to(device)

# -----------------------------------训练模型模块---------------------------------------------------------------------
# 定义要用的训练参数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
trainer = Trainer(model, optimizer, criterion, train_loader1, val_loader1, device)

# 1) 第一阶段训练流程:
epochs = 20
if DO_TRAIN:
    for epoch in range(1, epochs + 1):
        # 输出每一轮的训练结果
        train_loss = trainer.train_for_one_epoch()
        val_loss = trainer.validation()
        # 判断是否早停
        judge = trainer.early_stopped(val_loss, patience=6)
        if judge == False:
            print(f'训练轮数：{epoch},训练损失：{train_loss:.4f};'
                  f',验证损失：{val_loss:.4f}\n')

        else:
            print('早停了，不能训练了。\n\n\n')
            break

    print('第一阶段训练完成\n')

# 2) 第二阶段训练流程:

# 加载当前的模型
checkpoint = torch.load('../new_cache/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# 定义要用的训练参数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
trainer = Trainer(model, optimizer, criterion, train_loader2, val_loader2, device)

epochs = 20
if DO_TRAIN:
    for epoch in range(1, epochs + 1):
        # 输出每一轮的训练结果
        train_loss = trainer.train_for_one_epoch()
        val_loss = trainer.validation()
        # 判断是否早停
        judge = trainer.early_stopped(val_loss, patience=4)
        if judge == False:
            print(f'训练轮数：{epoch},训练损失：{train_loss:.4f};'
                  f',验证损失：{val_loss:.4f}\n')

        else:
            print('早停了，不能训练了。\n\n\n')
            break

    print('第二阶段训练完成\n')

# 3) 第三阶段训练流程:

# 加载当前的模型
checkpoint = torch.load('../new_cache/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# 定义要用的训练参数
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
trainer = Trainer(model, optimizer, criterion, train_loader3, val_loader3, device)

epochs = 10
if DO_TRAIN:
    for epoch in range(1, epochs + 1):
        # 输出每一轮的训练结果
        train_loss = trainer.train_for_one_epoch()
        val_loss = trainer.validation()
        # 判断是否早停
        judge = trainer.early_stopped(val_loss, patience=6)
        if judge == False:
            print(f'训练轮数：{epoch},训练损失：{train_loss:.4f};'
                  f',验证损失：{val_loss:.4f}\n')

        else:
            print('早停了，不能训练了。\n\n\n')
            break

    print('第三阶段训练完成\n')


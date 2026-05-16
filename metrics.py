import config
from training.train_curriculum import train_loader1,train_loader2,train_loader3
from training.train_mixed import train_loader
from load_model import load_model

curriculum_model = load_model(file_name='../new_cache/best_model.pt',vocab_size=config.curriculum_vocab_size)
mixed_model = load_model(file_name='../cache/best_model.pt',vocab_size=config.mixed_vocab_size)
import torch


def accuracy(outputs, labels):

    predicted = torch.argmax(outputs, dim=1)

    correct = (predicted == labels).sum().item()

    return correct / len(labels)

# ----------------------------------------------------------------------------------------------------------------------
print('课程学习法：')
for inputs,labels in train_loader1:
    outputs = curriculum_model(inputs)
    acc = accuracy(outputs,labels)
    print(f'第一阶段准确率为{acc}')

for inputs,labels in train_loader2:
    outputs = curriculum_model(inputs)
    acc = accuracy(outputs,labels)
    print(f'第二阶段准确率为{acc}')

for inputs,labels in train_loader3:
    outputs = curriculum_model(inputs)
    acc = accuracy(outputs,labels)
    print(f'第三阶段准确率为{acc}')

# ----------------------------------------------------------------------------------------------------------------------
for inputs,labels in train_loader:
    outputs = mixed_model(inputs)
    acc = accuracy(outputs,labels)
    print(f'混合训练的准确率为{acc}')
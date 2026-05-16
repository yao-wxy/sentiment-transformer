import torch
import torch.nn as nn
import math
from models.encoder import TRY_ENCODER

# 分类头
class Emotion_Classifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_classes, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  # 词表大小和向量维度。查找你的数据的词数量，每个词的向量维度。
        self.encoder = TRY_ENCODER(d_model, num_heads)      # 向量维度和头数量
        self.classifier = nn.Linear(d_model, num_classes)   # 代表线性层接收向量维度，转化为分类数量
        # --- 创建位置编码矩阵，注册为不参与训练的缓冲区 ---
        pe = torch.zeros(max_len, d_model)  # 位置编码表，每行代表一个位置
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 位置索引 [0,1,2,...]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 偶数维度用 sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数维度用 cos
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 增加 batch 维度，变成 [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 存起来，不参与梯度更新

    def forward(self, x):
        # x 形状: [batch_size, seq_len]
        seq_len = x.size(1)

        x = self.embedding(x)  # [batch, seq_len, d_model] #词嵌入

        # --- 新增：加上位置编码（只取当前序列长度对应的部分）---
        x = x + self.pe[:, :seq_len, :]  # 广播相加

        x = self.encoder(x)     # 编码器处理向量，提高转化质量
        x = x.mean(dim=1)       # 平均池化，将一个词序列转化为一个向量，平均化。dim代表的是Dimension(维度)
        return self.classifier(x)
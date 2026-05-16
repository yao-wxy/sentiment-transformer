import torch.nn as nn
import torch.nn.functional as F
import math
# 编码器
class TRY_ENCODER(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model//num_heads  # 计算每个头的维度
        # 检查能否整除
        assert d_model%num_heads==0,'d_model必须被num_heads整除'  # 断言的写法
        # 定义权重矩阵
        self.W_Q= nn.Linear(d_model,d_model)
        self.W_K= nn.Linear(d_model,d_model)
        self.W_V= nn.Linear(d_model,d_model)
        self.out= nn.Linear(d_model,d_model)

        # 前馈网络
        d_ff= d_model*4  # 中间层扩大4倍。目的是增加模型的表达能力和非线性
        self.ffn= nn.Sequential(  # 是一个容器，用于按顺序执行一系列网络层
            nn.Linear(d_model,d_ff),  # 第一层，扩大
            nn.ReLU(),  # 激活函数
            nn.Linear(d_ff,d_model)  # 第二层，缩回原尺寸
        )
        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)

        self.dropout= nn.Dropout(0.1)  # 随机使一部分神经元失活，正则化

    def forward(self,x):
        # 获取输入形状
        batch_size,seq_len,_=x.shape

        # 保存第一份残差
        residual1=x
        # 先线性变换
        Q=self.W_Q(x)
        K=self.W_K(x)
        V=self.W_V(x)
        # 再分头
        Q=Q.view(batch_size,seq_len,self.num_heads,self.head_dim)
        Q=Q.transpose(1,2)  # 换轴。样本数，头数，每个头看的词数，每词的维度。
        K=K.view(batch_size,seq_len,self.num_heads,self.head_dim)
        K=K.transpose(1,2)
        V=V.view(batch_size,seq_len,self.num_heads,self.head_dim)
        V=V.transpose(1,2)
        # 传入线性层以后，计算Q和K的转置作为注意力分数
        a=Q@(K.transpose(-2,-1))
        # 将它除以根号d_model,缩放数值大小，防止softmax饱和
        b=a/(math.sqrt(self.head_dim))
        # 用softmax函数计算概率，得到一个权重矩阵和V相乘计算出一个矩阵代表加权求和
        c=F.softmax(b,dim=-1)
        d=c@V
        d=d.transpose(1,2)  # 将头数和词数转置，使得每个词的所有头显示。
        d=d.contiguous()  # 重新拷贝数据，让内存连续排列
        d=d.view(batch_size,seq_len,self.d_model)  # 输入形状转化为 将维度打包相乘。
        d=self.out(d)  # 线性转换

        d= self.dropout(d)
        x= self.norm1(residual1+d)  # 第一次残差连接

        residual2= x    # 保存残差

        # 前馈网络
        ffn_output=self.ffn(x)
        ffn_output=self.dropout(ffn_output)
        # 第二次残差连接
        x= self.norm2(residual2+ffn_output)
        return x

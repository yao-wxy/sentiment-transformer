"""
1.数据读取与清洗
2.分词器分词，将词表编码
3.切分训练集和验证集
4.装载器分批数据
5.训练循环
6.验证循环
7.早停并保存best_model
"""
import random

import torch
from docx import Document
import jieba
import pickle
import os

from torch.utils.data import DataLoader

# 清洗管道
class Cleaner():
    def __init__(self):
        pass
    # 1.数据读取与清洗：
    def read_and_clean_data(self,path):
        full_texts = []
        all_files = os.listdir(path)
        docx_files = [f for f in all_files if f.endswith('.docx') and not f.startswith('~$')]

        for name in docx_files:
            fp = os.path.join(path, name)
            doc = Document(fp)

            for para in doc.paragraphs:
                t = para.text.strip()   # 移除首尾空白字符
                if t:
                    full_texts.append(t)
        return full_texts       # 将数据保存进列表里

    def label_cleaner(self,full_texts):
        texts=[]
        labels=[]
        for sentence in full_texts:
            if sentence[-1]=='爱' or sentence[-1]=='0' or sentence[-1]=='1':
                sentences=sentence[:-2]
                texts.append(sentences)
            else:
                sentences=sentence[:-3]
                texts.append(sentences)     # 去除标签的原始文本数据
        # 文本清洗结束
        positive = ['快乐', '幸福', '爱', '期待', '希望', '满意', '赞美', '肯定', '自豪', '自信', '感动', '感激']
        negative = ['愤怒', '悲伤','难过', '痛苦', '抑郁','失落', '厌恶', '恐惧', '失望', '不满']
        for sentence in full_texts:
            if sentence[-2:] in positive or sentence[-1]=='爱' or sentence[-1]=='1':
                labels.append(1)

            elif sentence[-2:] in negative or sentence[-1]=='0':
                labels.append(0)
            else:
                pass  # 标签分类好了
        return texts,labels

    # 2.分词器分词，制作词表
    def build_vocab(self,texts,max_length=39):
        # 把文本词语编号，放进一个叫vocab的大字典里。保存这个大字典，词表构建完成。
        vocab = {'<PAD>': 0, 'UNK': 1}
        count=0
        for sentence in texts:
            if count==0:
                count+=1
            cut_sentence=jieba.lcut(sentence)    # 用结巴分词
            # 用jieba把每一个句子分词
            for word in cut_sentence:
                if word not in vocab:
                    vocab[word]= len(vocab)
        with open('../cache/vocab.pkl', 'wb')as f:
            pickle.dump(vocab,f)  # 写入二进制，把对象存进去方便下次读取。
        return vocab

    # 给具体数据编码
    def encode_texts(self,texts,vocab):
        encoded_texts=[]
        for sentence in texts:  # 一句话
            word_id = []
            cut_sentence=jieba.lcut(sentence)   # 切分词的一句话
            for word in cut_sentence:
                if word in vocab:   # 一个词
                    word_id.append(vocab[word])
                else:
                    word_id.append(vocab['UNK'])
            encoded_texts.append(word_id)
        return encoded_texts

    # 补零
    def padding(self,encoded_texts,max_length=40):
        padded_texts=[]
        for i in encoded_texts:
            if len(i)<max_length:
                i=i+[0]*(max_length-len(i))
            else:
                i=i[:max_length]
            padded_texts.append(i)
        return padded_texts

# 3.切分训练集和验证集
    import random
    def classifier_train_and_validation(self,labels,padded_texts):
        data0=[]
        data1=[]
        for i in range(len(labels)):
            text=padded_texts[i]
            label=labels[i]
            if label==0:
                data0.append((text,0))
            else:
                data1.append((text,1))
        # 随机抽签，各选10%条当作验证集，其余都是训练集数据
        k=int(0.1* len(labels))
        val_data0= random.sample(data0,k)
        val_data1= random.sample(data1,k)
        val_data=val_data0+val_data1
        train_data0=[]
        train_data1=[]
        for i in data0:
            if i not in val_data0:
                train_data0.append(i)
        for i in data1:
            if i not in val_data1:
                train_data1.append(i)
        train_data=train_data0+train_data1
        return train_data,val_data

    # 4.装载器分批数据
    def MyDataLoader(self,train_data,val_data,batch_size):
        train_samples=[]
        val_samples=[]

        for i in range(len(train_data)):
            x=torch.tensor(train_data[i][0],dtype=torch.long)      # 文本
            y=torch.tensor(train_data[i][1],dtype=torch.long)      # 标签
            train_samples.append((x,y))

        for j in range(len(val_data)):
            x=torch.tensor(val_data[j][0],dtype=torch.long)        # 文本
            y=torch.tensor(val_data[j][1],dtype=torch.long)        # 标签
            val_samples.append((x,y))

        train_loader=DataLoader(train_samples,batch_size=batch_size,shuffle=True)
        val_loader=DataLoader(val_samples,batch_size=batch_size,shuffle=False)

        return train_loader,val_loader

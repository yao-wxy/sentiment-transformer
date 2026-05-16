#采用课程学习法，把样本分为简单数据和复杂数据，分阶段训练，测试效果。
'''
1.读取数据，放进不同的列表里。
2.用数据装载器按照不同比例混合数据。
3.进行训练。第一次用100%简单数据，第二次用50%简单+50%复杂数据，第三次用20%简单+80%复杂数据。
'''

import os
import pickle
import random

import jieba
import torch
from docx import Document
from torch.utils.data import DataLoader

DEBUG=False
class Clean_My_Data():
    def __init__(self):
        pass
    def read_docx(self,file_path):  # 文件路径
        doc=Document(file_path)
        contents=[]
        for para in doc.paragraphs:
            if para.text.strip():
                contents.append(para.text)
        return contents     # list[str]

    def prepare_data(self,path):    # 文件夹路径
        simple_path=['简单负面数据集汇总.docx','简单正面数据集.docx']
        simple_data=[]
        complex_data=[]

        all_files= os.listdir(path)
        docx_files=[f for f in all_files if f.endswith('.docx') and not f.startswith('~$')]

        for file_name in docx_files:
            total_path=os.path.join(path,file_name)
            try:
                contents=self.read_docx(total_path)
            except Exception as e:
                print(f'读取文件{file_name}失败：{e}')
                continue
            if file_name in simple_path:
                simple_data.extend(contents)
            else:
                complex_data.extend(contents)
        if DEBUG:
            print(f'\n完成：{len(simple_data)}个简单数据，{len(complex_data)}个复杂数据。')
        return simple_data,complex_data     # 简单数据和复杂数据列表建好了。

    def clean_data(self,data):
        cleaned_data=[]
        positive = ['快乐', '幸福', '爱', '期待', '希望', '满意', '赞美', '肯定', '自豪', '自信', '感动', '感激']
        negative = ['愤怒', '悲伤', '难过', '痛苦', '抑郁', '失落', '厌恶', '恐惧', '失望', '不满']
        for sentence in data:
            if sentence[-1]=='爱' or sentence[-1]=='1':
                text=sentence[:-2]
                cleaned_data.append((text,1))

            elif sentence[-2:] in positive:
                text=sentence[:-3]
                cleaned_data.append((text,1))

            elif sentence[-1]=='0':
                text=sentence[:-2]
                cleaned_data.append((text,0))

            elif sentence[-2:] in negative:
                text=sentence[:-3]
                cleaned_data.append((text,0))

        return cleaned_data     # (texts,labels)


class Classifier_My_Data():
    def __init__(self):
        pass

    def get_all_contents_from_complex_neg_files(self,path):
        complex_neg_files = ['悲伤低落数据集.docx', '失望愤怒集1.docx', '失望愤怒集2.docx', '厌恶恐惧数据集.docx', '隐含负面集.docx']
        all_contents=[]

        for file_name in os.listdir(path):
            total_path = os.path.join(path, file_name)
            if not os.path.isfile(total_path) or not file_name.endswith('.docx'):
                continue
            if DEBUG:
                print(f'读取：{file_name}')

            if file_name in complex_neg_files:
                cleaner = Clean_My_Data()
                content = cleaner.read_docx(total_path)
                content = cleaner.clean_data(content)
                # [(texts,labels),(texts,labels),...]
                all_contents.append(content)
        return all_contents     # [content1,content2,content3,...]

    def get_hard_contents(self,path):
        hard_files=['失望愤怒集1.docx']
        hard_contents=[]
        for file_name in os.listdir(path):
            total_path = os.path.join(path, file_name)
            if not os.path.isfile(total_path) or not file_name.endswith('.docx'):
                continue
            if file_name in hard_files:
                cleaner = Clean_My_Data()
                content = cleaner.read_docx(total_path)
                content = cleaner.clean_data(content)
                hard_contents.extend(content)
        return hard_contents        # [( ),( ),( ),...]

    def get_val_data(self,simple_data,complex_data,path):
        # 把数据分为简单正面，简单负面，复杂正面，复杂负面四个层级，分层抽样，各抽20%。
        # 其中，复杂负面数据由四个文件分别抽取20%组成，确保情绪类别均衡。
        # 把这些数据全部作为验证集数据，并返回结果。
        simple_pos=[]
        simple_neg=[]
        complex_pos=[]
        complex_neg=[]

        classifier=Classifier_My_Data()
        contents=classifier.get_all_contents_from_complex_neg_files(path)
        for content in contents:
            content = random.sample(content,len(content) // 5)
            complex_neg.extend(content)

        for data in simple_data:
            if data[1] == 1:
                simple_pos.append(data)
            if data[1] == 0:
                simple_neg.append(data)

        for data in complex_data:
            if data[1] == 1:
                complex_pos.append(data)

        m = len(simple_pos) // 5
        n = len(simple_neg) // 5
        l = len(complex_pos) // 5

        # 验证集数据
        simple_pos= random.sample(simple_pos,m)
        simple_neg= random.sample(simple_neg,n)
        complex_pos= random.sample(complex_pos,l)

        # 计算数据数量
        if DEBUG:
            print(f'验证集数据数量：简单正面{len(simple_pos)},简单负面{len(simple_neg)},复杂正面{len(complex_pos)},复杂负面：{len(complex_neg)}')

        val_data = simple_pos + simple_neg + complex_pos + complex_neg
        return val_data

    def get_train_data(self,val_data,simple_data,complex_data):
        # 训练集数据
        total_data = simple_data + complex_data
        train_data = [x for x in total_data if x not in val_data]

        train_simple=[]
        train_complex=[]

        # 切分简单数据和复杂数据
        for data in train_data:
            if data in simple_data:
                train_simple.append(data)
            elif data in complex_data:
                train_complex.append(data)

        return train_simple,train_complex

    def stage_1(self,train_simple,train_complex):
        # 简单：复杂=4：1
        if not train_simple:
            return '简单训练集数据为空，请核实情况。'
        else:
            k=len(train_simple)
            j=k//4
            train_complex=random.sample(train_complex,j)
            train_stage1= train_simple + train_complex
            if not train_stage1:
                return '无法添加第一阶段训练集数据，具体原因待核查。'
            if DEBUG:
                print('第一阶段训练集数据添加完成！')
            return train_stage1

    def stage_2(self,train_simple,train_complex,all_contents,hard_contents,n):
        # 简单：复杂=1：1
        # 先把困难样本筛选一遍，筛选出训练集内的数据
        new_hard_contents=[]
        for sentence in hard_contents:
            if sentence in train_complex:
                new_hard_contents.append(sentence)      # [(text,label),...]
        length = int(len(new_hard_contents) * n)
        if DEBUG:
            print(f'困难文本长度为{length}')
        new_hard_contents = random.sample(new_hard_contents, length)

        # 再把其他三个文件都筛选一遍，更新all_contents,保证都属于训练集数据，不能重复使用。
        new_all_contents=[]    # 目标是，[[],[],[],[]]
        new_count=[]
        for contents in all_contents:
            if set(contents) == set(hard_contents):
                contents= new_hard_contents
                new_all_contents.append(contents)
                new_count.append(len(contents))
            else:
                tem_contents = []
                for sentence in contents:
                    if sentence in train_complex:
                        tem_contents.append(sentence)
                new_all_contents.append(tem_contents)
                new_count.append(len(tem_contents))

        train_complex_neg=[]
        train_complex_pos=[]
        train_simple_neg=[]
        train_simple_pos=[]
        for data in train_complex:
            # print(type(data),type(data[0]))
            if data[1] == 1:
                train_complex_pos.append(data)
        for data in train_simple:
            if data[1] == 1:
                train_simple_pos.append(data)
            elif data[1]==0:
                train_simple_neg.append(data)
        for contents in new_all_contents:
            train_complex_neg.extend(contents)

        # 修正train_complex
        train_complex = train_complex_neg + train_complex_pos

        # 定义最小分配量，以此为基准
        half = min(len(train_simple), len(train_complex))

        # 按照比例抽样。
        # 简单数据/复杂数据
        a1 = len(train_complex) / len(train_simple)

        # 简单数据：负面/正面
        a2 = len(train_simple_neg) / len(train_simple_pos)
        k11=len(train_simple_neg)/len(train_simple)
        k22=len(train_simple_pos)/len(train_simple)

        # 复杂数据：负面/正面
        a3 = len(train_complex_neg) / len(train_complex_pos)
        k1= len(train_complex_neg)/len(train_complex)   # 复杂负面/复杂总体
        k2= len(train_complex_pos)/len(train_complex)   # 复杂正面/复杂总体

        if DEBUG:
            print('第二阶段训练数据核查中...')
            print(f'当前还未进行处理。有{len(train_simple)}条简单的训练数据，{len(train_complex)}条复杂的训练数据。')
            if not train_simple or not train_complex:
                return '训练集数据中简单数据或复杂数据为空，请核查情况。'
            elif a1>=2.5:
                print(f'简单的数据太少了。复杂是简单的{a1}倍')
            elif a1<=0.5:
                print(f'复杂的数据太少了。复杂是简单的{a1}倍')
            else:
                print('暂未发现简单数据与复杂数据的比例不均衡问题。')

            # 检查边界情况
            if a2 >= 2.5:
                print(
                    f'警告！！！\n检测到目前的简单训练数据中，一共有{len(train_simple_neg)}条负面数据，{len(train_simple_pos)}条正面数据。')
                print(f'正面的数据太少了。负面是正面数据的{a2}倍')
            elif a2 <= 0.5:
                print(
                    f'警告！！！\n检测到目前的简单训练数据中，一共有{len(train_simple_neg)}条负面数据，{len(train_simple_pos)}条正面数据。')
                print(f'负面的数据太少了。负面是正面数据的{a2}倍')
            elif a3 >= 2.5:
                print(
                    f'警告！！！\n检测到目前的复杂训练数据中，一共有{len(train_complex_neg)}条负面数据，{len(train_complex_pos)}条正面数据。')
                print(f'正面的数据太少了。负面是正面数据的{a3}倍')
            elif a3 <= 0.5:
                print(
                    f'警告！！！\n检测到目前的复杂训练数据中，一共有{len(train_complex_neg)}条负面数据，{len(train_complex_pos)}条正面数据。')
                print(f'负面的数据太少了。负面是正面数据的{a3}倍')

        # 1) 如果复杂更多
        # 如果复杂负面是复杂正面的两倍以上，取两倍计。
        if len(train_simple)<len(train_complex):
            # 开始添加数据
            if DEBUG:
                print('暂未发现正负数据失衡问题。\n开始添加数据中...')
            # 不用抽样，按比例添加进复杂数据中。
            k3=int(k2*half)  # 复杂正面
            train_complex_pos=random.sample(train_complex_pos,k3)
            if len(train_complex_neg)>2*len(train_complex_pos):
                k4= len(train_complex_pos)*2
            else:
                k4= half-k3       # 复杂负面
            new_train_complex_neg = []
            # total_num 是所有复杂负面的总数量
            total_num = sum(new_count)
            for content, num in zip(new_all_contents, new_count):
                if total_num == 0 or num == 0:
                    continue
                # 当前文件应该贡献多少条
                take = int(num / total_num * k4)
                if take > 0:
                    sampled = random.sample(content, min(take, len(content)))
                    new_train_complex_neg.extend(sampled)
            train_stage_2= train_simple + train_complex_pos + new_train_complex_neg
            train_complex= new_train_complex_neg + train_complex_pos
            if not train_stage_2:
                return '无法添加第二阶段数据，具体原因待查。'
            if DEBUG:
                print('第二阶段训练集数据添加完成！')
                print(f'一共有{len(train_simple)}条简单数据，{len(train_complex)}条复杂数据参与训练')
            return train_stage_2

        # 2) 如果简单更多
        if len(train_simple)>len(train_complex):
            # 打印提示语，但不做任何处理。
            if DEBUG:
                print('数据处理有问题，简单数据多于或等于复杂数据。请调整。')
                # 如果出现了，就按照标准分类。
                # 开始添加数据
                print('暂未发现正负数据失衡问题。\n开始添加数据中...')
            # 正负数据按照各自比例加入
            need= half
            k3= int(need*k11)   # 需要的简单负面数据数量
            k4= int(need*k22)   # 需要的简单正面数据数量
            train_simple_neg= random.sample(train_simple_neg,k3)
            train_simple_pos= random.sample(train_simple_pos,k4)
            train_simple= train_simple_pos + train_simple_neg
            train_stage_2= train_complex+train_simple_pos+train_simple_neg
            if not train_stage_2:
                return '无法添加第二阶段数据，具体原因待查。'
            if DEBUG:
                print('第二阶段训练集数据添加完成！')
                print(f'一共有{len(train_simple)}条简单数据，{len(train_complex)}条复杂数据参与训练')
            return train_stage_2

        # 3) 如果简单等于复杂数量或者相差无几
        if (len(train_simple) == len(train_complex) or len(train_simple)-len(train_complex)<=25
            or len(train_complex)-len(train_simple)<=25):
            if DEBUG:
                print('暂未发现正负数据失衡问题。\n开始添加数据中...')
            # 直接开始添加数据
            train_stage_2 = train_simple + train_complex
            if not train_stage_2:
                return '无法添加第二阶段数据，具体原因待查。'
            if DEBUG:
                print('第二阶段训练集数据添加完成！')
                print(f'一共有{len(train_simple)}条简单数据，{len(train_complex)}条复杂数据参与训练')
            return train_stage_2
        return '无法添加第二阶段数据，可能原因是：正负比例不均衡。'

    def stage_3(self,train_simple,train_complex):
        # 目前一共有580条正面数据，855条负面数据。
        # 简单：复杂=1：4
        if DEBUG:
            print(f'当前一共有{len(train_simple)}条简单数据参与训练，{len(train_complex)}条复杂数据参与训练。')
        if len(train_simple)*4 < len(train_complex):
            k=len(train_complex)
            j=k//4
            train_simple=random.sample(train_simple,j)
        train_stage_3 = train_simple+train_complex

        return train_stage_3



class Data_to_Torch():
    def __init__(self):
        pass

    #建立词表
    def build_vocab(self,cleaned_data):
        tokenized=[]
        vocab = {'<PAD>': 0, 'UNK': 1}
        for text,label in cleaned_data:
            tokens=jieba.lcut(text)
            tokenized.append((tokens,label))
        # print(tokenized[:3])
        for tokens,label in tokenized:
            for token in tokens:
                if token not in vocab:
                    vocab[token]= len(vocab)
        with open('../new_cache/vocab.pkl', 'wb')as f:
            pickle.dump(vocab,f)        # 写入二进制，把对象存进去方便下次读取。
        return vocab

    # 编码
    def encoded(self,cleaned_data,vocab):
        encoded_data = []
        # print(type(cleaned_data),type(cleaned_data[0]))
        for sentence,label in cleaned_data:
            word_id = []
            cut_sentence = jieba.lcut(sentence)
            for word in cut_sentence:
                if word in vocab:
                    word_id.append(vocab[word])
                else:
                    word_id.append(vocab['UNK'])
            encoded_data.append((word_id,label))
        return encoded_data

    # 补零
    def padding(self,encoded_data,max_length=39):
        padded_data = []
        # print(type(encoded_data),type(encoded_data[0]))
        for text,label in encoded_data:
            if len(text) < max_length:
                text = text + [0] * (max_length - len(text))
            else:
                text = text[:max_length]
            padded_data.append((text,label))
        # print(f'补零以后第一条编码后的数据是：{padded_data[0]}')
        return padded_data

    # 数据装载器。分批进行。
    def MyDataLoader(self,train_data,val_data,batch_size):
        train_samples = []
        val_samples = []

        for i in range(len(train_data)):
            x = torch.tensor(train_data[i][0], dtype=torch.long)  # 文本
            y = torch.tensor(train_data[i][1], dtype=torch.long)  # 标签
            train_samples.append((x, y))

        for j in range(len(val_data)):
            x = torch.tensor(val_data[j][0], dtype=torch.long)  # 文本
            y = torch.tensor(val_data[j][1], dtype=torch.long)  # 标签
            val_samples.append((x, y))

        train_loader = DataLoader(train_samples,batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_samples, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
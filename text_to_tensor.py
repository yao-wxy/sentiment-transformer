# 把用户的话交给词表一一比对得出一个向量列表，然后用我的模型进行推理。输出二分类结果
# 去掉首尾空白格
import torch,jieba
def text_to_tensor(user_input,vocab):
    user_text = user_input.strip()

    # 分词
    user_texts = jieba.lcut(user_text)

    # 编码
    user_encoded_text = []
    for word in user_texts:
        if word in vocab:
            user_encoded_text.append(vocab[word])  # 把对应的词编码，计入列表里。
        else:
            user_encoded_text.append(vocab['UNK'])  # 如果没见过，就计入未知字符

    # 补零
    user_padded_text = []
    fixed_length = 40
    if len(user_encoded_text) <= fixed_length:
        user_padded_text = user_encoded_text + [0] * (fixed_length - len(user_encoded_text))
    elif len(user_encoded_text) > fixed_length:
        user_padded_text = user_encoded_text[:fixed_length]

    # 变成张量
    user_batch = torch.tensor([user_padded_text], dtype=torch.long)
    return user_batch

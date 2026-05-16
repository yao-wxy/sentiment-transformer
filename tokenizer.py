import pickle
import jieba

# 建立词表
def build_vocab(cleaned_data):
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
def encoded(cleaned_data,vocab):
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
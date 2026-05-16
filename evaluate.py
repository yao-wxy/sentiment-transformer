# 测试脚本
 import config
from utils.load_model import load_model
from main.text_to_tensor import text_to_tensor
from training.train_curriculum import vocab
from main.predict import predict

# 课程学习的模型测试。
curriculum_model = load_model(file_name='../new_cache/best_model.pt',vocab_size=config.curriculum_vocab_size)
text = '很高兴见到你。'
correct_emotion = '正面'
user_batch = text_to_tensor(text, vocab)    # 把文本转化为张量
# 送进模型中推理得出分类结果。
pos_prob,emotion_type = predict(user_batch,curriculum_model)
if emotion_type == correct_emotion:
    print('判断正确！')
else:
    print('判断错误！')


# 混合训练的模型测试。
mixed_model = load_model(file_name='../cache/best_model.pt',vocab_size=config.mixed_vocab_size)

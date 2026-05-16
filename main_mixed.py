import config
from training.train_mixed import vocab
from utils.load_model import load_model
from text_to_tensor import text_to_tensor
from interaction import get_correct_count
from error_text import error_text
from predict import predict
# --------------------------------调试开关控制----------------------------------------------------------------------------
# 是否调试用户交互模块，生成错题本和问答正误判断。
VERBOSE_MODE = False


def main():
    # --------------------------------------用户交互模块------------------------------------------------------------------
    correct_count = 0  # 记录正确回答次数
    count = 0  # 记录总次数
    model = load_model(file_name='../new_cache/best_model.pt', vocab_size=config.curriculum_vocab_size)
    print('你好！我是你的AI情感小助手，可以判断正/负情绪。\n'
          '我主要适用于日常语境下的情绪判断。对于情绪极端或表达复杂的文本，判断结果可能不够准确。\n\n')

    while True:
        user_input = input("请输入一句话：")

        if user_input == '结束':
            break

        else:
            user_batch = text_to_tensor(user_input, vocab)

            # 处理用户的负面反馈：
            if '判错了' in user_input or '判断的不对' in user_input:
                print('抱歉，我只是一个初级模型，难以识别复杂情绪，我还会继续学习\n')

            user_text = user_input.strip()

            # 进入正常判断流程：
            pos_prob, emotion_type = predict(user_batch, model)
            print(f'您的情绪为{emotion_type}，判断为正面的概率为{pos_prob}。')
            count += 1

            # 调试，收集用户的反馈。
            if VERBOSE_MODE:
                correct_count = get_correct_count(user_text, emotion_type="正面", pos_prob=pos_prob,
                                                  correct_count=correct_count)

    # 输出错题本。
    if VERBOSE_MODE:
        new_error_text = error_text(count, correct_count)
        print(f'错题本为:\n{new_error_text}')

    if __name__ == '__main__':
        main()
error_text = []  # 记录判错的文本，用于复核数据。
# 写交互逻辑
def get_correct_count(user_text, emotion_type, pos_prob, correct_count):
    user_correction = input('请问这句话您觉得我判对了吗？输入1,y,对/2,n,错/不确定,3   ').strip().lower()
    while True:
        if user_correction == '1' or user_correction == 'y' or user_correction == '对':
            correct_count += 1
            print('好的，已记录。')
            break
        elif user_correction == '2' or user_correction == 'n' or user_correction == '错':
            print('抱歉，我只是一个初级模型，难以识别复杂情绪，我还会继续学习\n')
            error_text.append({'输入文本': user_text,
                               '判断情绪': emotion_type,
                               '判断概率': pos_prob
                               })
            break
        elif user_correction == '3' or user_correction == '不确定':
            print('好的，收到反馈。收录为待审核文本，放入错题本中...\n')
            error_text.append({'输入文本': user_text,
                               '判断情绪': emotion_type,
                               '判断概率': pos_prob
                               })
            break
        elif user_correction in ['跳过', '不知道', '不想答', '不想说', '不确定', '算了', ' ', '4', '']:
            print('好的，已跳过本次反馈。')
            break
        else:
            print('抱歉，未识别到有效字符，请重新输入。1,y,对/2,n,错/不确定,3')

    return correct_count
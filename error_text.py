def error_text(count,correct_count):
    error_text = []  # 记录判错的文本，用于复核数据。
    wrong_count = count - correct_count
    print(f'正确轮数为{correct_count},错误轮数为{wrong_count},'
          f'错误率为{wrong_count / count},正确率为{correct_count / count}\n')
    if wrong_count == 0:
        print('全对。')
    elif correct_count == 0:
        print('全错。')
    # 去重，输出最终结果。
    new_error_text = []
    for item in error_text:
        if item['输入文本'] not in [i['输入文本'] for i in new_error_text]:
            new_error_text.append(item)

    return new_error_text
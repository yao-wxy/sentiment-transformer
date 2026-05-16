import torch
def predict(user_batch,model):
    # 传入模型当中
    with torch.no_grad():
        logits = model(user_batch)
        probs = torch.softmax(logits, dim=-1)  # 将整数换算成概率值
        pos_prob = probs[0][1].item()  # 正面概率

    if pos_prob > 0.6:
        emotion_type = '正面'
    elif pos_prob < 0.4:
        emotion_type = '负面'
    else:
        emotion_type = '不确定'
    return pos_prob,emotion_type
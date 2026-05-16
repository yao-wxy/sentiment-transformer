import torch
from config import MAX_LENGTH
from models.emotional_classifier import Emotion_Classifier

def load_model(file_name,vocab_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Emotion_Classifier(vocab_size=vocab_size, num_heads=8, num_classes=2, d_model=128,max_len=MAX_LENGTH).to(device)

    # 加载训练好的模型
    checkpoint = torch.load(file_name, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)
    model.eval()

    return model


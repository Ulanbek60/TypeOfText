import torch
import torch.nn as nn
from torchtext.vocab import Vocab
import streamlit as st
import re

classes = ['technology', 'weather', 'love']

class TypeOfText(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # объединяем слева и справа
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загружаем vocab и модель
vocab = torch.load('vocab.pth')
model = TypeOfText(len(vocab)).to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Токенизатор для русского текста
def tokenizer(text):
    return re.findall(r'\w+', text.lower())

# Конвертация текста в индексы
def text_pipeline(text: str):
    return [vocab[i] for i in tokenizer(text)]



# Streamlit
st.title('Text Prediction')
st.text('Напишите текст в жанре (технологии, любви, погоде), модель найдет, что это за жанр')
text = st.text_area("Напишите что-нибудь 👇")

if not text:
    st.info('Напишите что-нибудь')
else:
    if st.button('Определить жанр текста'):
        try:
            tensor = torch.tensor(text_pipeline(text), dtype=torch.int64).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(tensor)
                label = torch.argmax(pred, dim=1).item()

            st.success({'Label': f'{classes[label]}'})
        except Exception as e:
            st.exception(f'Error: {str(e)}')

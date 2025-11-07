import torch, torch.nn as nn
from transformers import BertModel

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 512)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return self.fc(outputs.pooler_output)

class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 16, (3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, (3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(32, 512)

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x.view(x.size(0), -1))

class TextToVideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()

    def forward(self, text_inputs, video_inputs):
        text_feat = self.text_encoder(**text_inputs)
        video_feat = self.video_encoder(video_inputs)
        return text_feat, video_feat

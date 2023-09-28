import torch.nn as nn
from modules.fga.atten import Atten
import torch
import math


class TempEmbedder(nn.Module):
    def __init__(self, input_size=9216, hidden_size=2304, output_size=1024, class_token=True, seq_target_length=24):
        super(TempEmbedder, self).__init__()
        self.output_size = output_size
        self.class_token = class_token
        self.seq_target_length = seq_target_length

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.fc4 = nn.Linear(output_size, output_size)

        self.gelu = nn.GELU()
        if class_token:
            self.fga = Atten(util_e=[output_size], pairwise_flag=False)

    def forward(self, audio_embs):

        bs, seq_len = audio_embs.shape[0], audio_embs.shape[1]
        audio_embs = self.fc1(audio_embs)
        audio_embs = self.gelu(audio_embs)
        audio_embs = self.fc2(audio_embs)
        audio_embs = self.gelu(audio_embs)
        audio_embs = self.fc3(audio_embs)
        audio_embs = audio_embs.view(bs, self.seq_target_length, -1, self.output_size).mean(dim=2)
        audio_embs = self.gelu(audio_embs)
        audio_embs = self.fc4(audio_embs)

        attend = None
        if self.class_token:
            attend = self.fga([audio_embs])[0]

        local_window_1 = torch.cat([torch.unsqueeze(audio_embs[:, max(i-1, 0):min(i+1, audio_embs.shape[1]), :].mean(
            dim=1), dim=1) for i in range(audio_embs.shape[1])], dim=1)
        local_window_2 = torch.cat([torch.unsqueeze(audio_embs[:, max(i-2, 0):min(i+2, audio_embs.shape[1]), :].mean(
            dim=1), dim=1) for i in range(audio_embs.shape[1])], dim=1)
        local_window_3 = torch.cat([torch.unsqueeze(audio_embs[:, max(i-4, 0):min(i+4, audio_embs.shape[1]), :].mean(
            dim=1), dim=1) for i in range(audio_embs.shape[1])], dim=1)
        local_window_4 = torch.cat([torch.unsqueeze(audio_embs[:, max(i-8, 0):min(i+8, audio_embs.shape[1]), :].mean(
            dim=1), dim=1) for i in range(audio_embs.shape[1])], dim=1)

        return audio_embs, local_window_1, local_window_2, local_window_3, local_window_4, attend

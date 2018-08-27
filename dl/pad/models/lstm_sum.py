"""
1 简介 ：LSTMsum: input --> embedding --> indrnn --> sum --> fc1 --> fc2 --> output
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMsum(nn.Module):
    def __init__(self, emb_weights, emb_freeze, input_size, hidden_size, num_layers, l1_size, l2_size, num_classes, bidir, lstm_dropout):
        super(LSTMsum, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=emb_weights, freeze=emb_freeze)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidir, dropout=lstm_dropout)
        if bidir:
            self.l1 = nn.Linear(hidden_size*2, l1_size)
        else:
            self.l1 = nn.Linear(hidden_size, l1_size)

        self.l2 = nn.Linear(l1_size, l2_size)
        self.l3 = nn.Linear(l2_size, num_classes)

    def forward(self, input):
        emb_out = self.embedding(input)
        lstm_out, _ = self.lstm(emb_out)
        sum_out = torch.sum(lstm_out, dim=1)
        l1_out = F.relu(self.l1(sum_out))
        l2_out = F.relu(self.l2(l1_out))
        l3_out = self.l3(l2_out)
        return l3_out
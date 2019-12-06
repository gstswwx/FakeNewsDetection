# coding: utf-8

from __future__ import print_function
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.textEncoder = TextEncoder(coding_dim=32)
        self.fc = nn.Linear(self.textEncoder.coding_dim, 2)

    def forward(self, tokens):
        x = self.textEncoder(tokens)
        x = F.relu(self.fc(x))
        # x = F.softmax(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, coding_dim):
        super(TextEncoder, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-chinese')
        self.coding_dim = coding_dim
        model_embedding_dim = self.model.config.hidden_size
        self.fc = nn.Linear(model_embedding_dim, coding_dim)

    def forward(self, tokens):
        with torch.no_grad():
            output = self.model(tokens)
        model_embeddings = output[0][:, 0, :]
        codings = F.relu(self.fc(model_embeddings))
        return codings


tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

df = pd.read_csv('../data/train.csv')
train_ids = list(df['id'])
train_texts = list(df['text'])
for i in range(len(train_texts)):
    train_texts[i] = train_texts[i].decode('utf-8')
train_labels = list(df['label'])

tokens = []
# attention_masks = []

count = 0
for text in train_texts:
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens.append(indexed_tokens)
    # attention_masks.append([1]*len(indexed_tokens))
    count += 1
    print(count)
    if count > 39:
        break

max_text_len = max([len(each) for each in tokens])

for i in range(len(tokens)):
    padding = [0] * (max_text_len - len(tokens[i]))
    tokens[i].extend(padding)
    # attention_masks[i].extend(padding)

x_data = torch.LongTensor(tokens)
# print(x_data.size())
y_data = torch.LongTensor(train_labels[0:x_data.size()[0]])
# print(x_data)
# print(y_data)

full_dataset = TensorDataset(x_data, y_data)

train_size = int(0.8 * len(full_dataset))
valid_size = len(full_dataset) - train_size

train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=2)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=valid_size, shuffle=False, num_workers=2)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

for epoch in range(20):
    total_loss = 0.0
    count = 0
    for i, data in enumerate(train_loader):
        # print('gg')
        train_inputs, train_labels = data
        # print(train_inputs[0][0])
        optimizer.zero_grad()

        train_outputs = net(train_inputs)
        # print('xx')

        train_loss = criterion(train_outputs, train_labels)
        total_loss += train_loss
        count += 1
        # print(train_outputs, train_labels, train_loss)


        train_loss.backward()
        optimizer.step()
    avg_loss = total_loss / count
    valid_loss = 0.0
    # 应该只跑一次
    for i, data in enumerate(valid_loader):
        # print('hh')
        valid_inputs, valid_labels = data
        valid_outputs = F.softmax(net(valid_inputs), dim=1)

        print(valid_outputs)
        print(valid_labels)

        valid_loss = criterion(valid_outputs, valid_labels)

        correct_count = 0
        for j in range(0, valid_size):
            predict = 0
            if valid_outputs[j][0] < valid_outputs[j][1]:
                predict = 1
            if predict == valid_labels[j]:
                correct_count += 1
        valid_acc = float(correct_count) / valid_size

        break

    print('epoch=%d' % epoch, ':', 'train_loss=%f' % avg_loss, 'valid_loss=%f' % valid_loss, 'valid_acc=%f' % valid_acc)










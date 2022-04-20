import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

# batch_size / print_every
batch_size = 8

np.random.seed(2020)
torch.manual_seed(2020)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed(2020)
    print('using cuda device')

data = pd.read_csv('./dianping.csv', encoding='utf-8')
model_path = "pretrained_models/bert-base-cased"

torch.cuda.empty_cache()
# 首先处理数据
# 剔除标点符号,\xa0 空格
def pretreatment(comments):
    result_comments = []
    punctuation = '。，？！：%&~（）、；“”&|,.?!:%&~();""'
    for comment in comments:
        comment = ''.join([c for c in comment if c not in punctuation])
        comment = ''.join(comment.split())  # \xa0
        result_comments.append(comment)

    return result_comments


result_comments = pretreatment(list(data['comment'].values))
tokenizer = BertTokenizer.from_pretrained(model_path)
# 此处增加对氨基酸序列的分词结构
tokenizer.add_special_tokens({'additional_special_tokens': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                                                                         'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                                                                         'T', 'U', 'V', 'W', 'X', 'Y', 'Z']})
result_comments_id = tokenizer(result_comments, padding=True, truncation=True, max_length=200, return_tensors='pt')
X = result_comments_id['input_ids']
y = torch.from_numpy(data['sentiment'].values).float()

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, shuffle=True, stratify=y, random_state=2020)

X_valid, X_test, y_valid, y_test=train_test_split(X_test, y_test, test_size=0.5, shuffle=True, stratify=y_test, random_state=2020)

# create Tensor datasets
train_data = TensorDataset(X_train, y_train)
valid_data = TensorDataset(X_valid, y_valid)
test_data = TensorDataset(X_test, y_test)

# make sure the SHUFFLE your training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)


class bert_lstm(nn.Module):
    def __init__(self, hidden_dim, output_size, n_layers, bidirectional=True, drop_prob=0.4):
        super(bert_lstm, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        # Bert ----------------重点，bert模型需要嵌入到自定义模型里面
        self.bert = BertModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        # LSTM layers
        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)

        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layers
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)

        # self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)
        # 生成bert字向量
        x = self.bert(x)[0]  # bert 字向量

        # lstm_out
        # x = x.float()
        lstm_out, (hidden_last, cn_last) = self.lstm(x, hidden)
        # print(lstm_out.shape)   #[32,100,768]
        # print(hidden_last.shape)   #[4, 32, 384]
        # print(cn_last.shape)    #[4, 32, 384]

        # 双向LSTM需要单独处理
        if self.bidirectional:
            # 正向最后一层，最后一个时刻
            hidden_last_L = hidden_last[-2]
            # print(hidden_last_L.shape)  #[32, 384]
            # 反向最后一层，最后一个时刻
            hidden_last_R = hidden_last[-1]
            # print(hidden_last_R.shape)   #[32, 384]
            # 进行拼接
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
            # print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
        else:
            hidden_last_out = hidden_last[-1]  # [32, 384]

        # dropout and fully-connected layer
        out = self.dropout(hidden_last_out)
        # print(out.shape)    #[32,768]
        out = self.fc(out)

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        number = 1
        if self.bidirectional:
            number = 2
        if use_cuda:
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float().cuda()
                      )
        else:
            hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float(),
                      weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float()
                      )
        return hidden


output_size = 2  # 此处为分类 类别数
hidden_dim = 384
n_layers = 2
bidirectional = True
# initialize the model
net = bert_lstm(hidden_dim, output_size, n_layers, bidirectional)

# train
# loss and optimization functions
lr = 2e-5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
epochs = 10
print_every = 8
clip = 5  # gradient clipping

# move model to GPU, if available
if use_cuda:
    net.cuda()
net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)
    counter = 0

    # batch loop
    for inputs, labels in train_loader:
        counter += 1
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        h = tuple([each.data for each in h])
        net.zero_grad()
        output = net(inputs, h)
        loss = criterion(output.squeeze(), labels.long())
        loss.backward()
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            net.eval()
            with torch.no_grad():
                val_h = net.init_hidden(batch_size)
                val_losses = []
                for inputs, labels in valid_loader:
                    val_h = tuple([each.data for each in val_h])

                    if use_cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.long())
                    val_losses.append(val_loss.item())

            net.train()
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))


# test
test_losses = []  # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    if use_cuda:
        inputs, labels = inputs.cuda(), labels.cuda()
    output = net(inputs, h)
    test_loss = criterion(output.squeeze(), labels.long())
    test_losses.append(test_loss.item())
    output = torch.nn.Softmax(dim=1)(output)
    pred = torch.max(output, 1)[1]

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not use_cuda else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))

torch.save(net.state_dict(), 'sentiment.pth')


# 若想要使用保存的模型测试 利用如下代码再次复现一个相同的 net 即可
output_size = 2  # 此处为分类 类别数
hidden_dim = 384   # 768/2
n_layers = 2
bidirectional = True

net = bert_lstm(hidden_dim, output_size, n_layers, bidirectional)
net.load_state_dict(torch.load('sentiment.pth'))


def predict(net, test_comments):
    result_comments = pretreatment(test_comments)  # 预处理去掉标点符号
    # 转换为字id
    tokenizer = BertTokenizer.from_pretrained(model_path)
    result_comments_id = tokenizer(result_comments, padding=True, truncation=True, max_length=120, return_tensors='pt')
    tokenizer_id = result_comments_id['input_ids']
    inputs = tokenizer_id
    batch_size = inputs.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    if use_cuda:
        inputs = inputs.cuda()

    net.eval()
    with torch.no_grad():
        # get the output from the model
        output = net(inputs, h)
        output = torch.nn.Softmax(dim=1)(output)
        pred = torch.max(output, 1)[1]
        # printing output value, before rounding
        # print('预测概率为: {:.6f}'.format(output.item()))
        if pred.item() == 1:
            print("预测结果为:正向")
        else:
            print("预测结果为:负向")


if use_cuda:
    net.cuda()
comment1 = ['菜品一般，不好吃']
predict(net, comment1)

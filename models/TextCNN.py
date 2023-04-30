# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        
        self.sentence_length = 25                                       #New

'''Convolutional Neural Networks for Sentence Classification'''


# class Model(nn.Module):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         if config.embedding_pretrained is not None:
#             self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
#         else:
#             self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
#         self.dropout = nn.Dropout(config.dropout)
#         self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)).squeeze(3)
#         x = F.max_pool1d(x, x.size(2)).squeeze(2)
#         return x

#     def forward(self, x):
#         out = self.embedding(x[0])
#         out = out.unsqueeze(1)
#         out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
#         out = self.dropout(out)
#         out = self.fc(out)
#         return out

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs_1 = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.middle_relu = nn.ModuleList(
            [nn.ReLU() for _ in config.filter_sizes])
        self.avgpool_1 = nn.ModuleList(
            [nn.AvgPool2d((config.sentence_length - k + 1, 1)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def forward(self, x):
        print("-4",np.shape(x))
        out = self.embedding(x[0])
        print("-3",np.shape(x))
        out = out.unsqueeze(1)
        print("-2.5",np.shape(x))
        conv_out = [conv(out) for conv in self.convs_1]
        print("-2",np.shape(conv_out))
        conv_out = [self.middle_relu[i](conv_out[i]) for i in range(len(self.middle_relu))]
        print("-1",np.shape(conv_out))
        pooled_out = [pool(conv_out[i]).squeeze(3) for i, pool in enumerate(self.avgpool_1)]
        print("0",np.shape(pooled_out))
        pooled_out = [pool(pooled_out[i]).squeeze(2) for i, pool in enumerate(self.avgpool_1)]
        print("1",np.shape(pooled_out))
        out = torch.cat(pooled_out, dim=1)
        print("2",np.shape(out))
        out = self.dropout(out)
        print("3",np.shape(out))
        out = self.fc(out)
        return out
    
class ANN_TextCNN(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.convs_1 = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=args.filter_num, kernel_size=(filter_size, args.hidden_dim), bias=False)
            for filter_size in args.filters
        ])
        self.middle_relu = nn.ModuleList([
            nn.ReLU()
            for _ in args.filters
        ])
        self.avgpool_1 = nn.ModuleList([
            nn.AvgPool2d((args.sentence_length - filter_size + 1, 1)) for filter_size in args.filters
        ])
        # self.maxpool_1 = nn.ModuleList([
        #     nn.MaxPool2d((args.sentence_length - filter_size + 1, 1)) for filter_size in args.filters
        # ])
        self.relu_2 = nn.ReLU()
        self.drop = nn.Dropout(p=args.dropout_p)
        self.fc_1 = nn.Linear(len(args.filters)*args.filter_num, args.label_num, bias=False)
        
    def forward(self, x):
        x = x.float()
        batch_size = x.shape[0]
        x = x.unsqueeze(dim=1)
        conv_out = [conv(x) for conv in self.convs_1]
        conv_out = [self.middle_relu[i](conv_out[i]) for i in range(len(self.middle_relu))]
        # conv_out = [self.relu_1(i) for i in conv_out]
        # pooled_out = [self.maxpool_1[i](conv_out[i]) for i in range(len(self.maxpool_1))]
        pooled_out = [self.avgpool_1[i](conv_out[i]) for i in range(len(self.avgpool_1))]
        pooled_out = [self.relu_2(pool) for pool in pooled_out]
        flatten = torch.cat(pooled_out, dim=1).view(batch_size, -1)
        flatten = self.drop(flatten)
        fc_output = self.fc_1(flatten)
        return fc_output
    
    def cal_flop(self, x):
        flops_of_all_layers = []
        x = x.float()
        batch_size = x.shape[0]
        x = x.unsqueeze(dim=1)
        conv_out = [conv(x) for conv in self.convs_1]
        flops_of_all_layers.append(np.sum(
            [FlopCountAnalysis(conv, x).total() for conv in self.convs_1]
        ))
        conv_out = [self.middle_relu[i](conv_out[i]) for i in range(len(self.middle_relu))]
        # conv_out = [self.relu_1(i) for i in conv_out]
        # pooled_out = [self.maxpool_1[i](conv_out[i]) for i in range(len(self.maxpool_1))]
        pooled_out = [self.avgpool_1[i](conv_out[i]) for i in range(len(self.avgpool_1))]
        pooled_out = [self.relu_2(pool) for pool in pooled_out]
        flatten = torch.cat(pooled_out, dim=1).view(batch_size, -1)
        flatten = self.drop(flatten)
        flops_of_all_layers.append(FlopCountAnalysis(self.fc_1, flatten).total())
        fc_output = self.fc_1(flatten)
        return flops_of_all_layers

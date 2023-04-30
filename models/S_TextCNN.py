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
#         self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
#         self.num_filters = 256                                          # 卷积核数量(channels数)
        self.filter_sizes = (3, 4, 5)                                   # 卷积核尺寸
        self.num_filters = 100                                          # 卷积核数量(channels数)
        
        self.surrogate = 'fast_sigmoid'
        self.beta = 1.0
        #self.filters = [3,4,5]
        #self.filter_num = 100
        self.positive_init_rate = 0.55
        self.threshold = 1.0

#         # monitor
#         self.dead_neuron_checker = "False"
        

INITIAL_MEAN_DICT = {
    "conv-kaiming": {
        0.5: 0.0,
        0.55: 0.01,
        0.6: 0.02,
        0.7: 0.03,
        0.8: 0.06
        },
    "linear-kaiming": {
        0.5: 0.0,
        0.55: 0.08,
        0.6: 0.14,
        0.7: 0.28,
        0.8: 0.46
        },
}
'''Convolutional Neural Networks for Sentence Classification'''
   
# class Model(nn.Module):
#     def __init__(self, config):
#         super(Model, self).__init__()
#         if config.embedding_pretrained is not None:
#             self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
#         else:
#             self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size=(k, config.embed)) for k in config.filter_sizes])
#         self.relu = nn.ReLU()
#         self.avgpool = nn.ModuleList(
#             [nn.AvgPool2d(kernel_size=(config.pad_size - k + 1, 1)) for k in config.filter_sizes])
#         self.dropout = nn.Dropout(config.dropout)
#         self.fc = nn.Linear(len(config.filter_sizes) * config.num_filters, config.num_classes,bias=False)
#         self.batch_size = config.batch_size

#     def forward(self, x):
#         out = self.embedding(x[0])
#         batch_size = out.shape[0]
#         out = out.unsqueeze(1)
#         conv_out = [conv(out) for conv in self.convs]
#         conv_out = [self.relu(i) for i in conv_out]
#         pooled_out = [pool(i).squeeze(3) for i, pool in zip(conv_out, self.avgpool)]
#         pooled_out = [self.relu(i) for i in pooled_out]
#         out = torch.cat(pooled_out, dim=1).view(batch_size, -1)
#         out = self.dropout(out)
#         out = self.fc(out)
#         return out
   
import snntorch.surrogate as surrogate
import snntorch as snn
#from utils.config import INITIAL_MEAN_DICT
#from utils.monitor import Monitor
from fvcore.nn import FlopCountAnalysis

class Model(nn.Module):
    def __init__(self, config, spike_grad=surrogate.fast_sigmoid(slope=25)) -> None:
        super().__init__()
#         self.dead_neuron_checker = config.dead_neuron_checker
        self.positive_init_rate = config.positive_init_rate
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs_1 = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size=(filter_size, config.embed))
            for filter_size in config.filter_sizes
        ])
        self.middle_lifs = nn.ModuleList([
            snn.Leaky(beta=config.beta, spike_grad=spike_grad, init_hidden=True, threshold=config.threshold)
            for _ in config.filter_sizes
        ])
        self.avgpool_1 = nn.ModuleList([
            nn.AvgPool2d((config.pad_size - filter_size + 1, 1)) for filter_size in config.filter_sizes
        ])
        self.lif1 = snn.Leaky(beta=config.beta, spike_grad=spike_grad, init_hidden=True, threshold=config.threshold)
        self.fc_1 = nn.Linear(len(config.filter_sizes)*config.filter_num, config.num_classes, bias=False)
        self.lif2 = snn.Leaky(beta=config.beta, spike_grad=spike_grad, init_hidden=True, threshold=config.threshold, output=True)

        for c in self.convs_1:
            c.weight.data.add_(INITIAL_MEAN_DICT['conv-kaiming'][self.positive_init_rate])
        m = self.fc_1
        m.weight.data.add_(INITIAL_MEAN_DICT["linear-kaiming"][self.positive_init_rate])

    def forward(self, x):
        out = self.embedding(x[0])
        batch_size = out.shape[0]
        out = out.unsqueeze(dim=1)
        conv_out = [conv(out) for conv in self.convs_1]

        conv_out = [self.middle_lifs[i](conv_out[i]) for i in range(len(self.middle_lifs))]

        pooled_out = [self.avgpool_1[i](conv_out[i]) for i in range(len(self.avgpool_1))]
   
        spks = [self.lif1(pooled) for pooled in pooled_out]
        spks_1 = torch.cat(spks, dim=1).view(batch_size, -1)
        hidden_1 = self.fc_1(spks_1)
        # cur2 = self.fc_2(hidden_1)
        spk2, mem2 = self.lif2(hidden_1)
#         if self.dead_neuron_checker == "True":
#             temp_spks = spks_1.sum(dim=0)
#             Monitor.add_monitor(temp_spks, 0)
        return spks_1, spk2, mem2
   
# class ANN_TextCNN(nn.Module):
#     def __init__(self, args) -> None:
#         super().__init__()
#         self.convs_1 = nn.ModuleList([
#             nn.Conv2d(in_channels=1, out_channels=args.filter_num, kernel_size=(filter_size, args.hidden_dim), bias=False)
#             for filter_size in args.filters
#         ])
#         self.middle_relu = nn.ModuleList([
#             nn.ReLU()
#             for _ in args.filters
#         ])
#         self.avgpool_1 = nn.ModuleList([
#             nn.AvgPool2d((args.sentence_length - filter_size + 1, 1)) for filter_size in args.filters
#         ])
#         # self.maxpool_1 = nn.ModuleList([
#         #     nn.MaxPool2d((args.sentence_length - filter_size + 1, 1)) for filter_size in args.filters
#         # ])
#         self.relu_2 = nn.ReLU()
#         self.drop = nn.Dropout(p=args.dropout_p)
#         self.fc_1 = nn.Linear(len(args.filters)*args.filter_num, args.label_num, bias=False)
        
#     def forward(self, x):
#         x = x.float()
#         batch_size = x.shape[0]
#         x = x.unsqueeze(dim=1)
#         conv_out = [conv(x) for conv in self.convs_1]
#         conv_out = [self.middle_relu[i](conv_out[i]) for i in range(len(self.middle_relu))]
#         # conv_out = [self.relu_1(i) for i in conv_out]
#         # pooled_out = [self.maxpool_1[i](conv_out[i]) for i in range(len(self.maxpool_1))]
#         pooled_out = [self.avgpool_1[i](conv_out[i]) for i in range(len(self.avgpool_1))]
#         pooled_out = [self.relu_2(pool) for pool in pooled_out]
#         flatten = torch.cat(pooled_out, dim=1).view(batch_size, -1)
#         flatten = self.drop(flatten)
#         fc_output = self.fc_1(flatten)
#         return fc_output
    
#     def cal_flop(self, x):
#         flops_of_all_layers = []
#         x = x.float()
#         batch_size = x.shape[0]
#         x = x.unsqueeze(dim=1)
#         conv_out = [conv(x) for conv in self.convs_1]
#         flops_of_all_layers.append(np.sum(
#             [FlopCountAnalysis(conv, x).total() for conv in self.convs_1]
#         ))
#         conv_out = [self.middle_relu[i](conv_out[i]) for i in range(len(self.middle_relu))]
#         # conv_out = [self.relu_1(i) for i in conv_out]
#         # pooled_out = [self.maxpool_1[i](conv_out[i]) for i in range(len(self.maxpool_1))]
#         pooled_out = [self.avgpool_1[i](conv_out[i]) for i in range(len(self.avgpool_1))]
#         pooled_out = [self.relu_2(pool) for pool in pooled_out]
#         flatten = torch.cat(pooled_out, dim=1).view(batch_size, -1)
#         flatten = self.drop(flatten)
#         flops_of_all_layers.append(FlopCountAnalysis(self.fc_1, flatten).total())
#         fc_output = self.fc_1(flatten)
#         return flops_of_all_layers
    



# class SNN_TextCNN(nn.Module):
#     def __init__(self, args, spike_grad=surrogate.fast_sigmoid(slope=25)) -> None:
#         super().__init__()
#         self.dead_neuron_checker = args.dead_neuron_checker
#         #self.initial_method = args.initial_method
#         self.positive_init_rate = args.positive_init_rate
#         self.convs_1 = nn.ModuleList([
#             nn.Conv2d(in_channels=1, out_channels=args.filter_num, kernel_size=(filter_size, args.hidden_dim))
#             for filter_size in args.filters
#         ])
#         self.middle_lifs = nn.ModuleList([
#             snn.Leaky(beta=args.beta, spike_grad = spike_grad, init_hidden=True, threshold=args.threshold)
#             for _ in args.filters
#         ])
#         self.avgpool_1 = nn.ModuleList([
#             nn.AvgPool2d((args.sentence_length - filter_size + 1, 1)) for filter_size in args.filters
#         ])
#         self.lif1 = snn.Leaky(beta=args.beta, spike_grad=spike_grad, init_hidden=True, threshold=args.threshold)
#         self.fc_1 = nn.Linear(len(args.filters)*args.filter_num, args.label_num)
#         self.lif2 = snn.Leaky(beta=args.beta, spike_grad=spike_grad, init_hidden=True, threshold=args.threshold, output=True)
    
#     def initial(self):
#         for c in self.convs_1:
#             c.weight.data.add_(INITIAL_MEAN_DICT['conv-kaiming'][self.positive_init_rate])
#         m = self.fc_1
#         m.weight.data.add_(INITIAL_MEAN_DICT["linear-kaiming"][self.positive_init_rate])

#     def forward(self, x):
#         batch_size = x.shape[0]
#         x = x.unsqueeze(dim=1)
#         conv_out = [conv(x) for conv in self.convs_1]

#         conv_out = [self.middle_lifs[i](conv_out[i]) for i in range(len(self.middle_lifs))]

#         pooled_out = [self.avgpool_1[i](conv_out[i]) for i in range(len(self.avgpool_1))]
   
#         spks = [self.lif1(pooled) for pooled in pooled_out]
#         spks_1 = torch.cat(spks, dim=1).view(batch_size, -1)
#         hidden_1 = self.fc_1(spks_1)
#         # cur2 = self.fc_2(hidden_1)
#         spk2, mem2 = self.lif2(hidden_1)
#         if self.dead_neuron_checker == "True":
#             temp_spks = spks_1.sum(dim=0)
#             Monitor.add_monitor(temp_spks, 0)
#         return spks_1, spk2, mem2
        
#     def cal_flop_and_fire_rate(self, x):
#         flops_of_all_layers = []
#         fire_rates_of_all_layers = []
#         batch_size = x.shape[0]
#         x = x.unsqueeze(dim=1)
#         fire_rates_of_all_layers.append((torch.sum(x)/torch.prod(torch.tensor(x.shape))).cpu().detach().numpy())
#         conv_out = [conv(x) for conv in self.convs_1]
#         flops_of_all_layers.append(np.sum(
#             [FlopCountAnalysis(conv, x).total() for conv in self.convs_1]
#         ))
#         conv_out = [self.middle_lifs[i](conv_out[i]) for i in range(len(self.middle_lifs))]

#         pooled_out = [self.avgpool_1[i](conv_out[i]) for i in range(len(self.avgpool_1))]
#         # flops_of_all_layers.append(
#         #     np.sum([FlopCountAnalysis(self.avgpool_1[i],conv_out[i]).total() for i in range(len(self.avgpool_1))])
#         # )
#         spks = [self.lif1(pooled) for pooled in pooled_out]
#         spks_1 = torch.cat(spks, dim=1).view(batch_size, -1)
#         fire_rates_of_all_layers.append((torch.sum(spks_1)/torch.prod(torch.tensor(spks_1.shape))).cpu().detach().numpy())
#         hidden_1 = self.fc_1(spks_1)
#         flops_of_all_layers.append(FlopCountAnalysis(self.fc_1, spks_1).total())
#         spk2, mem2 = self.lif2(hidden_1)
#         return flops_of_all_layers, fire_rates_of_all_layers

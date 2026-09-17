import torch
from torch import nn
from torch.nn import functional as F

class AlternateLayer(nn.Module):
    def __init__(self, input_shape):
        self.seq_len, self.input_size = input_shape
        assert self.input_size % 30 == 0
        super(AlternateLayer, self).__init__()
        self.timdistLSTM = nn.LSTM(self.seq_len, 64, 1, batch_first=True)
        self.attFCN = nn.Linear(64, 64)
        self.seqLSTM = nn.LSTM(64, 64, batch_first=True)
        self.findense = nn.Linear(64, 2)
        self.fintanh = nn.Tanh()
        # Applying the Xavier initialization
        torch.nn.init.xavier_uniform_(self.attFCN.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.findense.weight, gain=1.0)


    def forward(self, x):
        x = x.flip(-1)
        batch_size, seq_len, input_dim = x.size()
        x = x.transpose(1, 2)
        x = x.reshape(batch_size*30, int(self.input_size/30), seq_len)
        _, (x, _) = self.timdistLSTM(x)
        x = x.reshape(batch_size, 30, 64)
        att = self.attFCN(x)
        att = F.softmax(att, dim =-1)
        x = x * att
        x, _ = self.seqLSTM(x)
        x = F.dropout(x, 0.2, training = self.training)
        x = self.findense(x)
        x = self.fintanh(x)
        x = x.reshape(batch_size, 60)
        return x


class WaveLayer(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bn = False):
        super(WaveLayer, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=self.padding, dilation=dilation)
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.filter = nn.Conv1d(in_channels, in_channels, 1)
        self.gate = nn.Conv1d(in_channels, in_channels, 1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, 1)
        self.bn = bn

        if (self.bn == True):
            self.bn1 = nn.BatchNorm1d(in_channels)
            self.bnf = nn.BatchNorm1d(in_channels)
            self.bng = nn.BatchNorm1d(in_channels)

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.filter.weight, gain=1.0)
        torch.nn.init.xavier_uniform_(self.gate.weight, gain=1.0)
       # self.skip = nn.Conv1d(out_channels, in_channels, 1)
       # self.residual = nn.Conv1d(out_channels, in_channels, 1)
        
    def forward(self, x):
        # x_padded = torch.nn.functional.pad(x, (self.padding, 0))
        output = F.relu(self.conv(x))
        if (self.bn):
            output = self.bn1(output)
        filter = self.filter(output)
        if (self.bn):
            filter = self.bnf(filter)
        gate = self.gate(output)
        if (self.bn):
            gate = self.bng(gate)
        tanh = self.tanh(filter)
        sig = self.sig(gate)
        z = tanh*sig
        z = z[:,:,:-self.padding]
        z = self.conv2(z)
        x = x + z
        return x

class WaveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rates, bn=False):
        super(WaveBlock, self).__init__()
        self.layers = nn.ModuleList()
        dilations = [2**i for i in range(dilation_rates)]
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1)
        for dilation in dilations:
            self.layers.append(WaveLayer(out_channels, kernel_size, dilation, bn))
        torch.nn.init.xavier_uniform_(self.conv1d.weight, gain=1.0, generator=None)

    def forward(self, x):
        x = self.conv1d(x)
        x = F.relu(x)
        for layer in self.layers:
            x = layer(x)
        return x


class WaveNet(nn.Module):
    def __init__(self, input_shape):
        super(WaveNet, self).__init__()
        channels, features = input_shape
        self.block1 = WaveBlock(channels, 16, 3, 8)
        self.block2 = WaveBlock(16, 32, 3, 5)
        self.block3 = WaveBlock(32, 64, 3, 3)
        self.block4 = WaveBlock(64, 64, 2, 2)
        self.lstmblock = nn.LSTM(int(features/2000), 64, 1, batch_first=True)
        self.dense_layer = nn.Linear(64, 2)
        torch.nn.init.xavier_uniform_(self.dense_layer.weight, gain=1.0, generator=None)


    def forward(self, x):
        x = self.block1(x)
        x = F.avg_pool1d(x, 10)
        x = self.block2(x)
        x = F.avg_pool1d(x, 10)
        x = self.block3(x)
        x = F.avg_pool1d(x, 10)
        x = self.block4(x)
        x = F.avg_pool1d(x, 2)
        _, (_, x) = self.lstmblock(x)
        x = x.squeeze(0)
        x = F.dropout(x, 0.5, training=self.training)
        return x


class WaveNetEnd(nn.Module):
    def __init__(self, input_size):
        super(WaveNetEnd, self).__init__()
        self.dense_layer = nn.Linear(input_size, 2)
        torch.nn.init.xavier_uniform_(self.dense_layer.weight, gain=1.0, generator=None)

    def forward(self, x):
        x = self.dense_layer(x)
        # x = F.softmax(x, dim=1)
        return x


class WaveNetFull(nn.Module):
    def __init__(self, input_shape):
        super(WaveNetFull, self).__init__()
        self.wavenet = WaveNet(input_shape)
        self.alternate = AlternateLayer(input_shape)
        self.wavenetend = WaveNetEnd(124)

    def forward(self, x):
        y = self.wavenet(x)
        z = self.alternate(x)
        x = torch.cat((y, z), -1)
        x = self.wavenetend(x)
        return x
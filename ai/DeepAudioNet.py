#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on  5/2/23 9:58

@author: Edward L. Campbell Hernández
contact: ecampbelldsp@gmail.com
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongDecoderLocal(nn.Module):
    def __init__(self, hidden_size, output_size, attention, batch_size, window_size=4, n_layers=1, drop_prob=0.1,
                 net="GRU", bidirectional=False):
        super(LuongDecoderLocal, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.net = net
        self.bidirectional = bidirectional
        self.half_window_size = int(window_size / 2)
        self.drop_prob = drop_prob
        self.batch_size = batch_size
        self.trans = nn.Linear(output_size, hidden_size)
        # The Attention Mechanism is defined in a separate class
        self.attention = attention
        if self.bidirectional is True:
            self.fac = 2
        else:
            self.fac = 1
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        #self.embedding = nn.DataParallel(self.embedding)

        self.dropout = nn.Dropout(self.drop_prob)
        self.norm = nn.BatchNorm1d(output_size)
        if self.net == "GRU":
            self.rnn = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers, batch_first=True,
                              bidirectional=self.bidirectional)

        elif self.net == "LSTM":
            self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.n_layers, batch_first=True,
                               bidirectional=self.bidirectional)
        #self.rnn = nn.DataParallel(self.rnn)

        self.classifier = nn.Linear(self.fac * self.hidden_size * 2, self.output_size)
        #self.classifier = nn.DataParallel(self.classifier)
        # self.act =  nn.ReLU(self.hidden_size)

    def forward(self, inputs, hidden, encoder_outputs, t):
        # Embed input words
        # inputs = self.norm(inputs)
        inputs = self.dropout(inputs)

        embedded = self.trans(inputs).unsqueeze(1)
        # inputs = self.dropout(inputs)

        if self.net == "LSTM":
            only_hidden = hidden[0]
        else:
            only_hidden = hidden
        # embedded = self.act(embedded)

        del inputs
        # embedded = self.dropout(embedded)

        # Passing previous output word (embedded) and hidden state into LSTM cell
        rnn_out, hidden = self.rnn(embedded, hidden)
        del embedded
        # Calculating Alignment Scores - see Attention class for the forward pass function
        alignment_scores = self.attention(rnn_out, encoder_outputs)
        init = t - self.half_window_size
        if init < 0: init = 0
        end = t + self.half_window_size
        # Softmaxing alignment scores to obtain Attention weights
        attn_weights = F.softmax(alignment_scores[:, init:end],
                                 dim=1)  # [:,init:end] #F.softmax(alignment_scores.view(1,-1), dim=1)
        encoder_outputs = encoder_outputs[:, init:end, :]
        # Multiplying Attention weights with encoder outputs to get context vector
        context_vector = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        del encoder_outputs
        # Concatenating output from LSTM with context vector
        output = torch.cat((rnn_out, context_vector), -1)
        del rnn_out, context_vector
        # Pass concatenated vector through Linear layer acting as a Classifier
        output = self.classifier(output)  # F.log_softmax(self.classifier(output), dim=1)
        # output = (output)

        return output[:, 0, :], hidden, attn_weights


class Attention(nn.Module):
    def __init__(self, hidden_size, batch_size, method="dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # Defining the layers/weights required depending on alignment scoring method
        if method == "general":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)

        elif method == "concat":
            self.fc = nn.Linear(hidden_size, hidden_size, bias=False)
            self.weight = nn.Parameter(torch.FloatTensor(self.batch_size, hidden_size).uniform_(0, 1))

    def forward(self, decoder_hidden, encoder_outputs):
        if self.method == "dot":
            # For the dot scoring method, no weights or linear layers are involved
            return encoder_outputs.bmm(torch.transpose(decoder_hidden, 1, 2)).squeeze(
                -1)  # encoder_outputs.bmm(decoder_hidden.view(1,-1,1)).squeeze(-1)

        elif self.method == "general":
            # For general scoring, decoder hidden state is passed through linear layers to introduce a weight matrix
            out = self.fc(decoder_hidden)
            return encoder_outputs.bmm(torch.transpose(out, 1, 2)).squeeze(-1)

        elif self.method == "concat":
            # For concat scoring, decoder hidden state and encoder outputs are concatenated first
            out = torch.tanh(self.fc(decoder_hidden + encoder_outputs))
            return out.bmm(self.weight.unsqueeze(-1)).squeeze(-1)


def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 3:
        (n_out, n_in, height) = layer.weight.size()
        n = n_in * height
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_lstm(layer):
    """
    Initialises the hidden layers in the LSTM - H0 and C0.

    Input
        layer: torch.Tensor - The LSTM layer
    """
    n_i1, n_i2 = layer.weight_ih_l0.size()
    n_i = n_i1 * n_i2

    std = math.sqrt(2. / n_i)
    scale = std * math.sqrt(3.)
    layer.weight_ih_l0.data.uniform_(-scale, scale)

    if layer.bias_ih_l0 is not None:
        layer.bias_ih_l0.data.fill_(0.)

    n_h1, n_h2 = layer.weight_hh_l0.size()
    n_h = n_h1 * n_h2

    std = math.sqrt(2. / n_h)
    scale = std * math.sqrt(3.)
    layer.weight_hh_l0.data.uniform_(-scale, scale)

    if layer.bias_hh_l0 is not None:
        layer.bias_hh_l0.data.fill_(0.)


def init_att_layer(layer):
    """
    Initilise the weights and bias of the attention layer to 1 and 0
    respectively. This is because the first iteration through the attention
    mechanism should weight each time step equally.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """
    layer.weight.data.fill_(1.)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """
    Initialize a Batchnorm layer.

    Input
        bn: torch.Tensor - The batch normalisation layer
    """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock1d(nn.Module):
    """
    Creates an instance of a 1D convolutional layer. This includes the
    convolutional filter but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, kernel, stride, pad,
                 normalisation, dil=1):
        super(ConvBlock1d, self).__init__()
        self.norm = normalisation
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel,
                               stride=stride,
                               padding=pad,
                               dilation=dil)
        if self.norm == 'bn':
            self.bn1 = nn.BatchNorm1d(out_channels)
        elif self.norm == 'wn':
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
        else:
            self.conv1 = self.conv1
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input):
        """
        Passes the input through the convolutional filter

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm == 'bn':
            x = self.relu(self.bn1(self.conv1(x)))
        else:
            x = self.relu(self.conv1(x))

        return x


class ConvBlock2d(nn.Module):
    """
    Creates an instance of a 2D convolutional layer. This includes the
    convolutional filter but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, kernel, stride, pad,
                 normalisation, att=None):
        super(ConvBlock2d, self).__init__()
        self.norm = normalisation
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel,
                               stride=stride,
                               padding=pad)
        if self.norm == 'bn':
            self.bn1 = nn.BatchNorm2d(out_channels)
        elif self.norm == 'wn':
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
        else:
            self.conv1 = self.conv1
        self.att = att
        if not self.att:
            self.act = nn.ReLU()
        else:
            self.norm = None
            if self.att == 'softmax':
                self.act = nn.Softmax(dim=-1)
            elif self.att == 'global':
                self.act = None
            else:
                self.act = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        if self.att:
            init_att_layer(self.conv1)
        else:
            init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input):
        """
        Passes the input through the convolutional filter

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.att:
            x = self.conv1(x)
            if self.act():
                x = self.act(x)
        else:
            if self.norm == 'bn':
                x = self.act(self.bn1(self.conv1(x)))
            else:
                x = self.act(self.conv1(x))

        return x


class FullyConnected(nn.Module):
    """
    Creates an instance of a fully-connected layer. This includes the
    hidden layers but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, activation, normalisation,
                 att=None):
        super(FullyConnected, self).__init__()
        self.att = att
        self.norm = normalisation
        self.fc = nn.Linear(in_features=in_channels,
                            out_features=out_channels)
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
            self.norm = None
        elif activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
            self.norm = None
        elif activation == 'global':
            self.act = None
            self.norm = None
        else:
            self.act = nn.ReLU()
            if self.norm == 'bn':
                self.bnf = nn.BatchNorm1d(out_channels)
            elif self.norm == 'wn':
                self.wnf = nn.utils.weight_norm(self.fc, name='weight')

        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        if self.att:
            init_att_layer(self.fc)
        else:
            init_layer(self.fc)
        if self.norm == 'bn':
            init_bn(self.bnf)

    def forward(self, input):
        """
        Passes the input through the fully-connected layer

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm is not None:
            if self.norm == 'bn':
                x = self.act(self.bnf(self.fc(x)))
            else:
                x = self.act(self.wnf(x))
        else:
            if self.att:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)
            else:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)

        return x

def lstm_with_attention(net_params):
    if 'LSTM_1' in net_params:
        arguments = net_params['LSTM_1']
    else:
        arguments = net_params['GRU_1']
    if 'ATTENTION_1' in net_params and 'ATTENTION_Global' not in net_params:
        if arguments[-1]:
            return 'forward'
        else:
            return 'whole'
    if 'ATTENTION_1' in net_params and 'ATTENTION_Global' in net_params:
        if arguments[-1]:
            return 'forward'
        else:
            return 'whole'
    if 'ATTENTION_1' not in net_params and 'ATTENTION_Global' in net_params:
        if arguments[-1]:
            return 'forward_only'
        else:
            return 'forward_only'


def reshape_x(x):
    """
    Reshapes the input 'x' if there is a dimension of length 1

    Input:
        x: torch.Tensor - The input

    Output:
        x: torch.Tensor - Reshaped
    """
    dims = x.dim()
    if x.shape[1] == 1 and x.shape[2] == 1 and x.shape[3] == 1:
        x = torch.reshape(x, (x.shape[0], 1))
    elif dims == 4:
        first, second, third, fourth = x.shape
        if second == 1:
            x = torch.reshape(x, (first, third, fourth))
        elif third == 1:
            x = torch.reshape(x, (first, second, fourth))
        else:
            x = torch.reshape(x, (first, second, third))
    elif dims == 3:
        first, second, third = x.shape
        if second == 1:
            x = torch.reshape(x, (first, third))
        elif third == 1:
            x = torch.reshape(x, (first, second))

    return x

class CNN_LSTM_attention_input(nn.Module):
    def __init__(self,in_channels = 40, outputs = 1):
        super(CNN_LSTM_attention_input, self).__init__()
        # self.outputs = outputs
        self.conv = ConvBlock2d(in_channels=in_channels,
                                out_channels=in_channels*2,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool2d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.conv2 = ConvBlock2d(in_channels=in_channels*2,
                                out_channels=in_channels*4,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool2 = nn.MaxPool2d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.conv3 = ConvBlock2d(in_channels=in_channels*4,
                                out_channels=in_channels*8,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool3 = nn.MaxPool2d(kernel_size=3,
                                 stride=3,
                                 padding=0)

        self.conv4 = ConvBlock2d(in_channels=in_channels*8,
                                out_channels=in_channels*16,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool4 = nn.MaxPool2d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        # self.conv5 = ConvBlock2d(in_channels=128,
        #                         out_channels=128,
        #                         kernel=3,
        #                         stride=1,
        #                         pad=1,
        #                         normalisation='bn')
        # self.pool5 = nn.MaxPool2d(kernel_size=2,
        #                          stride=1,
        #                          padding=0)


        self.fc = FullyConnected(in_channels=192,
                                 out_channels=outputs,
                                 activation='global',  #sigmoid
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        # batch, freq, width = x.shape
        batch = x.shape[0]
        x = self.conv(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        # x = self.conv5(x)
        # x = self.pool5(x)

        x = x.reshape(x.size(0),x.size(1)*x.size(2)*x.size(3))
        # x = x.transpose(0,1)
        # x = x.view(x.size(0)*x.size(1)*x.size(2), -1)

        # x, _ = self.lstm(x)
        x = self.fc(x)

        return x


class SequenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SequenceClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through the LSTM layers
        out, _ = self.rnn(torch.transpose(x,1,2), (h0, c0))

        # Apply attention mechanism
        attention_weights = self.attention(out).squeeze(2)
        attention_scores = self.softmax(attention_weights)
        attention_out = torch.bmm(attention_scores.unsqueeze(1), out).squeeze(1)

        # Apply dropout
        out = self.dropout(attention_out)

        # Apply linear layer for classification
        logits = self.fc(out)

        return logits
class CustomMel1(nn.Module):
    def __init__(self,in_channels = 40, outputs = 1):
        super(CustomMel1, self).__init__()
        # self.outputs = outputs
        self.normalization1 = nn.BatchNorm1d(in_channels)
        self.conv = ConvBlock1d(in_channels=in_channels,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.conv2 = ConvBlock1d(in_channels=128,
                                out_channels=64,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool2 = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.conv3 = ConvBlock1d(in_channels=64,
                                out_channels=64,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool3 = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.normalization2 = nn.BatchNorm1d(64)
        self.gru = nn.GRU(input_size=64,
                            hidden_size=64,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=outputs,
                                 activation='global',  #sigmoid
                                 normalisation=None)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, x):
        batch, freq, width = x.shape
        x = self.normalization1(x)
        x = self.conv(x)
        x = self.pool(x)

        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x

class CustomEdward_attention(nn.Module):
    def __init__(self, timestep,in_channels = 40, outputs = 1, hidden_size = 128, batch_size = 128,device="cpu"):
        super(CustomEdward_attention, self).__init__()
        # self.outputs = outputs
        self.hidden_size = hidden_size
        self.device = device
        self.timestep = timestep
        self.batch_size = batch_size
        # self.attention = Attention(hidden_size, batch_size, method="dot")  # concat dot general
        # # attn_decoder1 = LuongDecoder(hidden_size, 1, attention, batch_size,n_layers=num_layers, drop_prob=Dropout_factor).to(device)
        # self.attn_decoder1 = LuongDecoderLocal(hidden_size, outputs, self.attention, batch_size, window_size=150,
        #                                   n_layers=1, drop_prob=0, net="LSTM",
        #                                   bidirectional=False).to(device)

        self.conv = ConvBlock1d(in_channels=in_channels,
                                out_channels=128,
                                kernel=5, # 3
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.conv2 = ConvBlock1d(in_channels=128,
                                out_channels=128,
                                kernel=3, # 3
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,  # 3
                                 stride=3,
                                 padding=0)
        # self.drop = nn.Dropout(0.05)
        self.trans = nn.Linear(332, timestep)

        self.attn = nn.Linear(hidden_size+timestep, 1)
        self.lstm = nn.LSTM(in_channels+1,hidden_size,
                            1, batch_first = True)  # if we are using embedding hidden_size should be added with embedding of vocab size
        self.final = nn.Linear(hidden_size, outputs)



    def init_hidden(self, input_size):
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
        return (torch.zeros((1, input_size, self.hidden_size), device=torch.device(self.device)),
                torch.zeros((1, input_size, self.hidden_size), device=torch.device(self.device)) )

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        #x = self.drop(x)
        x = self.conv(x)
        x = self.pool(x)
        # x = self.conv2(x)
        # x = self.pool(x)
        x=self.trans(x)
        #x = self.drop(x)
        # x = torch.transpose(x, 1, 2)
        # x, _ = self.lstm(x)
        # #x = self.drop(x)
        # x = self.fc(x[:, -1, :].reshape(batch, -1))
        # x = self.fc2(x)
        #x = self.fc3(x)
        #x = self.fc4(x)
        #x = self.fc2(x)

        weights = []
        decoder_hidden = self.init_hidden(net_input.size(0))
        for i in range(x.size(1)):
            # print(decoder_hidden[0][0].shape)
            # print(encoder_outputs[0].shape)
            try:
                weights.append(self.attn(torch.cat((decoder_hidden[0][0],
                                                x[:,i,:]), dim=1)))
            except RuntimeError:
                a = 1
        normalized_weights = F.softmax(torch.cat(weights, 1), 1)

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
                                 x)

        input_lstm = torch.transpose(torch.cat((attn_applied, net_input),
                               dim=1),1,2)  # if we are using embedding, use embedding of input here instead

        output, hidden = self.lstm(input_lstm, decoder_hidden)

        output = self.final(output[:,-1,:])

        return output#, hidden, normalized_weights


        # return x


class CustomEdward(nn.Module):
    def __init__(self,in_channels = 40, outputs = 1, hidden_size = 128, batch_size = 128, device="cpu"):
        super(CustomEdward, self).__init__()
        # self.outputs = outputs
        self.attention = Attention(hidden_size, batch_size, method="dot")  # concat dot general
        # attn_decoder1 = LuongDecoder(hidden_size, 1, attention, batch_size,n_layers=num_layers, drop_prob=Dropout_factor).to(device)
        self.attn_decoder1 = LuongDecoderLocal(hidden_size, outputs, self.attention, batch_size, window_size=150,
                                          n_layers=1, drop_prob=0, net="LSTM",
                                          bidirectional=False).to(device)

        self.conv = ConvBlock1d(in_channels=in_channels,
                                out_channels=128,
                                kernel=5, # 3
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.conv2 = ConvBlock1d(in_channels=128,
                                out_channels=128,
                                kernel=3, # 3
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,  # 3
                                 stride=3,
                                 padding=0)
        self.pool2 = nn.MaxPool1d(kernel_size=3,  # 3
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.05)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=64,
                                 activation='global',  #sigmoid
                                 normalisation=None)
        self.fc2 = FullyConnected(in_channels=64,
                                 out_channels=outputs,
                                 activation='global',  #sigmoid
                                 normalisation=None)
        #self.fc3 = FullyConnected(in_channels=256,
         #                        out_channels=128,
          #                       activation='global',  #sigmoid
           #                      normalisation=None)
        #self.fc4 = FullyConnected(in_channels=,
         #                        out_channels=outputs,
          #                       activation='global',  #sigmoid
           #                      normalisation=None)


    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        #x = self.drop(x)
        x = self.conv(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool2(x)

        #x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        #x = self.drop(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))
        x = self.fc2(x)
        #x = self.fc3(x)
        #x = self.fc4(x)
        #x = self.fc2(x)
        return x



class CustomMel2(nn.Module):
    def __init__(self):
        super(CustomMel2, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='global',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)

        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel3(nn.Module):
    def __init__(self):
        super(CustomMel3, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='global',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)

        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel4(nn.Module):
    def __init__(self):
        super(CustomMel4, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=4,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='global',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)

        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel5(nn.Module):
    def __init__(self):
        super(CustomMel5, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=5,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='global',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)

        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel6(nn.Module):
    def __init__(self, in_channels = 40, outputs = 1):
        super(CustomMel6, self).__init__()
        self.conv = ConvBlock1d(in_channels=in_channels,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.3) #0.05 0.20  0.3
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=outputs,
                                 activation='global',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        x = self.drop(x)
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel7(nn.Module):
    def __init__(self):
        super(CustomMel7, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.05)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel8(nn.Module):
    def __init__(self):
        super(CustomMel8, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.05)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel9(nn.Module):
    def __init__(self):
        super(CustomMel9, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.05)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=4,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel10(nn.Module):
    def __init__(self):
        super(CustomMel10, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.05)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=5,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel11(nn.Module):
    def __init__(self):
        super(CustomMel11, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.15)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel12(nn.Module):
    def __init__(self):
        super(CustomMel12, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.15)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel13(nn.Module):
    def __init__(self):
        super(CustomMel13, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.15)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel14(nn.Module):
    def __init__(self):
        super(CustomMel14, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.15)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=4,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomMel15(nn.Module):
    def __init__(self):
        super(CustomMel15, self).__init__()
        self.conv = ConvBlock1d(in_channels=40,
                                out_channels=128,
                                kernel=3,
                                stride=1,
                                pad=1,
                                normalisation='bn')
        self.pool = nn.MaxPool1d(kernel_size=3,
                                 stride=3,
                                 padding=0)
        self.drop = nn.Dropout(0.15)
        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=5,
                            batch_first=True,
                            bidirectional=False)
        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv(x)
        x = self.pool(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomRaw1(nn.Module):
    def __init__(self):
        super(CustomRaw1, self).__init__()
        # x = [(in + (2*pad) - (kernel-1) - 1) / stride] + 1
        self.conv1 = ConvBlock1d(in_channels=1,
                                 out_channels=128,
                                 kernel=1024,
                                 stride=512,
                                 pad=0,
                                 dil=1,
                                 normalisation='bn')

        self.conv2 = ConvBlock1d(in_channels=128,
                                 out_channels=128,
                                 kernel=3,  # 6874
                                 stride=1,
                                 pad=1,
                                 normalisation='bn')

        self.pool1 = nn.MaxPool1d(kernel_size=3,
                                  stride=3,
                                  padding=0)

        self.drop = nn.Dropout(.05)

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomRaw2(nn.Module):
    def __init__(self):
        super(CustomRaw2, self).__init__()
        # x = [(in + (2*pad) - (kernel-1) - 1) / stride] + 1
        self.conv1 = ConvBlock1d(in_channels=1,
                                 out_channels=128,
                                 kernel=512,
                                 stride=256,
                                 pad=0,
                                 dil=1,
                                 normalisation='bn')

        self.conv2 = ConvBlock1d(in_channels=128,
                                 out_channels=128,
                                 kernel=3,  # 6874
                                 stride=1,
                                 pad=1,
                                 normalisation='bn')

        self.pool1 = nn.MaxPool1d(kernel_size=3,
                                  stride=3,
                                  padding=0)
        self.drop = nn.Dropout(.05)

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomRaw3(nn.Module):
    def __init__(self):
        super(CustomRaw3, self).__init__()
        # x = [(in + (2*pad) - (kernel-1) - 1) / stride] + 1
        self.conv1 = ConvBlock1d(in_channels=1,
                                 out_channels=128,
                                 kernel=1024,
                                 stride=512,
                                 pad=0,
                                 dil=1,
                                 normalisation='bn')

        self.conv2 = ConvBlock1d(in_channels=128,
                                 out_channels=128,
                                 kernel=3,  # 6874
                                 stride=1,
                                 pad=1,
                                 normalisation='bn')

        self.pool1 = nn.MaxPool1d(kernel_size=3,
                                  stride=3,
                                  padding=0)

        self.drop = nn.Dropout(.05)

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x


class CustomRaw4(nn.Module):
    def __init__(self):
        super(CustomRaw4, self).__init__()
        # x = [(in + (2*pad) - (kernel-1) - 1) / stride] + 1
        self.conv1 = ConvBlock1d(in_channels=1,
                                 out_channels=128,
                                 kernel=512,
                                 stride=256,
                                 pad=0,
                                 dil=1,
                                 normalisation='bn')

        self.conv2 = ConvBlock1d(in_channels=128,
                                 out_channels=128,
                                 kernel=3,  # 6874
                                 stride=1,
                                 pad=1,
                                 normalisation='bn')

        self.pool1 = nn.MaxPool1d(kernel_size=3,
                                  stride=3,
                                  padding=0)
        self.drop = nn.Dropout(.05)

        self.lstm = nn.LSTM(input_size=128,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        self.fc = FullyConnected(in_channels=128,
                                 out_channels=1,
                                 activation='sigmoid',
                                 normalisation=None)

    def forward(self, net_input):
        x = net_input
        batch, freq, width = x.shape
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.drop(x)
        x = torch.transpose(x, 1, 2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :].reshape(batch, -1))

        return x
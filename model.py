# Importing Libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# import librosa
# import IPython.display as ipd

from torchvision.io import read_video

from sklearn.model_selection import train_test_split

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/ConvLSTM_pytorch
from ConvLSTM_pytorch.convlstm import ConvLSTM


if sys.version_info[0] < 3:
   # for Python 2
   import cPickle as pickle
else:
   # for Python 3
   import pickle

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device=", device)

emotions = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
label_map = {emotions[i] : i for i in range(len(emotions))} # 'ang':0, 'fru':1, ...

"""### Define Audio AE Model """

class ConvAutoencAudio(nn.Module):
    def __init__(self):
        super(ConvAutoencAudio, self).__init__()
        
        # Encoder LSTM layer
        self.lstm1 = nn.LSTM(1, 16)

        # Decoder LSTM layer
        self.lstm2 = nn.LSTM(16, 1)

        self.latent_space = None

    def encoder(self, x):
        #Analysis:
        y, (h,c) = self.lstm1(x)
        y = torch.tanh(y)
        # y = y.permute(0,2,1)

        self.latent_space = y
        return y
      
    def decoder(self, y):
        #Synthesis:
        y,_ = self.lstm2(y)
        y = torch.tanh(y)
        return y
      
    def forward(self, x):
        y=self.encoder(x)
        # y=torch.round(y/0.125)*0.125
        y = self.decoder(y)
        return y #xrek

def init_audio_model():
    audio_loss_func = nn.MSELoss()
    audio_lr = 0.0001
    torch.manual_seed(0)

    audio_model = ConvAutoencAudio()

    audio_optim = torch.optim.Adam(audio_model.parameters(), lr=audio_lr)

    audio_model.to(device)

    summary(audio_model, (1,204459,1))

    return audio_model, audio_optim, audio_loss_func


"""### Define Video AE Model"""

class TimeDistributed(nn.Module):

    def __init__(self, layer, summary=False):
        super(TimeDistributed, self).__init__()
        self.layer = layer
        self.summary = summary


    def forward(self, input_seq, output_size=None):

        if len(input_seq.size()) <= 2:
            return input_seq

        if self.summary:
            print("Module:",self.layer)
            print("Input size:", input_seq.size())

        # reshape input from (batch_size, num_frames, num_channels, width, height) to (batch_size * num_frames, num_channels, width, height)
        squashed_inp = input_seq.view(-1, *input_seq.shape[2:])

        # apply layer(module) on the squashed input
        if type(self.layer) == nn.ConvTranspose2d:
            layer_output = self.layer(squashed_inp, output_size=output_size)
        else:
            layer_output = self.layer(squashed_inp)
        
        # reshape output back to (batch_size, num_frames, num_channels, width, height) shape
        layer_output = layer_output.contiguous().view(-1, input_seq.size(1), *layer_output.shape[1:])

        if self.summary:
            print("Output size:", layer_output.size())
            print()
            
        return layer_output

class EncoderLSTM(nn.Module):
    
    def __init__(self, num_channels, summary=False):
        super().__init__()
        self.summary = summary

        self.encoderCNN= nn.Sequential(
            TimeDistributed(nn.Conv2d(in_channels=num_channels, out_channels=128, kernel_size=11, stride=4, padding=4), summary),
            nn.ReLU(True),
            # TimeDistributed(nn.BatchNorm2d(128), summary),

            TimeDistributed(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=5, stride=2, padding=2), summary),
            TimeDistributed(nn.BatchNorm2d(64), summary),
            nn.ReLU(True),

            # TimeDistributed(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2), summary),
            # nn.ReLU(True),
            # TimeDistributed(nn.BatchNorm2d(32), summary),
        )
        self.relu = nn.ReLU(True)

        self.convLSTM1 = ConvLSTM(input_dim=64, hidden_dim=32, kernel_size=(3,3), num_layers=1, batch_first=True, return_all_layers=True)
        self.convLSTM2 = ConvLSTM(input_dim=32, hidden_dim=16, kernel_size=(3,3), num_layers=1, batch_first=True, return_all_layers=True)

        self.batch_norm1 = TimeDistributed(nn.BatchNorm2d(32), summary)
        self.batch_norm2 = TimeDistributed(nn.BatchNorm2d(16), summary)

        # self.time_flatten = TimeDistributed(nn.Flatten(), summary)
        # self.lstm = nn.LSTM(np.prod([32, 15, 15]), 16)
        self.flatten = nn.Flatten()
        

    def printSummary(self, is_output, size, module=None):

        if not is_output:
            print("Module:", module)
            print("Input size:", size)
        else:
            print("Output size:", size)
            print()

    def forward_convLSTM(self, x):

        if self.summary:
            self.printSummary(False, x.size(), self.convLSTM1._get_name())
        x, _ = self.convLSTM1(x)
        if self.summary:
            self.printSummary(True, x[0].size())

        # x = self.relu(x[0])
        x = self.batch_norm1(x[0])

        if self.summary:
            self.printSummary(False, x.size(), self.convLSTM2._get_name())
        x, _ = self.convLSTM2(x)
        if self.summary:
            self.printSummary(True, x[0].size())
        
        # x = self.relu(x[0])
        x = self.batch_norm2(x[0])

        return x

    def forward_linear(self, x):

        # x = self.time_flatten(x)
        # if self.summary:
        #     self.printSummary(False, x.size(), self.lstm)
        # x, _ = self.lstm(x)
        # if self.summary:
        #     self.printSummary(True, x.size())

        if self.summary:
            self.printSummary(False, x.size(), self.flatten)
        x = self.flatten(x)
        if self.summary:
            self.printSummary(True, x.size())

        return x


    def forward(self, x):
        x = self.encoderCNN(x)
        x = self.forward_convLSTM(x)
        x = self.forward_linear(x)

        if self.summary:
            print("*******************************************")

        return x

class DecoderLSTM(nn.Module):

    def __init__(self, num_channels, summary=False):
        super().__init__()

        self.summary = summary

        self.convLSTM1 = ConvLSTM(input_dim=16, hidden_dim=32, kernel_size=(3,3), num_layers=1, batch_first=True, return_all_layers=True)
        self.batch_norm1 = TimeDistributed(nn.BatchNorm2d(32), summary)

        self.convLSTM2 = ConvLSTM(input_dim=32, hidden_dim=64, kernel_size=(3,3), num_layers=1, batch_first=True, return_all_layers=True)
        self.batch_norm2 = TimeDistributed(nn.BatchNorm2d(64), summary)

        # self.convTranspose1 = TimeDistributed(nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2), summary)
        # self.batch_norm3 = TimeDistributed(nn.BatchNorm2d(64), summary)

        self.convTranspose2 = TimeDistributed(nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2), summary)
        self.batch_norm4 = TimeDistributed(nn.BatchNorm2d(128), summary)

        self.convTranspose3 = TimeDistributed(nn.ConvTranspose2d(in_channels=128, out_channels=num_channels, kernel_size=11, stride=4, padding=4), summary)
        self.batch_norm5 = TimeDistributed(nn.BatchNorm2d(num_channels), summary)

        self.sigmoid = nn.Sigmoid()
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=[138, 16, 15, 15])

        self.relu = nn.ReLU(True)

    def printSummary(self, is_output, size, module=None):

        if not is_output:
            print("Module:", module)
            print("Input size:", size)
        else:
            print("Output size:", size)
            print()  

    
    def forward_CNN(self, x):
        # x = self.convTranspose1(x, output_size=[20, 30])
        # x = self.batch_norm3(x)
        # x = self.relu(x)

        x = self.convTranspose2(x, output_size=[30, 30])
        x = self.batch_norm4(x)
        x = self.relu(x)

        x = self.convTranspose3(x, output_size=[120, 120])
        # x = self.batch_norm5(x)
        # x = self.relu(x)

        x = self.sigmoid(x)

        return x

    def forward_convLSTM(self, x):
        if self.summary:
            self.printSummary(False, x.size(), self.convLSTM1._get_name())
        x, _ = self.convLSTM1(x)
        if self.summary:
            self.printSummary(True, x[0].size())

        # x = self.relu(x[0])
        x = self.batch_norm1(x[0])

        if self.summary:
            self.printSummary(False, x.size(), self.convLSTM2._get_name())        
        x, _ = self.convLSTM2(x)
        if self.summary:
            self.printSummary(True, x[0].size())
        x = self.batch_norm2(x[0])

        return x

    def forward_linear(self, x):
        if self.summary:
            self.printSummary(False, x.size(), self.unflatten)
        x = self.unflatten(x)
        if self.summary:
            self.printSummary(True, x.size())

        # if self.summary:
        #     self.printSummary(False, x.size(), self.lstm)
        # x, _ = self.lstm(x)
        # if self.summary:
        #     self.printSummary(True, x.size())

        # x = self.td_unflatten(x)

        return x

    def forward(self, x):
        x = self.forward_linear(x)
        x = self.forward_convLSTM(x)
        x = self.forward_CNN(x)

        return x

class VideoAutoencoder(nn.Module):

    def __init__(self, num_channels, summary=False):
        super().__init__()

        self.encoder = EncoderLSTM(num_channels=num_channels, summary=summary)
        self.decoder = DecoderLSTM(num_channels=num_channels, summary=summary)

        self.latent_space = None

    def forward(self, x):
        x = self.encoder(x)
        self.latent_space = x

        x = self.decoder(x)

        return x

def init_video_model():
    # # Print Summary of the Autoencoder using sample video
    # # encoder = EncoderLSTM(num_channels=3, summary=True)
    # # decoder = DecoderLSTM(num_channels=3, summary=True)
    # videoAE = VideoAutoencoder(num_channels=1, summary=True)

    # # op = encoder(train_video_data[0].unsqueeze(0))
    # # op = decoder(encoder(train_video_data[0].unsqueeze(0)))
    # op = videoAE(train_video_data[0][0].unsqueeze(0))
    # # summary(encoder, (1, 135, 3, 160, 240))

    video_loss_func = torch.nn.MSELoss()
    video_lr = 0.0001
    torch.manual_seed(0)

    num_channels = 1

    video_model = VideoAutoencoder(num_channels=num_channels)

    params_to_optimise = [
        {'params' : video_model.parameters()}
    ]

    video_optim = torch.optim.Adam(params_to_optimise, lr=video_lr, weight_decay=1e-05)

    video_model.to(device)

    return video_model, video_optim, video_loss_func

"""## Classifier

### Define Classifier Model
"""

class LatentClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        ## Architecture
        # DenseUnit
        # ReLU
        # BatchNorm
        # Dropout
        # Dense Unit

        # Layers for downsampling the 1D output
        self.convClassifier = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=128, stride=32, bias=True), #Padding for 'same' filters (kernel_size/2-1)
            nn.ReLU(True),
            # nn.BatchNorm1d(32),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=32, stride=16, bias=True), #padding=255
            nn.ReLU(True),

            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=11, stride=4, bias=True), #padding=255
            nn.ReLU(True),

            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=5, stride=2, bias=True), #padding=255
            nn.ReLU(True),
        )


        #Linear (Dense) unit classifier
        self.classifier = nn.Sequential(       

            nn.Flatten(start_dim=1),     
            nn.Linear(np.prod([8, 917]), 64),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.BatchNorm1d(512),
            nn.Linear(64, len(emotions))
            # nn.ReLU(True)
            # nn.Softmax(dim=1)
        )
      
    def forward(self, x):
        # print("Sizes:")
        # print(x.size())
        x = self.convClassifier(x)
        # print(x.size())
        x = self.classifier(x)

        return x

def init_classifier_model():
    clf_loss_func = nn.CrossEntropyLoss()
    clf_lr = 0.0001
    torch.manual_seed(0)

    classifier_model = LatentClassifier()

    clf_optim = torch.optim.Adam(classifier_model.parameters(), lr=clf_lr)

    classifier_model.to(device)

    summary(classifier_model, (1, 1, (np.prod([204459, 16]) + np.prod([138, 16, 15, 15]))))

    return classifier_model, clf_optim, clf_loss_func
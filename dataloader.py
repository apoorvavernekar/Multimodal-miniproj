# Importing Libraries

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision.io import read_video

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random

import librosa
# import IPython.display as ipd

from sklearn.model_selection import train_test_split


if sys.version_info[0] < 3:
   # for Python 2
   import cPickle as pickle
else:
   # for Python 3
   import pickle



"""Save Audio Values"""
# mean_samples = 197917
# min_samples = 44540
# max_samples = 1179675
# std_samples = 138305
# audio_percentiles = {2.5: 55125.0, 10: 73647.0, 25: 100988.0,
#                50: 152921.0, 75: 260520.0, 90: 379987.3,
#                97.5: 570858}
# data_samplerate = 44100

# Saved values of cropped audio data
mean_samples = 204459
min_samples = 25798
max_samples = 1284632
std_samples = 138020
audio_percentiles = {2.5: 57771, 10: 79247, 25: 108486, 
               50: 162728, 75: 257984, 90: 387673, 
               97.5: 577370}
data_samplerate = 44100


"""Saved video values"""
# # Saved values found from previous section
# mean_frames = 135
# min_frames = 31
# max_frames = 802
# std_frames = 94
# video_percentiles = {2.5: 38, 10: 51, 25: 69, 
#                50: 104, 75: 178, 
#                90: 259, 97.5: 388}

# Saved values for cropped data 
mean_frames = 138
min_frames = 17
max_frames = 873
std_frames = 93
video_percentiles = {2.5: 39, 10: 53, 25: 73, 
               50: 110, 75: 174, 
               90: 263, 97.5: 391}

emotions = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
label_map = {emotions[i] : i for i in range(len(emotions))} # 'ang':0, 'fru':1, ...


def get_avpaths(audio_dir):
    # audio_dir='/content/drive/MyDrive/Autoencoder_proj/cropped_data/audio'
    # video_dir='/content/drive/MyDrive/Autoencoder_proj/cropped_data/videos'

    aud_vid_paths = []
    labels = []

    for dirpath, dirnames, filenames in os.walk(audio_dir):
        print(dirpath)
        split_path = dirpath.split(os.path.sep)[-2:]
        class_path = os.path.sep.join(split_path)

        for audi in filenames:
            labels.append(split_path[0])
            av_path=os.path.join(class_path,audi[:-4])
            aud_vid_paths.append(av_path)

    aud_vid_paths = np.array(aud_vid_paths)
    labels = np.array(labels)

    return aud_vid_paths, labels


def get_train_val_test_ind(num_data, labels):

    # num_data = len(indices)
    indices = np.array(range(num_data))

    # Split data into train, validation and test data
    train_size = int(num_data * 0.7)
    test_size = num_data - train_size

    train_ind, test_ind, train_y, test_y = train_test_split(indices, labels, train_size=train_size, test_size=test_size) #, stratify=labels)

    val_size = int(test_size * 0.5)
    test_size = test_size - val_size

    val_ind, test_ind, val_y, test_y = train_test_split(test_ind, test_y, train_size=val_size, test_size=test_size) #, stratify=test_y)

    return train_ind, val_ind, test_ind

"""## Audio Autoencoder

### Prepare Audio Data
"""

# Create Custom Dataset to read audios
class AudioDataset(Dataset):

    def __init__(self, basepath, audios_list, max_samples_per_audio=mean_samples, transform=None):
        self.basepath = basepath
        self.audios_list = audios_list
        self.max_samples_per_audio = max_samples_per_audio
        self.transform = transform

    def signal2pytorch(self, x):
        """
        Function to convert a signal vector x, like a mono audio signal, into a 3-d Tensor that conv1d of Pytorch expects,
        https://pytorch.org/docs/stable/nn.html
        Argument x: a 1-d signal as numpy array
        input x[batch,sample]
        output: 2-d Tensor X for conv1d input.
        for conv1d Input: (N,Cin,Lin), Cin: numer of input channels (e.g. for stereo), Lin: length of signal, N: number of Batches (signals) 
        """
        X = np.expand_dims(x, axis=1)  #add channels dimension (here only 1 channel)
        X = torch.from_numpy(X)
        X = X.type(torch.Tensor)
        return X        

    def generate_stride_seq(self, num_audio_samples):

        if num_audio_samples <= audio_percentiles[75]:
            stride_seq = [j for i in range(0, num_audio_samples-1, 3) for j in (i,i+1)]
        else:
            stride_seq = [i for i in range(0, num_audio_samples, 2)]

        stride_seq = stride_seq[:self.max_samples_per_audio]

        return stride_seq


    def read_audio_from_index(self, index, take_strides=False):

        # print("Audio:",self.audios_list[index])
        path = os.path.join(self.basepath, self.audios_list[index]+".wav")
        audio, samplerate = librosa.load(path, mono=True, sr=None, offset=0)
        audio /= np.abs(audio).max()

        if audio.shape[0] > self.max_samples_per_audio:
            if not take_strides:
                audio = audio[:self.max_samples_per_audio]
            else:
                stride_seq = self.generate_stride_seq(audio.shape[0])
                audio = audio[stride_seq]

        return audio    


    def pad_audios(self, audio, empty=False):

        rem_samples = self.max_samples_per_audio - audio.shape[0]

        if empty:
            extra_samples = np.zeros((rem_samples,))
        else:
            extra_samples = audio[:rem_samples]

        audio = np.append(audio, extra_samples, axis=0)

        return audio


    def __getitem__(self, index):

        audio = self.read_audio_from_index(index, take_strides=False)

        while audio.shape[0] < self.max_samples_per_audio:
            audio = self.pad_audios(audio)
        
        audio = self.signal2pytorch(audio)

        if self.transform:
            audio = self.transform(audio)    
        
        return audio


    def __len__(self):
        return len(self.audios_list)

def init_audio_data(audio_dir, aud_vid_paths):

    # Create Dataset object
    audio_data = AudioDataset(audio_dir, aud_vid_paths, max_samples_per_audio=mean_samples)
    transform = T.Compose([])

    audio_data.transform = transform

    return audio_data


def load_audio_data(audio_data, train_ind, val_ind, test_ind):
    # Split data into train, validation and test data
    train_audio_data = Subset(audio_data, train_ind)
    val_audio_data = Subset(audio_data, val_ind)
    test_audio_data = Subset(audio_data, test_ind)
    batch_size = 1

    # Create dataloaders for train, validation and test data 
    train_audio_loader = DataLoader(train_audio_data, batch_size = batch_size, shuffle=False)
    val_audio_loader = DataLoader(val_audio_data, batch_size = batch_size, shuffle=False)
    test_audio_loader = DataLoader(test_audio_data, batch_size = batch_size, shuffle=False)

    return train_audio_loader, val_audio_loader, test_audio_loader


"""## Video Autoencoder

### Prepare Video Data
"""

# Create Custom Dataset to read videos
class VideoDataset(Dataset):

    def __init__(self, basepath, videos_list, max_frames_per_video=mean_frames, transform=None):
        self.basepath = basepath
        self.videos_list = videos_list
        self.max_frames_per_video = max_frames_per_video
        self.transform = transform

        self.emotions = emotions
        self.label_map = label_map


    def generate_stride_seq(self, num_video_frames):

        if num_video_frames <= video_percentiles[75]:
            stride_seq = [j for i in range(0, num_video_frames-1, 3) for j in (i,i+1)]
        else:
            stride_seq = [i for i in range(0, num_video_frames, 2)]

        stride_seq = stride_seq[:self.max_frames_per_video]

        return stride_seq


    def read_video_from_index(self, index):

        label = self.videos_list[index].split(os.path.sep)[0]

        # print("Video:",self.videos_list[index], 'Label:', label)
        path = os.path.join(self.basepath, self.videos_list[index]+".npy")
        
        # load video from .npy file
        frames = np.load(path)
        frames = np.expand_dims(frames,len(frames.shape))
        frames = torch.from_numpy(frames)

        ## Load video from a .avi file
        # frames, _, _ = read_video(path)

        if frames.size(0) > self.max_frames_per_video:
            stride_seq = self.generate_stride_seq(frames.size(0))
            frames = frames[stride_seq]

        return frames, self.label_map[label]    


    def pad_videos(self, frames, empty=False):

        rem_frames = self.max_frames_per_video - frames.size(0)

        if empty:
            extra_frames = torch.empty((rem_frames, frames.size(1), frames.size(2), frames.size(3)))
        else:
            extra_frames = frames[:rem_frames]

        frames = torch.cat((frames, extra_frames))

        return frames


    def __getitem__(self, index):

        frames, label = self.read_video_from_index(index)
        
        while frames.size(0) < self.max_frames_per_video:
            frames = self.pad_videos(frames)

        frames = frames.permute(0,3,1,2)

        if self.transform:
            frames = self.transform(frames)    
        
        return frames, label


    def __len__(self):
        return len(self.videos_list)

def init_video_data(video_dir, aud_vid_paths):

    # Create Dataset object
    video_data = VideoDataset(video_dir, aud_vid_paths, max_frames_per_video=mean_frames)
    transform = T.Compose(
        [
        T.ConvertImageDtype(torch.float32),
        #  T.Resize(size=(480//3, 720//3))
        ]
    )

    video_data.transform = transform

    return video_data

def load_video_data(video_data, train_ind, val_ind, test_ind):
    # Split data into train, validation and test data
    train_video_data = Subset(video_data, train_ind)
    val_video_data = Subset(video_data, val_ind)
    test_video_data = Subset(video_data, test_ind)
    batch_size = 1

    # Split data into train, validation and test data
    train_video_loader = DataLoader(train_video_data, batch_size = batch_size, shuffle=False)
    val_video_loader = DataLoader(val_video_data, batch_size = batch_size, shuffle=False)
    test_video_loader = DataLoader(test_video_data, batch_size = batch_size, shuffle=False)

    return train_video_loader, val_video_loader, test_video_loader

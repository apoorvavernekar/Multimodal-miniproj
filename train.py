# Importing Libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, Subset
# import torchvision.transforms as T
# from torchinfo import summary

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

import librosa
# import IPython.display as ipd

from torchvision.io import read_video

from sklearn.model_selection import train_test_split

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/ConvLSTM_pytorch
from ConvLSTM_pytorch.convlstm import ConvLSTM

import dataloader
import model


if sys.version_info[0] < 3:
   # for Python 2
   import cPickle as pickle
else:
   # for Python 3
   import pickle

maxx=0

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device=", device)

emotions = ['ang', 'exc', 'fru', 'hap', 'neu', 'sad']
label_map = {emotions[i] : i for i in range(len(emotions))} # 'ang':0, 'fru':1, ...


"""## Combine and Train

### Plotting Functions
"""

def concat(aud_latent, vid_latent):

    # print("Aud Latent Shape:", aud_latent.size())
    # print("Vid Latent Shape:", vid_latent.size())

    # Squashing aidio latent space to one dimension
    aud_latent = aud_latent.contiguous().view(aud_latent.size(0),-1)

    #Video latent space shape - torch.Size([1, 135, 16, 15, 15])

    ## Squash video autoencoder if not 1-dimensional 
    # vid_latent = vid_latent.contiguous().view(-1, *vid_latent.shape)
    # vid_latent = vid_latent.permute(0,2,1,3,4)
    # vid_latent_view = vid_latent.contiguous().view(*(vid_latent.shape[:2]),-1)
    # print("final size:",vid_latent.size()) 

    # print("Aud Latent Shape:", aud_latent.size())
    # print("Vid Latent Shape:", vid_latent.size())

    data = torch.cat((aud_latent, vid_latent), dim=1)
    data = data.unsqueeze(0)

    # print("Concat Shape:", data.size())

    return data

# from sklearn.metrics import ConfusionMatrixDisplay

# Accuracy Calculation
def calc_accuracy(preds, actual_labels):
    # print("actual labels:", actual_labels)

    pred_labels = None
    for pred in preds:
        pred_softmax = torch.log_softmax(pred, 1) #[_ _ _ _ _ _]
        _, max_pred_ind = torch.max(pred_softmax, 1)
        pred_labels = torch.cat([pred_labels, max_pred_ind], 0) if pred_labels is not None else max_pred_ind

    # print(max_pred_ind)
    # print(actual_labels)

    # print("pred labels:", pred_labels)
    pred_correct = (pred_labels == actual_labels.to(device)).float()
    acc = pred_correct.sum() / len(actual_labels)

    # plt.figure(figsize=(15,15))
    
    # print("Confusion Matrices:")
    # cm_normalize_type = ['true', 'pred', 'all', None]
    # for i in range(len(cm_normalize_type)):
    #     ax = plt.subplot(2,2,i+1)
    #     ConfusionMatrixDisplay.from_predictions(actual_labels.detach().cpu().numpy(), 
    #                                             pred_labels.detach().cpu().numpy(), 
    #                                             normalize=cm_normalize_type[i],
    #                                             ax=ax)
    # plt.show()

    return acc

"""#### Plotting functions for model training"""

def plot_video_outputs(ae_model, data, n=10):

    plt.figure(figsize=(32, 9))
    ae_model.eval()

    latents = None
    true_labels = []

    for i in range(n):
        img, label = data[i]
        # print(type(label), label)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            rec_img = ae_model(img)
            latents = torch.cat([latents, ae_model.latent_space.unsqueeze(0)]) if latents is not None else ae_model.latent_space.unsqueeze(0)

        true_labels.append(label) #= torch.cat([true_labels, label], dim=0) if true_labels is not None else label

        ax = plt.subplot(2,n,i+1)
        orig_frame = img.cpu().squeeze()[i] #.permute(1,2,0)
        rec_frame = rec_img.cpu().squeeze()[i] #.permute(1,2,0)

        plt.imshow(orig_frame.numpy())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n//2:
            ax.set_title('First {} Original Video frames'.format(n))

        # print("n:",n,"i+1+n:", i+1+n)
        ax = plt.subplot(2,n,i+1+n)
        plt.imshow(rec_frame.numpy())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n//2:
            ax.set_title('First {} Reconstructed Video Frames'.format(n))
    plt.show()

    true_labels = torch.Tensor(true_labels) #, dtype=torch.long)
    return latents, true_labels

def plot_audio_outputs(model, data, n=10):

    n = min(n, len(data))
    plt.figure(figsize=(32, 9))
    rows = (n // 2) + (n % 2)

    latents = None

    for i in range(n):
        audio = data[i].unsqueeze(0).to(device)
        model.eval()

        with torch.no_grad():
            pred_audio = model(audio)
            latents = torch.cat([latents, model.latent_space.unsqueeze(0)]) if latents is not None else model.latent_space.unsqueeze(0)

        ind = i+1 
        ax = plt.subplot(rows,2,ind)
        orig_aud = audio.cpu().squeeze().squeeze()
        pred_aud = pred_audio.cpu().squeeze().squeeze()
        # print("pred size:", pred_aud.size())
        # print("pred min max", torch.min(pred_aud), torch.max(pred_aud))

        ax.plot(orig_aud, alpha=0.7)
        ax.plot(pred_aud, alpha=0.7)

        print('\nOriginal Audio (Top), Predicted Audio(Bottom)')
        display(ipd.Audio(orig_aud, rate=data_samplerate))
        display(ipd.Audio(pred_aud, rate=data_samplerate))
        print("\n************************************************\n")

    plt.show()

    return latents

def plot_outputs(audio_ae_model, video_ae_model, clf_model, aud_data, vid_data, n=5):

    n = min(n, len(aud_data))
    aud_latents = plot_audio_outputs(audio_ae_model, aud_data, n)
    vid_latents, true_labels = plot_video_outputs(video_ae_model, vid_data, n)

    preds = None

    for i in range(n):

        clf_model.eval()

        av_latent = concat(aud_latents[i], vid_latents[i])

        pred_class = clf_model(av_latent)

        preds = torch.cat([preds, pred_class.unsqueeze(0)]) if preds is not None else pred_class.unsqueeze(0)

    test_acc = calc_accuracy(preds, true_labels)

    print("Classifier Accuracy:", test_acc)

    plt.show()

"""### Validating/Testing Functions"""

def plot_test_video_outputs(orig_videos, rec_videos):

    n = len(orig_videos)
    plt.figure(figsize=(32, 9))

    for i in range(n):
        img, rec_img = orig_videos[i], rec_videos[i]

        ax = plt.subplot(2,n,i+1)
        orig_frame = img.cpu().squeeze()[0] #.permute(1,2,0)
        rec_frame = rec_img.cpu().squeeze()[0] #.permute(1,2,0)

        plt.imshow(orig_frame.numpy())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n//2:
            ax.set_title('First {} Original Video frames'.format(n))

        # print("n:",n,"i+1+n:", i+1+n)
        ax = plt.subplot(2,n,i+1+n)
        plt.imshow(rec_frame.numpy())
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        if i == n//2:
            ax.set_title('First {} Reconstructed Video Frames'.format(n))

    plt.show()


def test_video_model(ae_model, data_loader, loss_fn, device, vis_n=5):

    ae_model.eval()

    latents = None
    true_labels = None
    orig_videos = []
    rec_videos = []
    losses = []
    i = 0 

    for video, emo_label in data_loader:

        video = video.to(device)
        emo_label = emo_label.to(device)

        with torch.no_grad():
            rec_vid = ae_model(video)
            latents = torch.cat([latents, ae_model.latent_space.unsqueeze(0)]) if latents is not None else ae_model.latent_space.unsqueeze(0)

        true_labels = torch.cat([true_labels, emo_label], dim=0) if true_labels is not None else emo_label #true_labels.append(emo_label) #.detach().cpu().numpy()) #= torch.cat([true_labels, label], dim=0) if true_labels is not None else label

        rec_loss = loss_fn(rec_vid, video)
        losses.append(rec_loss.detach().cpu().numpy())

        if i < vis_n:
            orig_videos.append(video)
            rec_videos.append(rec_vid)
            i += 1

    if vis_n > 0:
        plot_test_video_outputs(orig_videos, rec_videos)

    # true_labels = torch.Tensor(true_labels) #, dtype=torch.long)
    
    return latents, true_labels, losses

def plot_test_audio_outputs(orig_audios, rec_audios):

    n = len(orig_audios)
    plt.figure(figsize=(32, 9))
    rows = (n // 2) + (n % 2)

    for i in range(n):
        orig_audio, rec_audio = orig_audios[i], rec_audios[i]

        ind = i+1 
        ax = plt.subplot(rows,2,ind)
        orig_audio = orig_audio.cpu().squeeze().squeeze()
        rec_audio = rec_audio.cpu().squeeze().squeeze()

        ax.plot(orig_audio, alpha=0.6)
        ax.plot(rec_audio, alpha=0.6)

        # print('\nOriginal Audio (Top), Predicted Audio(Bottom)')
        # display(ipd.Audio(orig_audio, rate=data_samplerate))
        # display(ipd.Audio(rec_audio, rate=data_samplerate))
        # print("\n************************************************\n")

    plt.show()


def test_audio_model(ae_model, data_loader, loss_fn, device, vis_n=5):

    latents = None
    ae_model.eval()

    i = 0
    orig_audios = []
    rec_audios = []
    losses = []

    for audio in data_loader:

        audio = audio.to(device)

        with torch.no_grad():
            pred_audio = ae_model(audio)
            latents = torch.cat([latents, ae_model.latent_space.unsqueeze(0)]) if latents is not None else ae_model.latent_space.unsqueeze(0)

        if i < vis_n:
            orig_audios.append(audio)
            rec_audios.append(pred_audio)
            i += 1

        rec_loss = loss_fn(pred_audio, audio)
        losses.append(rec_loss.detach().cpu().numpy())

    if vis_n > 0:
        plot_test_audio_outputs(orig_audios, rec_audios)

    return latents, losses

# Train Model
def val_test_model(audio_ae, video_ae, classifier, device, 
                audio_dataloader, video_dataloader, 
                audio_loss_fn, video_loss_fn, clf_loss_fn, vis_n=5):

    aud_latents, aud_losses = test_audio_model(audio_ae, audio_dataloader, audio_loss_fn, device, vis_n)
    vid_latents, true_labels, vid_losses = test_video_model(video_ae, video_dataloader, video_loss_fn, device, vis_n)

    classifier.eval()

    test_classifier_loss = []

    iter_num = 1

    preds = None

    for i in range(len(aud_latents)):

        audio_latent = aud_latents[i]
        video_latent = vid_latents[i]
        emo_labels = true_labels[i]

        audio_latent = audio_latent.to(device)
        video_latent = video_latent.to(device)
        emo_labels = emo_labels.to(device)

        av_latent = concat(audio_latent, video_latent)

        pred_class = classifier(av_latent)
        preds = torch.cat([preds, pred_class.unsqueeze(0)]) if preds is not None else pred_class.unsqueeze(0)

        clf_loss = None
        # print("type pred-class", type(pred_class), pred_class)
        # print("type emo_labels", type(emo_labels), emo_labels)
        # clf_loss = clf_loss_fn(pred_class.to(device), emo_labels.to(device))
        # test_classifier_loss.append(clf_loss.detach().cpu().numpy())


        if iter_num % 10 == 0:
            print("Batch:", iter_num, "\t test audio loss {} \t test video loss {} \t test clf loss {}".format(aud_losses[i], vid_losses[i], clf_loss))
        
        iter_num += 1


    # print(preds, true_labels)
    test_acc = calc_accuracy(preds, true_labels) #if epoch_num > no_test_clf_epochs else None
    losses = [np.mean(aud_losses), np.mean(vid_losses), np.mean(test_classifier_loss)]

    print("Test Accuracy:", test_acc)
    return losses, test_acc

"""### Train Model"""

# Train Model
def train_epoch(audio_ae, video_ae, classifier, device, 
                audio_dataloader, video_dataloader, 
                audio_loss_fn, audio_optimiser,
                video_loss_fn, video_optimiser,
                clf_loss_fn, clf_optimiser,
                epoch_num, no_train_clf_epochs):

    audio_ae.train()
    video_ae.train()
    classifier.train()

    train_audio_ae_loss = []
    train_video_ae_loss = []
    train_classifier_loss = []

    iter_num = 1

    audio_data_iter = iter(audio_dataloader)
    preds = None
    true_labels = None

    for video, emo_labels in video_dataloader:

        # Zero out all gradients
        audio_optimiser.zero_grad()
        video_optimiser.zero_grad()

        # Run Audio Autoencoder Epoch
        audio = next(audio_data_iter)
        audio = audio.to(device)

        pred_audio = audio_ae(audio)
        audio_latent = audio_ae.latent_space
        audio_rec_loss = audio_loss_fn(pred_audio, audio)

        audio_rec_loss.backward(retain_graph=(True if epoch_num > no_train_clf_epochs else False))


        # Run Video Autoencoder Epoch
        video = video.to(device)

        pred_video = video_ae(video)
        video_latent = video_ae.latent_space
        video_rec_loss = video_loss_fn(pred_video, video)

        video_rec_loss.backward(retain_graph=(True if epoch_num > no_train_clf_epochs else False))
        clf_loss = None

        if epoch_num > no_train_clf_epochs:
            # Run Classifier Epoch
            clf_optimiser.zero_grad()

            audio_latent = audio_latent.to(device)
            video_latent = video_latent.to(device)

            av_latent = concat(audio_latent, video_latent)

            pred_class = classifier(av_latent)

            clf_loss = clf_loss_fn(pred_class.to(device), emo_labels.to(device))

            preds = torch.cat([preds, pred_class.unsqueeze(0)]) if preds is not None else pred_class.unsqueeze(0)
            true_labels = torch.cat([true_labels, emo_labels], dim=0) if true_labels is not None else emo_labels

            clf_loss.backward()
            clf_optimiser.step()
            train_classifier_loss.append(clf_loss.detach().cpu().numpy())

        audio_optimiser.step()
        video_optimiser.step()


        if iter_num % 10 == 0:
            print("Batch:", iter_num, "\t train audio loss {} \t train video loss {} \t train clf loss {}".format(audio_rec_loss, video_rec_loss, clf_loss))
        
        iter_num += 1
        
        train_audio_ae_loss.append(audio_rec_loss.detach().cpu().numpy())
        train_video_ae_loss.append(video_rec_loss.detach().cpu().numpy())


    # print(preds, true_labels)
    train_acc = calc_accuracy(preds, true_labels) if epoch_num > no_train_clf_epochs else None
    losses = [np.mean(train_audio_ae_loss), np.mean(train_video_ae_loss), (np.mean(train_classifier_loss) if epoch_num > no_train_clf_epochs else None)]

    return losses, train_acc

def save_model(model, optim, epoch, train_loss, val_loss, path, train_acc=None, val_acc=None):
    file_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'train_loss': train_loss,
                'val_loss' : val_loss
                }
    if train_acc:
        file_dict['train_acc'] = train_acc
        file_dict['val_acc'] = val_acc

    torch.save(file_dict, path)

# models_path = "/content/drive/MyDrive/Autoencoder_proj"

# checkpoint = torch.load(os.path.join(models_path,"classifier_model.pt"))
# classifier_model.load_state_dict(checkpoint['model_state_dict'])
# clf_optim.load_state_dict(checkpoint['optimizer_state_dict'])
# train_acc = checkpoint['train_acc']
# val_accuracies = checkpoint['val_acc']

# checkpoint = torch.load(os.path.join(models_path,"video_model.pt"))
# video_model.load_state_dict(checkpoint['model_state_dict'])
# video_optim.load_state_dict(checkpoint['optimizer_state_dict'])

# checkpoint = torch.load(os.path.join(models_path,"audio_model.pt"))
# audio_model.load_state_dict(checkpoint['model_state_dict'])
# audio_optim.load_state_dict(checkpoint['optimizer_state_dict'])

def train(audio_model, video_model, classifier_model, device,
            train_audio_loader, train_video_loader, 
            audio_loss_func, audio_optim, 
            video_loss_func, video_optim, 
            clf_loss_func, clf_optim, models_path):

    num_epochs = 30
    train_loss = []
    train_acc = []

    val_loss = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print('\nTraining Epoch:',epoch+1)
        losses, acc = train_epoch(audio_model, video_model, classifier_model, device,
                                train_audio_loader, train_video_loader, 
                                audio_loss_func, audio_optim, 
                                video_loss_func, video_optim, 
                                clf_loss_func, clf_optim, epoch+1, 0)
        train_loss.append(losses)
        train_acc.append(acc)

        print('\n EPOCH {}/{} \t train accuracy {} \t train audio loss {} \t train video loss {} \t train clf loss {}'.format(epoch + 1, num_epochs, acc, *losses))

        val_losses, val_acc = val_test_model(audio_model, video_model, classifier_model, device,
                        val_audio_loader, val_video_loader, 
                        audio_loss_func, video_loss_func, clf_loss_func, vis_n=(5 if (((epoch+1) % 3 == 0) and ((epoch+1) < 12)) else 0))
        val_loss.append(val_losses)
        val_accuracies.append(val_acc)
        print('\n EPOCH {}/{} \t val accuracy {} \t val audio loss {} \t val video loss {} \t val clf loss {}'.format(epoch + 1, num_epochs, val_acc, *val_losses))
        pd.DataFrame(data={'Epoch':epoch,'Loss':train_loss,'Accuracy':train_acc,'Val_loss':val_loss,'Val_acc':val_accuracies}).to_csv("/content/drive/Mydrive/model/CSV/everyepoch.csv")#csvpathallepochs
        '''if (epoch+1) % 5 == 0:
            print("Saving Models")
            save_model(classifier_model, clf_optim, epoch+1, losses[2], val_losses[2],
                        os.path.join(models_path,"classifier_model_1.pt"), train_acc, val_accuracies)
            save_model(video_model, video_optim, epoch+1, losses[1], val_losses[1],
                        os.path.join(models_path,"video_model_1.pt"))
            save_model(audio_model, audio_optim, epoch+1, losses[0], val_losses[0],
                        os.path.join(models_path,"audio_model_1.pt"))'''
        if val_acc>maxx:
            print("improvement detected...")
            maxx=val_acc
            pd.DataFrame(data={'Epoch':epoch,'Loss':train_loss,'Accuracy':train_acc,'Val_loss':val_loss,'Val_acc':val_accuracies}).to_csv("/content/drive/Mydrive/best_model/CSV/best.csv")#csvpathbestmodel
            print("Saving Models")
            save_model(classifier_model, clf_optim, epoch+1, losses[2], val_losses[2],
                        os.path.join(models_path,"classifier_model_{epoch}.pt"), train_acc, val_accuracies)
            save_model(video_model, video_optim, epoch+1, losses[1], val_losses[1],
                        os.path.join(models_path,"video_model_{epoch}.pt"))
            save_model(audio_model, audio_optim, epoch+1, losses[0], val_losses[0],
                        os.path.join(models_path,"audio_model_{epoch}.pt"))


def plot_train_val_loss(train_loss, val_loss):
    train_loss_pd = pd.DataFrame(train_loss, columns=['audio', 'video', 'clf'])
    val_loss_pd = pd.DataFrame(val_loss, columns=['audio', 'video', 'clf'])


    plt.plot(train_loss_pd['audio'])
    plt.plot(list(range(1,num_epochs+1)), val_loss_pd['audio'])
    plt.plot(train_loss_pd['video'])
    plt.plot(list(range(1,num_epochs+1)), val_loss_pd['video'])
    # plt.plot(loss_pd['clf'])
    plt.legend(['train_audio', 'val_audio', 'train_video', 'val_video'])
    plt.show()

def plot_train_val_acc(train_acc, val_accuracies):

    train_acc = [acc.cpu().numpy() for acc in train_acc]
    train_acc = np.array(train_acc)

    val_acc = [acc.cpu().numpy() for acc in val_accuracies]
    val_acc = np.array(val_acc)
    # val_acc = np.concatenate((old_val, val_acc), axis=0)

    print("Mean train accuracy across epochs:",np.mean(train_acc))
    print("Max train accuracy across epochs:",np.max(train_acc))
    print()
    print("Mean val accuracy across epochs:",np.mean(val_acc))
    print("Max val accuracy across epochs:",np.max(val_acc))

    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.legend(['train_acc', 'val_acc'])
    plt.show()


if __name__ == '__main__':

    audio_dir = os.path.abspath('./data/cropped_data/audio')
    video_dir = os.path.abspath('./data/cropped_data/videos')
    models_path = os.path.abspath('./best_models') # path to save models

    aud_vid_paths, labels = dataloader.get_avpaths(audio_dir)
    # aud_vid_paths = ['exc\\F\\977', 'exc\\M\\660', 'exc\\F\\2802', 'exc\\F\\1219', 
    #                 'sad\\M\\1451', 'sad\\F\\1580', 'sad\\F\\281', 'sad\\M\\2515', 
    #                 'neu\\M\\1886', 'neu\\F\\1050', 'neu\\F\\1158', 'neu\\F\\1087', 
    #                 'ang\\M\\5317', 'ang\\M\\5907',  'ang\\M\\4935', 'ang\\M\\5441']
    # labels = ['exc', 'exc', 'exc', 'exc',
    #           'sad', 'sad', 'sad', 'sad',
    #           'neu', 'neu', 'neu', 'neu',
    #           'ang', 'ang', 'ang', 'ang']
    num_data = len(aud_vid_paths) #20

    train_ind, val_ind, test_ind = dataloader.get_train_val_test_ind(num_data, labels)

    ## Prepare Audio data
    audio_data = dataloader.init_audio_data(audio_dir, aud_vid_paths)
    train_audio_loader, val_audio_loader, test_audio_loader = dataloader.load_audio_data(audio_data, train_ind, val_ind, test_ind)

    ## Prepare Video Data
    video_data = dataloader.init_video_data(video_dir, aud_vid_paths)
    train_video_loader, val_video_loader, test_video_loader = dataloader.load_video_data(video_data, train_ind, val_ind, test_ind)


    ## Initialise audio, video and classifier models
    audio_model, audio_optim, audio_loss_func = model.init_audio_model()
    video_model, video_optim, video_loss_func = model.init_video_model()
    classifier_model, clf_optim, clf_loss_func = model.init_classifier_model()

    train(audio_model, video_model, classifier_model, device,
            train_audio_loader, train_video_loader, 
            audio_loss_func, audio_optim, 
            video_loss_func, video_optim, 
            clf_loss_func, clf_optim, models_path)
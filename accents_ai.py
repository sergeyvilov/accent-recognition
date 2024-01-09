#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa as lr
import numpy as np

import subprocess
import io
#
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.rcParams.update({'font.size': 16})

import torch
from torch import nn

from torch.nn.functional import softmax

# from scipy import stats

accents = ['German','English','Spanish','French','Russian']

class AttRnn(nn.Module):

    '''
    Attention RNN from
    De Andrade, Douglas Coimbra, et al. "A neural attention model for speech command recognition." arXiv preprint arXiv:1808.08929 (2018).
    '''

    def __init__(self, device, n_languages = 5, input_height = 20,
                 lstm_layers=2, dropout=0.25, **kwargs):

        super().__init__()

        self.device = device

        self.image_height = input_height

        self.lstm_layers = lstm_layers
        self.lstm_hout = 256

        #input size: N_batch x 1 x image_height x L

        self.conv1 = nn.Conv2d(1, 10, kernel_size=(1,5), padding='same') #conv only in time dimension
        self.bn1 = nn.BatchNorm2d(10)

        self.conv2 = nn.Conv2d(10, 1, kernel_size=(1,5), padding='same') #conv only in time dimension
        self.bn2 = nn.BatchNorm2d(1)

        #convert dimensions to N_batch x L x image_height here!

        self.rnn = nn.LSTM(self.image_height, self.lstm_hout, self.lstm_layers, bidirectional=True, batch_first=True) #Hin, Hout

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_languages)

    def forward(self, x):

        n_batch, n_features, L = x.shape

        x = torch.unsqueeze(x,1)

        h0 = torch.zeros(2*self.lstm_layers, n_batch, self.lstm_hout).to(self.device)
        c0 = torch.zeros(2*self.lstm_layers, n_batch, self.lstm_hout).to(self.device)

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = torch.permute(out, (0, 3, 2, 1)).squeeze(dim=-1) #remove last singleton dimension

        out, (hn, cn) = self.rnn(out, (h0, c0))

        ################################Attention Block################################

        xFirst = out[:,-1,:] #first element in the sequence

        query = self.fc1(xFirst)

        #dot product attention
        attScores = torch.bmm(query.unsqueeze(1), out.permute((0,2,1))) / np.sqrt(2*self.lstm_hout)

        attScores = self.softmax(attScores)

        attVector = torch.bmm(attScores, out).squeeze(dim=1)

        ##############################################################################

        out = self.fc2(attVector)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.fc4(out)

        return out

class EnsemblePredictor():

    def __init__(self, models_dir, n_models, max_duration=None):
        #load model ensemble

        self.ensemble = []

        for seed in range(1,n_models+1):
            state_dict = torch.load(f'{models_dir}/{seed}/epoch_65.pt',map_location=torch.device('cpu'))
            model = AttRnn(n_languages = 5, input_height = 13, device=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            self.ensemble.append(model)

        self.max_duration = max_duration

    def extract_mel_spect(self, path, n_mels=13, win_time=0.040, hop_time=0.020,
                    log_epsilon=1e-14, mean_=-2.55, std_=5.15, clip_at=10):
        '''
        Extract mel spectrogram
        '''
        signal, sr = lr.load(path, sr = None) #2x faster without resampling!

        img = lr.feature.melspectrogram(y=signal, sr=sr, n_fft=1024,
                                        win_length=int(win_time*sr), hop_length=int(hop_time*sr),
                                        fmin=20, fmax=8000,
                                        n_mels=n_mels)

        img = np.log(img + log_epsilon) #to normal distribution
        img = (img - mean_) / std_
        img = np.clip(img,-clip_at,clip_at)
        return img

    def load_audio(self, waveform):

        #convert to wav and downsample to 16K
        #waveform=subprocess.check_output(f"ffmpeg -y -nostdin -hide_banner -loglevel error -i {path} -ar 16000 -f wav -to 00:00:59 pipe:1", shell=True).rstrip()

        file_obj = io.BytesIO(waveform) #convert to file object for librosa

        return file_obj

    def __call__(self, path):

        f = self.load_audio(path)

        mel_spect = self.extract_mel_spect(f) #get a mel spectrogram

        mel_spect = torch.Tensor(mel_spect).unsqueeze(0) #convert to tensor for model inference

        ensemble_probs = []

        with torch.no_grad():
            for model in self.ensemble:
                model_probs = softmax(model(mel_spect),dim=1).numpy()
                ensemble_probs.append(model_probs)

        ensemble_probs = np.vstack(ensemble_probs)

        prob_per_lang = ensemble_probs.mean(0)
        prob_per_lang = prob_per_lang/prob_per_lang.sum()

        return {lang:min(int(prob*100),100) for lang,prob in zip(accents, prob_per_lang)} #dictionary with probabilities for each accent
#
#
# # In[12]:
#
#
# accent_predictor = EnsemblePredictor(models_dir, n_models=10, max_duration=15)
#
#
#
# ensemble_probs = accent_predictor(sample_audio)
#
#
#
#
#
# # In[23]:
#
#
# language_colors = {'German':"#E69F00",'English':"#56B4E9",
#                 'Spanish':"#F0E442",'French':"#D55E00",
#                 'Russian':"#009E73"}
#
#
# # In[24]:
#
#
#
#
# fig, ax = plt.subplots(dpi=300, figsize=(10,3))
#
# ax.bar(height=prob_per_lang,x=accents,color=[language_colors[x] for x in accents], alpha=0.5)
# ax.grid(axis='y')
# ax.set_axisbelow(True)
# ax.set_ylabel('accent probability')
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_yticks(np.arange(0,1.1,0.2));
# ax.set_ylim([0,1.01]);
# ax.set_yticklabels([str(x)+'%' for x in range(0,101,20)], fontsize=14);
#
#
# # In[ ]:

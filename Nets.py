import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import os
import string 
# from google.colab import drive
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import time
import matplotlib.pyplot as plt
import pandas as pd
import os
from data import *
from Data_Handling import split_vector_to_seqs, combine_seqs_to_vector
import hyper_parameters as hp

device = 'cuda' if torch.cuda.is_available() else 'cpu' # Getting the device runnig on

def save_model_with_checkpoint(model, lr, model_type='spectra', name=None):
    if model.with_drop == True:
        if name is None:
            model_path = f'/home/drors/project/Models/{model_type.lower()}_drop_{model.drop}_lr_{lr}.net'
        else:
            model_path = f'/home/drors/project/Models/{name.lower()}_{model_type.lower()}_drop_{model.drop}_lr_{lr}.net'
    else:
        if name is None:
            model_path = f'/home/drors/project/Models/{model_type.lower()}_without_lr_{lr}.net'
        else:
            model_path = f'/home/drors/project/Models/{name.lower()}_{model_type.lower()}_without_lr_{lr}.net'

    if model_type.lower() == 'spectra':
        checkpoint = {'with_drop': model.with_drop,
                      'hidden_size': model.hidden_size,
                      'n_layers': model.n_layers,
                      'drop': model.drop,
                      'wave_amount': model.wave_amount,
                      'seq_size': model.seq_size,
                      'output_amount':model.output_amount,
                      'seq_amount': model.seq_amount,
                      'state_dict': model.state_dict()}
    elif model_type.lower() == 'doppler':
        checkpoint = {'method': model.method,
                      'with_drop': model.with_drop,
                      'hidden_size': model.hidden_size,
                      'n_layers': model.n_layers,
                      'drop': model.drop,
                      'wave_amount_sample': model.wave_amount_sample,
                      'wave_amount_predict': model.wave_amount_predict,
                      'seq_size': model.seq_size,
                      'obs_amount':model.obs_amount,
                      'combine_wave_input': model.combine_wave_input,
                      'batch_size': model.batch_size,
                      'state_dict': model.state_dict()}
        
    with open(model_path, 'wb') as f:
        torch.save(checkpoint, f)
 

def load_model_from_file(file_name: str):
    with open(file_name, 'rb') as f:
        checkpoint = torch.load(f)
    
    if 'spectra' in file_name:
        
        loaded_model = SpectraNet(checkpoint['with_drop'],
                               checkpoint['hidden_size'],
                               checkpoint['n_layers'],
                               checkpoint['drop'],
                               20,
                               checkpoint['wave_amount'],
                               checkpoint['seq_size'],
                               checkpoint['output_amount'],
                               checkpoint['seq_amount'])
    elif 'doppler' in file_name:
        loaded_model = DopplerNet(checkpoint['method'],
                                  checkpoint['with_drop'],
                                  checkpoint['hidden_size'],
                                  checkpoint['n_layers'],
                                  checkpoint['drop'], 
                                  checkpoint['batch_size'],
                                  checkpoint['wave_amount_sample'],
                                  checkpoint['wave_amount_predict'],
                                  checkpoint['seq_size'],
                                  checkpoint['obs_amount'],
                                  checkpoint['combine_wave_input'])
        
    loaded_model.load_state_dict(checkpoint['state_dict'])
    return loaded_model


def load_model_from_mode(model_type: str, with_drop, drop: float, lr: float, name=None):
    try:
        if with_drop:
            if name is None:
                return load_model_from_file(f'/home/drors/project/Models/{model_type.lower()}_drop_{drop}_lr_{lr}.net')
            else:
                return load_model_from_file(f'/home/drors/project/Models/{name.lower()}_{model_type.lower()}_drop_{drop}_lr_{lr}.net')
        else:
            if name is None:
                return load_model_from_file(f'/home/drors/project/Models/{model_type.lower()}_without_lr_{lr}.net')
            else: 
                return load_model_from_file(f'/home/drors/project/Models/{name.lower()}_{model_type.lower()}_without_lr_{lr}.net')
    except FileNotFoundError as e:
        print("no such file exist, if entered with drop try and put a different drop value")


# Data Pre-Processing

def get_sizes(train, test, val):
    iterator = iter(train)
    it1 = next(iterator)
    obs, ground_truth, sys_id, times, velocities = it1
    print(obs.shape, obs.type()) # observations: Obs_amount X wave_amount
    print(ground_truth.shape, ground_truth.type()) # Ground Truth Spectras: stars_amount (2) X wave_output_amount
    print(sys_id) # System ID: int
    print(times.shape, times.type()) # Time of sample: Obs_amount
    print(velocities.shape, velocities.type()) # Velocities per sample time: stars_amount (2) X Obs_amount
    Obs_amount, wave_amount, wave_output = obs.shape[0], obs.shape[1], ground_truth.shape[1]
    return Obs_amount, wave_amount, wave_output
    
    
'''Model Class Template'''

class SpectraNet(nn.Module):
    def __init__(self, with_drop=True, hidden_size=hp.hidden_size, n_layers=hp.num_layers_spectra, drop=hp.drop_spectra, 
                 batch_size=hp.batch_size, wave_amount=hp.observation_amount, seq_size=hp.wave_sequence_length, 
                 output_amount=hp.wave_output, seq_amount=hp.wave_seq_amount):
        super(SpectraNet, self).__init__()
        self.with_drop = with_drop
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.drop = drop
        self.batch_size = batch_size
        self.wave_amount = wave_amount
        self.output_amount = output_amount
        self.seq_size = seq_size
        self.seq_amount = seq_amount
        self.hidden_layer_size = int(self.wave_amount/2)
        self.conv1 = nn.Sequential(nn.Conv2d(1,4, kernel_size = 5, stride = 2),
                                   nn.ReLU()) # 1*1*20*15000 -> 1*4*8*7498
        self.conv2 = nn.Sequential(nn.Conv2d(4,8, kernel_size = 5, stride = 2),
                                   nn.ReLU(),
                                   nn.AvgPool2d(kernel_size = (1,4))) #1*4*8*7498 -> 1*8*2*3747/2
        self.convfc = nn.Sequential(nn.Linear(14976,8000),
                                    nn.ReLU(),
                                    nn.Dropout(drop))
        self.convfc1 = nn.Linear(8000, self.output_amount)
        self.convfc2 = nn.Linear(8000 + self.output_amount, self.output_amount)
        if with_drop == True:
            self.rnn = nn.LSTM(self.seq_size, hidden_size, n_layers,
                               dropout=drop, batch_first=True, bidirectional=True)
            self.attention = nn.MultiheadAttention(seq_size, num_heads=1, dropout=drop)
            self.dropout = nn.Dropout(drop)
            '''should take the whole RNN outpur vector (concatanate the outputs from each lstm-cell-forward_pass) and feed it to one FC'''
            self.fc1 = nn.Sequential(nn.Linear(self.wave_amount, self.hidden_layer_size),
                                    nn.ReLU(),
                                    self.dropout,
                                    nn.Linear(self.hidden_layer_size, self.output_amount))
            self.fc2 = nn.Linear(self.output_amount, self.output_amount)

        else:
            self.rnn = nn.LSTM(self.seq_size, hidden_size, n_layers,
                                   batch_first=True, bidirectional=True)
            self.attention = nn.MultiheadAttention(seq_size, num_heads=1)
            self.fc1 = nn.Sequential(nn.Linear(self.wave_amount, self.hidden_layer_size),
                                    nn.SiLU(),
                                    nn.Linear(self.hidden_layer_size, self.output_amount))
            self.fc2 = nn.Linear(self.output_amount, self.output_amount)
    
        #ADD THE ATTENTION METHOD.
        
    def forward(self, spectra, hidden, method: int = 3):
        '''     
        1. sum the losses of each spectra and then backprop - I think it will backprop in the net twice
        (for each spectra)
        2. train the first layers until spectra1 and then only the last FC on the spectra2 with all the
        others params grad's are off
        '''
        if self.with_drop: 
            spectra = self.dropout(spectra)
        spectra, hidden = self.rnn(spectra, hidden)
        if self.with_drop:
            spectra = self.dropout(spectra)
        '''To give the attention input we need to view it in: seq_amount X batch_size X seq_length'''
#         ##spectra = spectra.reshape(self.seq_amount, self.batch_size, self.seq_size)
#         ##spectra, _ = self.attention(spectra, spectra, spectra, need_weights=False)
#         ##spectra = spectra.reshape(self.batch_size, self.seq_amount, self.seq_size)
        spectra = combine_seqs_to_vector(spectra.contiguous(), self.wave_amount)
        if method == 1:
            spectra1 = self.fc1(spectra)
            spectra2 = self.fc2(spectra1.detach().data)
            final_spectra1 = torch.mean(spectra1, dim=0) #the batch dim - might need to also resize it to get rid of this dim
            final_spectra2 = torch.mean(spectra2, dim=0) #the batch dim - might need to also resize it to get rid of this dim
            return final_spectra1, final_spectra2, hidden
        elif method == 2:
            spectra1 = self.fc1(spectra)
            final_spectra1 = torch.mean(spectra1, dim=0) #the batch dim - might need to also resize it to get rid of this dim
            return final_spectra1, hidden
        elif method == 3:
            spectra = spectra.view(1,1,spectra.shape[0],spectra.shape[1])
            spectra = self.conv1(spectra)
            spectra = self.conv2(spectra)
            spectra = spectra.view(spectra.shape[0],-1)
            spectra = self.convfc(spectra)
            final_spectra1 = self.convfc1(spectra)
            for_second = torch.cat((spectra, final_spectra1), 1)
            final_spectra2 = self.convfc2(for_second)
            return final_spectra1, final_spectra2, hidden
    def forward_method_2(self, spectra1):
        '''
        need to be called only after we finished training the first part
        '''
        return torch.mean(self.fc2(spectra1.detach().view(1,-1)), dim=0) #the batch dim - might need to also resize it to get rid of this dim
    def init_hidden(self):
        weight = next(self.parameters()).data
        if device == 'cuda':
            # 2* because we are bidirectional
            hidden = (weight.new(2*self.n_layers, self.batch_size, self.hidden_size).zero_().cuda(),
                      weight.new(2*self.n_layers, self.batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(2*self.n_layers, self.batch_size, self.hidden_size).zero_(),
                      (weight.new(2*self.n_layers, self.batch_size, self.hidden_size).zero_()))
        
        return hidden

    

# The Speed - Dopler Network
'''This network is meant to calculate the speed of the stars at the sampling point according to the spectra measured
in said time points.

we can either give the net a vector of all the wave samples at a specific time and ask for a single result - single speed of the
star at the time of measurments *(one-directional RNN)*. OR: give it the resulted spectra from the spectra network and output a
full vector of speeds *(bidirectional RNN)*'''

class DopplerNet(nn.Module):
    def __init__(self, method=2.5, with_drop=True, hidden_size=hp.hidden_size_doppler,
                 n_layers=hp.num_layers_doppler, drop=hp.drop_doppler, batch_size=hp.batch_size,
                 wave_amount_sample=hp.observation_amount, wave_amount_predict=hp.wave_output, 
                 seq_size=hp.doppler_sequence_length25, obs_amount=hp.observation_amount,
                 combine_wave_input=hp.concat_value_output):
        '''
        wave_amount: the size of the amount of inputs in the RNN
        
        we need to create 3 different dopppler nets INSTANCES (according to the 3 different methods - we can make the 
        forward method change according to the train method)
        Also add the correct variables for the init method 
        '''
        super(DopplerNet, self).__init__()
        self.with_drop = with_drop
        self.method = method
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.drop = drop
        self.batch_size = 20 #batch_size
        self.seq_size = seq_size
        self.wave_amount_sample = wave_amount_sample #7500
        self.wave_amount_predict = wave_amount_predict #1700
        self.combine_wave_input = combine_wave_input #9650
        self.obs_amount = obs_amount # the amount of observation - velocities.

        if self.with_drop: 
            if self.method == 1 or self.method == 2:
                #Training method 1&2
                # We get a spectra of a single star and we need to return a vector of velocities.
                self.rnn_doppler = nn.LSTM(seq_size, hidden_size, n_layers, 
                                           dropout=drop, batch_first=True, bidirectional=True)
                self.dropout = nn.Dropout(drop)
                self.fc = nn.Sequential(nn.Linear(self.new_wave_amount, 512),#we get the spectra and return a vector of 20 velocities
                                        nn.ReLU(),
                                        self.dropout,
                                        nn.Linear(512, self.obs_amount))
                if self.wave_amount_predict % self.seq_size == 0:
                    self.seq_amount = self.wave_amount_predict // self.seq_size
                else:
                    raise ValueError("seq_size deosn't devide the input length")
            elif self.method == 2.5:
                #Training method 2.5
                self.rnn_doppler = nn.LSTM(seq_size, hidden_size, n_layers, 
                                           dropout=drop, batch_first=True, bidirectional=False)
                #We take the output of the last cell and use it
                self.dropout = nn.Dropout(drop)
                self.fc = nn.Sequential(nn.Linear(self.seq_size * 2, 64 * 2),#we get the spectra and return a vector of 20 velocities
                                        nn.ReLU(),
                                        self.dropout,
                                        nn.Linear(64 * 2, 1 * 2))
                if self.combine_wave_input % self.seq_size == 0:
                    self.seq_amount = self.combine_wave_input // self.seq_size
                else:
                    raise ValueError("seq_size deosn't devide the input length")
                
        else:
            if self.method == 1 or self.method == 2:
                #Training method 1&2
                # We get a spectra of a single star and we need to return a vector of velocities.
                self.rnn_doppler = nn.LSTM(seq_size, hidden_size, n_layers, 
                                           batch_first=True, bidirectional=True)
                self.fc = nn.Sequential(nn.Linear(self.new_wave_amount, 512),#we get the spectra and return a vector of 20 velocities
                                        nn.ReLU(),
                                        nn.Linear(512, self.obs_amount))
                if self.wave_amount_predict % self.seq_size == 0:
                    self.seq_amount = self.wave_amount_predict // self.seq_size
                else:
                    raise ValueError("seq_size deosn't devide the input length")
            elif self.method == 2.5:
                #Training method 2.5
                self.rnn_doppler = nn.LSTM(seq_size, hidden_size, n_layers, 
                                           batch_first=True, bidirectional=False)
                #We take the output of the last cell and use it
                self.fc = nn.Sequential(nn.Linear(800, 64 * 2),#we get the spectra and return a vector of 20 velocities
                                        nn.ReLU(),
                                        nn.Linear(64, 1 * 2)) 
                if self.combine_wave_input % self.seq_size == 0:
                    self.seq_amount = self.combine_wave_input // self.seq_size
                else:
                    raise ValueError("seq_size deosn't devide the input length")
            
      
    def forward(self, waves, hidden):
        '''
        WE SPLIT THE INPUTS TO SEQUENCES BEFORE THE FORWARD FUNCTION
        
        waves: the the star predicted spectra tensor (batch_size X seq_amount X seq_length) - we "duplicate"
        it so its has a batch size of 20 and not 1 in methhod 2.5, in 1&2 the batch size is 2 (only one system per forward)
        hidden: the hidden state with its regular size
        
        NOTICE: if we use method 2.5 waves is a concatenation of the observations and the predicted spectra
        (splitted to seqs and with a batch_size = observation_amount) 
        '''
        velocity = None #Currently equal to none
        
        if self.with_drop:
            velocity = self.dropout(waves)
        velocity, hidden = self.rnn_doppler(velocity, hidden)
        if self.with_drop:
            velocity = self.dropout(velocity)
        if self.method == 2.5:
            velocity = velocity[:,-1,:] #taking the last output sequence = most affected by the predicted spectra of the star
            velocity = velocity.view(2*self.batch_size, self.seq_size)
        elif self.method == 1 or self.method == 2:
            velocity = combine_seqs_to_vector(velocity, velocity.shape[1]*velocity.shape[2]) # 20 X seq_amount*seq_length
        velocity = torch.cat((velocity[:20,:], velocity[20:,:]), dim = 1)
        velocity = self.fc(velocity)
        return velocity, hidden
    
              
    def init_hidden(self):
        weight = next(self.parameters()).data
        if device == 'cuda':
            hidden = (weight.new(self.n_layers, 2*self.batch_size, self.hidden_size).zero_().cuda(),
                weight.new(self.n_layers, 2*self.batch_size, self.hidden_size).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, 2*self.batch_size, self.hidden_size).zero_(),
                    weight.new(self.n_layers, 2*self.batch_size, self.hidden_size).zero_())
        return hidden
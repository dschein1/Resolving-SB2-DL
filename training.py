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
from Data_Handling import *
import hyper_parameters as hp
from Nets import device
import Nets


def calc_loss(loss_function,gen_a,gen_b,real_a,real_b):
    loss1a = loss_function(gen_a,real_a)
    loss2a = loss_function(gen_b,real_b)
    loss1b = loss_function(gen_a,real_b)
    loss2b = loss_function(gen_b,real_a)
    if loss1a + loss2a < loss1b + loss2b:
        return loss1a + loss2a, 0
    else:
        return loss1b + loss2b, 1
#     return min(,loss1b + loss2b)

def normalize_tensor(value):
    value = value - torch.mean(value)
    value = value / torch.std(value)
    return value

''' Spectra Training '''

def train_spectra1(model: Nets.SpectraNet, optimizer, lr_scheduler, train_loader, val_loader, epochs:int = 20, clip:int = 5):
    loss_function = nn.MSELoss(reduction = 'mean')
    model.train()
    seq_size, seq_amount, wave_amount = model.seq_size, model.seq_amount, model.wave_amount
    train_loss_list = []
    val_loss_list = []
    train_amount = len(train_loader)
    val_amount = len(val_loader)
    
    for e in range(epochs):
        total_epoch_loss_train = 0
        total_epoch_loss_val = 0
        loss1 = 0
        loss2 = 0
        hidden = model.init_hidden()
        model.train()
        for obs, specs, _, _, _ in train_loader:
            hidden = tuple([each.data for each in hidden]) 
            obs, specs = split_vector_to_seqs(obs, seq_size, seq_amount, wave_amount).to(device), specs.to(device)
            spectra1, spectra2, hidden = model(obs, hidden, method=3)
            optimizer.zero_grad()
            #spectra1 = normalize_tensor(spectra1)
            #spectra2 = normalize_tensor(spectra1)
            loss, _ = calc_loss(loss_function, spectra1,spectra2,specs[0,:].view(1, -1),specs[1,:].view(1, -1))
            #loss1 = loss_function(spectra1.view(1,-1), normalize_tensor(specs[0,:].view(1, -1)))
            #loss1 = loss_function(spectra1.view(1,-1), specs[0,:].view(1, -1))
            #loss1.backward(retain_graph = True)
            #loss2 = loss_function(spectra2.view(1,-1), normalize_tensor(specs[1,:].view(1, -1)))
            #loss2 = loss_function(spectra2.view(1,-1),specs[1,:].view(1, -1))
            #loss2.backward(retain_graph = True)
            loss.backward()
            #total_epoch_loss_train += loss1.item() + loss2.item()
            total_epoch_loss_train += loss.item()
            #nn.utils.clip_grad_norm_(model.parameters(), clip) 
            optimizer.step()

            
        val_hidden = model.init_hidden()
        model.eval()
#         print("VALIDATION STARTS")
        with torch.no_grad():
            for obs, specs, _, _, _ in val_loader:
                val_hidden = tuple([each.data for each in val_hidden]) 
                obs, specs = split_vector_to_seqs(obs, seq_size, seq_amount, wave_amount).to(device), specs.to(device)
                spectra1, spectra2, val_hidden = model(obs, val_hidden, method=3)
#                 print(spectra1.shape, spectra1.view(1,-1).shape, specs[0,:].shape, specs[0,:].view(1,-1).shape)
                #loss1 = loss_function(normalize_tensor(spectra1).view(1,-1), normalize_tensor(specs[0,:]).view(1,-1))
                #loss1 = loss_function(spectra1.view(1,-1), specs[0,:].view(1,-1))
                #loss2 = loss_function(normalize_tensor(spectra2).view(1,-1), normalize_tensor(specs[1,:]).view(1,-1))
                #loss2 = loss_function(spectra2.view(1,-1), specs[1,:].view(1,-1))
                loss, _ = calc_loss(loss_function, spectra1,spectra2,specs[0,:].view(1, -1),specs[1,:].view(1, -1))
                #total_epoch_loss_val += loss1.item() + loss2.item()
                total_epoch_loss_val += loss.item()
        
        train_loss_list.append(total_epoch_loss_train/train_amount)
        val_loss_list.append(total_epoch_loss_val/val_amount) 
        print("Epoch: {}/{}...".format(e+1, epochs),
            "Train Loss: {:.4f}...".format(train_loss_list[e]),
            "Validation Loss: {:.4f}...".format(val_loss_list[e]),
             "LR: {}".format(lr_scheduler.get_last_lr()))
        lr_scheduler.step()
    return model, train_loss_list, val_loss_list

def train_spectra2(model, optimizer, train_loader, val_loader, epochs:int = 20, clip:int = 5):
    loss_function = nn.MSELoss(reduction = 'sum')
    model.train()
    seq_size, seq_amount, wave_amount = model.seq_size, model.seq_amount, model.wave_amount
    '''Training the First layers (the 1st spectra)'''
    train_loss_list1 = []
    val_loss_list1 = []
    train_loss_list2 = []
    val_loss_list2 = []
    train_amount = len(train_loader)
    val_amount = len(val_loader)
    for e in range(epochs):
        total_epoch_loss_train = 0
        total_epoch_loss_val = 0
        loss = 0
        hidden = model.init_hidden()
        model.train()
        for obs, specs, _, _, _ in train_loader:
            hidden = tuple([each.data for each in hidden]) 
            obs, specs = split_vector_to_seqs(obs, seq_size, seq_amount, wave_amount).to(device), specs.to(device)
            spectra1, hidden = model(obs, hidden, method=2)
            
            optimizer.zero_grad()
            loss = loss_function(spectra1.view(1,-1), specs[0,:].view(1,-1))
            loss.backward()
            total_epoch_loss_train += loss.item()
            
            nn.utils.clip_grad_norm_(model.parameters(), clip) 
            optimizer.step()
            
        val_hidden = model.init_hidden()
        model.eval()
        with torch.no_grad():
            for obs, specs, _, _, _ in val_loader:
                hidden = tuple([each.data for each in hidden]) 
                obs, specs = split_vector_to_seqs(obs, seq_size, seq_amount, wave_amount).to(device), specs.to(device)
                spectra1, hidden = model(obs, hidden, method=2)

                loss = loss_function(spectra1.view(1,-1), specs[0,:].view(1,-1))
                total_epoch_loss_train += loss.item()
        
        train_loss_list1.append(total_epoch_loss_train/train_amount)
        val_loss_list1.append(total_epoch_loss_val/val_amount) 
        print("Epoch: {}/{}...".format(e+1, epochs),
            "Train Loss: {:.4f}...".format(train_loss_list1[e]),
            "Validation Loss: {:.4f}...".format(val_loss_list1[e]))
    print()
    print('End of Spectra1 Training')
    print()
    
    '''Training the Last layer (the 2nd spectra)'''
    
    model.eval() #eval because we wan't the dropout to not drop because we don't backprop on layers connected to it
    req_grad = False
    for param in model.parameters():
        if param.size() == torch.Size([1700, 1700]): # the params from here are the last layer and its bias
            req_grad = True
        param.requires_grad = req_grad
        
    for e in range(epochs//2):
        total_epoch_loss_train = 0
        total_epoch_loss_val = 0
        loss = 0
        hidden = model.init_hidden()
        for obs, specs, _, _, _ in train_loader:
            hidden = tuple([each.data for each in hidden]) 
            obs, specs = split_vector_to_seqs(obs, seq_size, seq_amount, wave_amount).to(device), specs.to(device)
            spectra1, hidden = model(obs, hidden, method=2)
            spectra2 = model.forward_method_2(spectra1.data)
#             print(spectra2.shape)
            optimizer.zero_grad()
            loss = loss_function(spectra2.view(1,-1), specs[1,:].view(1,-1))
            loss.backward()
            total_epoch_loss_train += loss.item()
            
            nn.utils.clip_grad_norm_(model.parameters(), clip) 
            optimizer.step()
            
        val_hidden = model.init_hidden()
        model.eval()
        with torch.no_grad():
            for obs, specs, _, _, _ in val_loader:
                hidden = tuple([each.data for each in hidden]) 
                obs, specs = split_vector_to_seqs(obs, seq_size, seq_amount, wave_amount).to(device), specs.to(device)
                spectra1, hidden = model(obs, hidden, method=2)
                spectra2 = model.forward_method_2(spectra1.data)

                loss = loss_function(spectra2.view(1,-1), specs[1,:].view(1,-1))
                total_epoch_loss_train += loss.item()
        
        train_loss_list2.append(total_epoch_loss_train/train_amount)
        val_loss_list2.append(total_epoch_loss_val/val_amount) 
        print("Epoch: {}/{}...".format(e+1, epochs),
            "Train Loss: {:.4f}...".format(train_loss_list2[e]),
            "Validation Loss: {:.4f}...".format(val_loss_list2[e]))
    print()
    print('End of Spectra2 Training')
    print()
    return model, train_loss_list1, val_loss_list1

''' The Training - DopplerNet
    we have 2.5 training methods: '''

## 1. end-to-end
# this style of training, train both the spectra and the speed net at the same time with backprop going on both for each input and 
# output - because we use a lot of data and big RNN it might cause some gradient vanishings.

## 2. Auto-Encoder style training
# we first train the spectra net and achieve a good enough loss value, and then we train the doppler net with the spectra results.

### 2.5. Auto-Encoder + time sampled spectra
# we first train the spectra net and achieve a good enough loss value, and then we train the doppler net with the data of both the 
# spectra sampling of the whole system at a time point + the spectra we got from our net (either concatanating or doing
# some other operation).
def train_doppler_together25(spectra_net, doppler_net, optimizer, scheduler, train_loader, val_loader,
                             epochs:int = 20 ,clip:int = 2):
    loss_function = nn.MSELoss(reduction = 'mean')
    doppler_net.train()
    spec_seq_size, spec_seq_amount, spec_wave_amount = spectra_net.seq_size, spectra_net.seq_amount, spectra_net.wave_amount
    input_length, seq_size, seq_amount = doppler_net.combine_wave_input, doppler_net.seq_size, doppler_net.seq_amount
    spectra_net.eval()
    
    for param in spectra_net.parameters():
        param.requires_grad = False      
    
    train_loss_list = []
    val_loss_list = []
    for e in range(epochs):
        total_epoch_loss_train = 0
        total_epoch_loss_val = 0
        loss = 0
        spectra_hidden = spectra_net.init_hidden()
        doppler_hidden = doppler_net.init_hidden()
        
        doppler_net.train()
        for observ, specs, _, times, velocities in train_loader:
            spectra_hidden = tuple([each.data for each in spectra_hidden])
            doppler_hidden = tuple([each.data for each in doppler_hidden])
            
            obs, specs = split_vector_to_seqs(observ, spec_seq_size, spec_seq_amount, spec_wave_amount).to(device), specs.to(device)
            spec_out = spectra_net(obs, spectra_hidden, method=3)
            spectra_hidden = spec_out[2]
            _, direction = calc_loss(loss_function, spec_out[0], spec_out[1], specs[0,:].view(1, -1), specs[1,:].view(1, -1))

            doppler_input = torch.cat((observ.to(device).repeat(2,1),torch.cat((spec_out[direction][0].repeat(20,1), spec_out[1-direction][0].repeat(20,1)),0)),1)
            
            doppler_input = split_vector_to_seqs(doppler_input, seq_size, seq_amount, input_length).to(device)

            optimizer.zero_grad()
            output, doppler_hidden = doppler_net(doppler_input, doppler_hidden)
            #output = output.view(2,20)
            output = torch.transpose(output,0,1)
            velocities = velocities.to(device)
            '''HERE WE DO NEED THAT THE 1st VELOCITY WOULD CORRESPOND TO THE 1st OUTPUT AND VISE VERSA'''
#             print(output[0].shape, velocities[direction].shape, output[1].shape, velocities[1-direction].shape)
            loss = loss_function(output[0], velocities[direction]) + loss_function(output[1], velocities[1-direction])
#             loss = calc_loss(loss_function, output[0,:],output[1,:],velocities[0,:], velocities[1,:])
            #loss = loss_function(output, velocities.to(device))
            loss.backward()
            total_epoch_loss_train += loss.item()
            
#             nn.utils.clip_grad_norm_(doppler_net.parameters(), clip) 
            optimizer.step()
            
        doppler_val_hidden = doppler_net.init_hidden()
        spectra_val_hidden = spectra_net.init_hidden()
        doppler_net.eval()
#         print("VALIDATION STARTS")
        with torch.no_grad():
            for observ, specs, _, times, velocities in val_loader:
                spectra_val_hidden = tuple([each.data for each in spectra_val_hidden])
                doppler_val_hidden = tuple([each.data for each in doppler_val_hidden])

                obs, specs = split_vector_to_seqs(observ, spec_seq_size, spec_seq_amount, spec_wave_amount).to(device), specs.to(device)
                spec_out = spectra_net(obs, spectra_hidden, method=3)
                spectra_val_hidden = spec_out[2]
                _, direction = calc_loss(loss_function, spec_out[0], spec_out[1], specs[0,:].view(1, -1), specs[1,:].view(1, -1))

                doppler_input = torch.cat((observ.to(device).repeat(2,1),torch.cat((spec_out[direction][0].repeat(20,1), spec_out[1-direction][0].repeat(20,1)),0)),1)

                doppler_input = split_vector_to_seqs(doppler_input, seq_size, seq_amount, input_length).to(device)
                output, doppler_val_hidden = doppler_net(doppler_input, doppler_hidden)
                #output = output.view(2,20)
                output = torch.transpose(output,0,1)
                velocities = velocities.to(device)
                '''HERE WE DO NEED THAT THE 1st VELOCITY WOULD CORRESPOND TO THE 1st OUTPUT AND VISE VERSA'''
                loss = loss_function(output[0], velocities[direction]) + loss_function(output[1], velocities[1-direction])
                total_epoch_loss_val += loss.item()
#               loss = calc_loss(loss_function, output[0,:],output[1,:],velocities[0,:], velocities[1,:])
#               loss = loss_function(output, velocities.to(device))
        
        train_loss_list.append(total_epoch_loss_train/len(train_loader))
        val_loss_list.append(total_epoch_loss_val/len(val_loader)) 
        print("Epoch: {}/{}...".format(e+1, epochs),
            "Train Loss: {:.4f}...".format(train_loss_list[e]),
            "Validation Loss: {:.4f}...".format(val_loss_list[e]),
             "LR: {}".format(scheduler.get_last_lr()))
        scheduler.step()
    return doppler_net, train_loss_list, val_loss_list

def train_doppler_separate25(doppler_net, optimizer, scheduler, train_loader, val_loader, epochs:int = 20 ,clip:int = 2):
    loss_function = nn.MSELoss(reduction = 'mean')
    doppler_net.train()
    input_length, seq_size, seq_amount = doppler_net.combine_wave_input, doppler_net.seq_size, doppler_net.seq_amount
    
    train_loss_list = []
    val_loss_list = []
    for e in range(epochs):
        total_epoch_loss_train = 0
        total_epoch_loss_val = 0
        loss = 0
        doppler_hidden = doppler_net.init_hidden()
        
        doppler_net.train()
        for obs, specs, _, times, velocities in train_loader:
            doppler_hidden = tuple([each.data for each in doppler_hidden])

            doppler_input = torch.cat((obs.repeat(2,1),torch.cat((specs[0,:].repeat(20,1), specs[1,:].repeat(20,1)),0)),1)
            
            doppler_input = split_vector_to_seqs(doppler_input, seq_size, seq_amount, input_length).to(device)

            optimizer.zero_grad()
            output, doppler_hidden = doppler_net(doppler_input, doppler_hidden)
            #output = output.view(2,20)
            output = torch.transpose(output,0,1)
            velocities = velocities.to(device)
            '''HERE WE DO NEED THAT THE 1st VELOCITY WOULD CORRESPOND TO THE 1st OUTPUT AND VISE VERSA'''
            loss = loss_function(output[0,:], velocities[0,:]) + loss_function(output[1,:], velocities[1,:])
#             loss = calc_loss(loss_function, output[0,:],output[1,:],velocities[0,:], velocities[1,:])
            #loss = loss_function(output, velocities.to(device))
            loss.backward()
            total_epoch_loss_train += loss.item()
            
#             nn.utils.clip_grad_norm_(doppler_net.parameters(), clip) 
            optimizer.step()
            
        doppler_val_hidden = doppler_net.init_hidden()
        doppler_net.eval()
#         print("VALIDATION STARTS")
        with torch.no_grad():
            for obs, specs, _, times, velocities in val_loader:
                doppler_val_hidden = tuple([each.data for each in doppler_val_hidden]) 
                doppler_input = torch.cat((obs.repeat(2,1),
                                      torch.cat((specs[0,:].repeat(20,1), specs[1,:].repeat(20,1)), 0)), 1)
#             print(f'spectra1 size: {spectra1.size()}, specs: {specs[:, 0].size()}')
                output, doppler_val_hidden = doppler_net(split_vector_to_seqs(doppler_input, seq_size, seq_amount,
                                                                          input_length).to(device), doppler_val_hidden)
                #output = output.view(2,20)
                output = torch.transpose(output,0,1)

                velocities = velocities.to(device)
                '''HERE WE DO NEED THAT THE 1st VELOCITY WOULD CORRESPOND TO THE 1st OUTPUT AND VISE VERSA'''
#                 loss = loss_function(output[0,:], velocities[0,:]) + loss_function(output[1,:], velocities[1,:])
                loss, _ = calc_loss(loss_function, output[0,:], output[1,:] ,velocities[0,:], velocities[1,:])
                #loss = loss_function(output, velocities.to(device))
                total_epoch_loss_val += loss.item()
        
        train_loss_list.append(total_epoch_loss_train/len(train_loader))
        val_loss_list.append(total_epoch_loss_val/len(val_loader)) 
        print("Epoch: {}/{}...".format(e+1, epochs),
            "Train Loss: {:.4f}...".format(train_loss_list[e]),
            "Validation Loss: {:.4f}...".format(val_loss_list[e]),
             "LR: {}".format(scheduler.get_last_lr()))
        scheduler.step()
    return doppler_net, train_loss_list, val_loss_list

def check_spectra_model(model: Nets.SpectraNet, data, plot_amount:int = 0, standard_vector = None):
    loss_function = nn.MSELoss(reduction = 'mean')
    model.eval()
    seq_size, seq_amount, wave_amount = model.seq_size, model.seq_amount, model.wave_amount
    sample_amount = len(data)
    
    for param in model.parameters():
        param.requires_grad = False  
    
    running_loss = 0
    with torch.no_grad():
        for i, (obs, specs, c, _, _) in enumerate(data):
            hidden = model.init_hidden() 
            obs, specs = split_vector_to_seqs(obs, seq_size, seq_amount, wave_amount).to(device), specs.to(device)
            spectra1, spectra2, hidden = model(obs, hidden, method=3)
            loss, _ = calc_loss(loss_function, spectra1, spectra2, specs[0,:].view(1, -1),specs[1,:].view(1, -1))
            running_loss += loss.item()
            
            if i < plot_amount and standard_vector is not None:
                display_graphs(loss_function, spectra1.view(-1), spectra2.view(-1), specs[0,:], specs[1,:], standard_vector, c)
        avg_loss = running_loss/len(data)
        print("Loss: {:.4f}...".format(avg_loss))
    return avg_loss


def check_doppler_model(model: Nets.DopplerNet, data, plot_amount:int = 0):
    
    '''NEED TO UPDATE THE USE OF DISPLAY GRAPHS TO FIT THE TIME VECTOR'''
    
    loss_function = nn.MSELoss(reduction = 'mean')
    model.eval()
    sample_amount = len(data)
    input_length, seq_size, seq_amount = model.combine_wave_input, model.seq_size, model.seq_amount    
    for param in model.parameters():
        param.requires_grad = False  
    running_loss = 0
    with torch.no_grad():
        for obs, specs, _, times, velocities in data:
            doppler_val_hidden = model.init_hidden()
            doppler_input = torch.cat((obs.repeat(2,1),
                                  torch.cat((specs[0,:].repeat(20,1), specs[1,:].repeat(20,1)), 0)), 1)
#             print(f'spectra1 size: {spectra1.size()}, specs: {specs[:, 0].size()}')
            output, doppler_val_hidden = model(split_vector_to_seqs(doppler_input, seq_size, seq_amount,
                                                                      input_length).to(device), doppler_val_hidden)
            #output = output.view(2,20)
            output = torch.transpose(output,0,1)

            velocities = velocities.to(device)
#             '''HERE WE DO NEED THAT THE 1st VELOCITY WOULD CORRESPOND TO THE 1st OUTPUT AND VISE VERSA'''
#             loss = loss_function(output[0,:], velocities[0,:]) + loss_function(output[1,:], velocities[1,:])
            loss, direction = calc_loss(loss_function, output[0,:], output[1,:], velocities[0,:], velocities[1,:])
#             loss = loss_function(output, velocities.to(device))
            running_loss += loss.item()
            if i < plot_amount:
#                 display_graphs(loss_function, output[0,:], output[1,:], velocities[0,:], velocities[1,:], range(20), c)
                f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = (15, 5)) 
                # gen_a -> real_a, gen_b -> real_b
                ax0.scatter(times%repeat_time, vel[direction,:].cpu().numpy(), color='blue', label='real')
                ax1.scatter(times%repeat_time, vel[1-direction,:].cpu().numpy(), color='blue', label='real')
                ax0.scatter(times%repeat_time, output[0,:].cpu().numpy(), color='orange', label='predicted')
                ax1.scatter(times%repeat_time, output[1,:].cpu().numpy(), color='orange', label='predicted')
                ax0.set_xlabel('time')
                ax1.set_xlabel('time')
                ax0.set_ylabel('velocity')
                ax1.set_ylabel('velocity')
                ax0.legend()
                ax1.legend()
                f.suptitle(f'system num: {c}')
            avg_loss = running_loss/len(data)
        print("Loss: {:.4f}...".format(avg_loss))
    return avg_loss

def check_all_forward(spectra_net: Nets.SpectraNet, doppler_net: Nets.DopplerNet, data, plot_amount:int = 10, std_vec = None):
    repeat_time = 7
    spectra_net.eval()
    doppler_net.eval()
    loss_function = nn.MSELoss(reduction = 'mean')
    sample_amount = len(data)
    spec_seq_size, spec_seq_amount, spec_wave_amount = spectra_net.seq_size, spectra_net.seq_amount, spectra_net.wave_amount
    dop_seq_size, dop_seq_amount, dop_input_length = doppler_net.seq_size, doppler_net.seq_amount, doppler_net.combine_wave_input 
    
    spec_running_loss = 0
    dop_running_loss = 0
    with torch.no_grad():
        for i, (observ, specs, c, times, vel) in enumerate(data):
            spec_hidden = spectra_net.init_hidden()
            dop_hidden = doppler_net.init_hidden()
            obs, specs = split_vector_to_seqs(observ, spec_seq_size, spec_seq_amount, spec_wave_amount).to(device), specs.to(device)
            spec_out = spectra_net(obs, spec_hidden, method=3)
            spec_hidden = spec_out[2]
            loss, direction = calc_loss(loss_function, spec_out[0], spec_out[1], specs[0,:].view(1, -1), specs[1,:].view(1, -1))
            spec_running_loss += loss.item()
            
#             doppler_input = torch.cat((observ.to(device).repeat(2,1),
#                                   torch.cat((specs[direction,:].repeat(20,1), specs[1-direction,:].repeat(20,1)), 0)), 1)
            doppler_input = torch.cat((observ.to(device).repeat(2,1),
                                  torch.cat((spec_out[direction][0].repeat(20,1), spec_out[1-direction][0].repeat(20,1)), 0)), 1)

#             print(f'spectra1 size: {spectra1.size()}, specs: {specs[:, 0].size()}')
            output, doppler_val_hidden = doppler_net(split_vector_to_seqs(doppler_input, dop_seq_size, dop_seq_amount,
                                                                      dop_input_length).to(device), dop_hidden)
            output = torch.transpose(output,0,1)
            vel = vel.to(device)
#             '''HERE WE DO NEED THAT THE 1st VELOCITY WOULD CORRESPOND TO THE 1st OUTPUT AND VISE VERSA'''
            loss = loss_function(output[0,:], vel[direction,:]) + loss_function(output[1,:], vel[1-direction,:])
#             loss = calc_loss(loss_function, output[0,:], output[1,:], vel[0,:], vel[1,:])
#             loss = loss_function(output, velocities.to(device))
            dop_running_loss += loss.item()
    
            if i < plot_amount:
                f, ax = plt.subplots(2, 2, sharey=False, figsize = (15, 10)) 
                # gen_a -> real_a, gen_b -> real_b
                ax[0,0].plot(std_vec, specs[direction,:].cpu().numpy(), color='blue', label = 'real')
                ax[0,1].plot(std_vec, specs[1-direction,:].cpu().numpy(), color='blue', label = 'real')
                ax[0,0].plot(std_vec, spec_out[0][0].cpu().numpy(), color='red', alpha = 0.5, label = 'predicted')
                ax[0,1].plot(std_vec, spec_out[1][0].cpu().numpy(), color='red', alpha = 0.5, label = 'predicted')
                ax[1,0].scatter(times%repeat_time, vel[direction,:].cpu().numpy(), color='blue', label='real')
                ax[1,1].scatter(times%repeat_time, vel[1-direction,:].cpu().numpy(), color='blue', label='real')
                ax[1,0].scatter(times%repeat_time, output[0,:].cpu().numpy(), color='orange', label='predicted')
                ax[1,1].scatter(times%repeat_time, output[1,:].cpu().numpy(), color='orange', label='predicted')
                ax[1,0].set_xlabel('time')
                ax[1,1].set_xlabel('time')
                ax[1,0].set_ylabel('velocity')
                ax[1,1].set_ylabel('velocity')
                ax[1,0].legend()
                ax[1,1].legend()
                f.suptitle(f'system num: {c}')
                ax[0,0].set_xlabel('wave length')
                ax[0,1].set_xlabel('wave length')
                ax[0,0].legend()
                ax[0,1].legend()
        spec_avg_loss = spec_running_loss/len(data)
        dop_avg_loss = dop_running_loss/len(data)
        print("Spectra Loss: {:.4f}...".format(spec_avg_loss),
             "Doppler Loss: {:.4f}...".format(dop_avg_loss))
    return spec_avg_loss, dop_avg_loss

def check_all_forward_optimal(spectra_net: Nets.SpectraNet, doppler_net: Nets.DopplerNet, data, plot_amount:int = 10, std_vec = None):
    repeat_time = 7
    spectra_net.eval()
    doppler_net.eval()
    loss_function = nn.MSELoss(reduction = 'mean')
    sample_amount = len(data)
    spec_seq_size, spec_seq_amount, spec_wave_amount = spectra_net.seq_size, spectra_net.seq_amount, spectra_net.wave_amount
    dop_seq_size, dop_seq_amount, dop_input_length = doppler_net.seq_size, doppler_net.seq_amount, doppler_net.combine_wave_input 
    
    spec_running_loss = 0
    dop_running_loss = 0
    with torch.no_grad():
        for i, (observ, specs, c, times, vel) in enumerate(data):
            spec_hidden = spectra_net.init_hidden()
            dop_hidden = doppler_net.init_hidden()
            obs, specs = split_vector_to_seqs(observ, spec_seq_size, spec_seq_amount, spec_wave_amount).to(device), specs.to(device)
            spec_out = spectra_net(obs, spec_hidden, method=3)
            spec_hidden = spec_out[2]
            loss, direction = calc_loss(loss_function, spec_out[0], spec_out[1], specs[0,:].view(1, -1), specs[1,:].view(1, -1))
            spec_running_loss += loss.item()
            
#             doppler_input = torch.cat((observ.to(device).repeat(2,1),
#                                   torch.cat((specs[direction,:].repeat(20,1), specs[1-direction,:].repeat(20,1)), 0)), 1)
            doppler_input = torch.cat((observ.to(device).repeat(2,1),
                                  torch.cat((spec_out[0][0].repeat(20,1), spec_out[1][0].repeat(20,1)), 0)), 1)

#             print(f'spectra1 size: {spectra1.size()}, specs: {specs[:, 0].size()}')
            output, doppler_val_hidden = doppler_net(split_vector_to_seqs(doppler_input, dop_seq_size, dop_seq_amount,
                                                                      dop_input_length).to(device), dop_hidden)
            output = torch.transpose(output,0,1)
            vel = vel.to(device)
#             '''HERE WE DO NEED THAT THE 1st VELOCITY WOULD CORRESPOND TO THE 1st OUTPUT AND VISE VERSA'''
#             loss = loss_function(output[0,:], vel[direction,:]) + loss_function(output[1,:], vel[1-direction,:])
            loss, dop_direction = calc_loss(loss_function, output[0,:], output[1,:], vel[0,:], vel[1,:])
#             loss = loss_function(output, velocities.to(device))
            dop_running_loss += loss.item()
    
            if i < plot_amount:
                f, ax = plt.subplots(2, 2, sharey=False, figsize = (15, 10)) 
                # gen_a -> real_a, gen_b -> real_b
                ax[0,0].plot(std_vec, specs[direction,:].cpu().numpy(), color='blue', label = 'real')
                ax[0,1].plot(std_vec, specs[1-direction,:].cpu().numpy(), color='blue', label = 'real')
                ax[0,0].plot(std_vec, spec_out[0][0].cpu().numpy(), color='red', alpha = 0.5, label = 'predicted')
                ax[0,1].plot(std_vec, spec_out[1][0].cpu().numpy(), color='red', alpha = 0.5, label = 'predicted')
                ax[1,0].scatter(times%repeat_time, vel[dop_direction,:].cpu().numpy(), color='blue', label='real')
                ax[1,1].scatter(times%repeat_time, vel[1-dop_direction,:].cpu().numpy(), color='blue', label='real')
                ax[1,0].scatter(times%repeat_time, output[0,:].cpu().numpy(), color='orange', label='predicted')
                ax[1,1].scatter(times%repeat_time, output[1,:].cpu().numpy(), color='orange', label='predicted')
                ax[1,0].set_xlabel('time')
                ax[1,1].set_xlabel('time')
                ax[1,0].set_ylabel('velocity')
                ax[1,1].set_ylabel('velocity')
                ax[1,0].legend()
                ax[1,1].legend()
                f.suptitle(f'system num: {c}')
                ax[0,0].set_xlabel('wave length')
                ax[0,1].set_xlabel('wave length')
                ax[0,0].legend()
                ax[0,1].legend()
        spec_avg_loss = spec_running_loss/len(data)
        dop_avg_loss = dop_running_loss/len(data)
        print("Spectra Loss: {:.4f}...".format(spec_avg_loss),
             "Doppler Loss: {:.4f}...".format(dop_avg_loss))
    return spec_avg_loss, dop_avg_loss



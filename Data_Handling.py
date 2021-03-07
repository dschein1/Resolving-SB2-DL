import torch
import matplotlib.pyplot as plt
import hyper_parameters as hp

## Data Hendling
# We don't have a lot of different stars to work with, we have a lot of data on their systems (20 observations, like in real life) but not a lot of different systems. so we will work with 1 batch size probably. but the observation amount will act as the inner net batch size - meaning we will work on all the 20 observations simultaniously and then average them out to get on single solution - spectra vector  

def split_vector_to_seqs(wave_tensor, seq_length:int = hp.wave_sequence_length,
                         seq_amount=hp.wave_seq_amount, new_length=hp.observation_amount):
    '''
    wave_vector: a 2-D torch.tensor of shape: obs_amount X wave_samples,
    so in the end for each system we have data that looks like:
    obs_amount X seq_amount X seq_length
    
    to save calculations we calculate the new lengths in the __init__ of the model
    '''
#     sequences_amount = wave_tensor.size()[-1] // seq_length
#     new_length = sequences_amount * seq_length
    # print(f'length of data: {len(data)} sequences amount: {sequences_amount} ')
    if new_length > wave_tensor.shape[1]:
        zero_pad = torch.zeros(wave_tensor.shape[0], new_length-wave_tensor.shape[1])
        wave_tensor = torch.cat((wave_tensor, zero_pad), 1)
    else:
        wave_tensor = wave_tensor[:, :new_length] #truncating the last not complete seq
    # print(f'length of new data: {len(data)}') 
    wave_tensor = wave_tensor.view(-1, seq_amount, seq_length)
#     print(f'in function: {wave_tensor.size()}')
    return wave_tensor

def combine_seqs_to_vector(wave_splitted_tensor, new_length=hp.observation_amount):
    '''
    the input is from the shape: obs_amount X seq_amount X seq_length
    output will be a 2-D tensor that combines seq_amountXseq_length to one vector
    '''
#     new_length = observation_amount
#    wave_splitted_tensor = wave_splitted_tensor.reshape(-1, new_length) # 20 - batch size
    wave_splitted_tensor = wave_splitted_tensor.reshape(20,-1) # 20 - batch size
#     print(f'in function: {wave_splitted_tensor.size()}')
    return wave_splitted_tensor
    
    
def display_graphs(loss,gen_a,gen_b,real_a,real_b, standard_vector, system_id = 'real'):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = (15, 5))
    if loss(gen_a,real_a) + loss(gen_b,real_b) < loss(gen_a,real_b) + loss(gen_b,real_a): 
        # gen_a -> real_a, gen_b -> real_b
        ax1.plot(standard_vector, real_a.cpu().numpy(), color='blue', label = 'real')
        ax2.plot(standard_vector, real_b.cpu().numpy(), color='blue', label = 'real')
        ax1.plot(standard_vector, gen_a.cpu().numpy(), color='red', alpha = 0.5, label = 'generated')
        ax2.plot(standard_vector, gen_b.cpu().numpy(), color='red', alpha = 0.5, label = 'generated')
    else:
        # gen_a -> real_b, gen_b -> real_a
        ax1.plot(standard_vector, real_a.cpu().numpy(), color='blue', label = 'real')
        ax2.plot(standard_vector, real_b.cpu().numpy(), color='blue', label = 'real')
        ax2.plot(standard_vector, gen_a.cpu().numpy(), color='red', alpha = 0.5, label = 'generated')
        ax1.plot(standard_vector, gen_b.cpu().numpy(), color='red', alpha = 0.5, label = 'generated')
    f.suptitle(f'system num: {system_id}')
    ax1.set_xlabel('wave length')
    ax2.set_xlabel('wave length')
    ax1.legend()
    ax2.legend()
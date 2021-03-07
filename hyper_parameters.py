'''Data Hyper-Parameters'''

# observation_amount The amount of time point we observe the star, it's also the length of the output speed vector 
# observation_amount is the batch size in the spectra net
batch_size = 20 # batch size is the batch size of the doppler net because there we calculate each velocity at a certian time point, independently of the other (so we can also mix the systems together)

observation_amount = 7500
wave_output = 1700
wave_sequence_length = 500
output_wave_sequence_length = 170
doppler_sequence_length12 = 100
doppler_sequence_length25 = 400

wave_seq_amount = observation_amount//wave_sequence_length #7500
gt_seq_amount = wave_output//output_wave_sequence_length  #1700
concat_value_output, concat_seq_amount = observation_amount+wave_output, (observation_amount+wave_output)//doppler_sequence_length25
'''the output spectra should be the same length (be structured from the same wave length) as the input'''
hidden_size = wave_sequence_length # we keep the hidden size the same as seq_length so we won't have to change
hidden_size_doppler = doppler_sequence_length25
num_layers_spectra = 2
num_layers_doppler = 2
drop_spectra = 0.5
drop_doppler = 0.5
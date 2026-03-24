#!/usr/bin/env python3

from scipy.io.wavfile import read

import numpy as np
import noisereduce as nr
import torch
import math

import FreeSimpleGUI as sg
# import threading
# import time as Time

import sounddevice as sd

# TODO: add UI for each of these options
# TODO: add settings saving and loading, which autosaves the most recently used settings to an .ini

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Running on device:", device)


# Sound device latency setting - desired latency between input and output
# TODO: dynamically determine latency by adjusting smaller and smaller until processing completes just under desired latency
# TODO: add UI
# sd_latency = 'low'
sd_latency = 0.06
# blocksize 0 is special setting that makes it determined by latency
sd_blocksize = 0 
# number of channels for the audio devices to use
# TODO: add UI
sd_channels = 2

# context duration needed to understand noise
# TODO: change user input to seconds, make size in frames determined by sample rate
# TODO: add UI
context_buffer_size = 12000
context_buffer = np.zeros((context_buffer_size, 2))
context_cut = None
# number of frames to crossfade between each block to prevent phase misalignment
# TODO: change user input to a percentage of blocksize
# TODO: add UI
context_blend_size = 400

# TODO: Pull noise from recording
# TODO: add UI
# noise = np.zeros(args.wave_buffer_size)
# noise_sr is used for all input and output sample rates so they match up.
noise_sr, noise = read("noisy.wav")
# print("noise SR:", noise_sr)

## noise reduction params
enable_noisereduction = False
nr_stationary = True
# if cuda is available, use it
nr_torch = torch.cuda.is_available()
# nr_jobs = 1
nr_prop_decrease = 1.0
nr_freq_mask_smooth_hz = 500
nr_time_mask_smooth_ms = 50
nr_chunk_size = 60000
nr_n_fft = 1024
nr_padding = 30000
nr_win_length = nr_n_fft
nr_hop_length = nr_win_length // 4

## stationary exclusive noise reduction params
# TODO: only visible when stationary active
nr_s_n_std_thresh = 1.5

## non-stationary exclusive noise reduction params
# TODO: only visible when non-stationary active
nr_ns_time_constant_s = 2.0

# TODO: add ui element for starting audio stream so horrible feedback loop does not start immediately upon launch with speakers
stop_audio = False

# block_num = 0 # for bug testing

# removes the last context_blend_size worth from the returned context buffer, at the cost of latency
# crossfades the first context_blend_size with context_cut cut from the previous wave
# returns the cut part to blend with the next wave
# |  buffer  |cut|
#            | next buffer |cut|
#                          | next next buffer |cut|
def next_buffer(buffer, buffer_cut, frames):
    buffer_channels = buffer.shape[1]
    buffer_left = buffer.shape[0] - frames - context_blend_size
    buffer_right = buffer.shape[0] - context_blend_size
    buffer_return = buffer[buffer_left:buffer_right].squeeze()
    buffer_cut_next = buffer[buffer_right:].squeeze()
    
    # TODO: improve blend smoothness, or find some way to do phase alignment?

    buffer_weight = np.linspace(np.zeros(buffer_channels), np.ones(buffer_channels), context_blend_size)
    buffer_return[:context_blend_size] = buffer_weight * buffer_return[:context_blend_size] + (1 - buffer_weight) * buffer_cut
    buffer_return = np.clip(buffer_return, -1, 1)

    return buffer_return, buffer_cut_next

# callback used in the sounddevice loop to process the audio
def callback(indata, outdata, frames, time, status):
    global context_buffer
    global context_cut
    # global block_num
    # print(status)

    audio = indata
    
    context_buffer = np.concatenate((context_buffer, audio), axis=0) # appends audio to end of context_buffer
    buffer_cut = int(context_buffer.shape[0] - context_buffer_size)
    context_buffer = context_buffer[max(0, buffer_cut):, ...] # shift back by blocksize, so size is context_buffer_size

    if enable_noisereduction:
        if nr_stationary:
                wave = nr.reduce_noise(context_buffer.transpose(), sr=noise_sr, stationary=True, y_noise=noise.transpose(),
                                prop_decrease=nr_prop_decrease, n_std_thresh_stationary=nr_s_n_std_thresh,
                                freq_mask_smooth_hz=nr_freq_mask_smooth_hz, time_mask_smooth_ms=nr_time_mask_smooth_ms,
                                chunk_size=nr_chunk_size, n_fft=nr_n_fft, padding=nr_padding,
                                win_length=nr_win_length, hop_length=nr_hop_length, use_torch=nr_torch)
                wave = wave.transpose()
        else:
                wave = nr.reduce_noise(context_buffer.transpose(), sr=noise_sr, prop_decrease=nr_prop_decrease,
                                    time_constant_s=nr_ns_time_constant_s,
                                    freq_mask_smooth_hz=nr_freq_mask_smooth_hz, time_mask_smooth_ms=nr_time_mask_smooth_ms,
                                    chunk_size=nr_chunk_size, n_fft=nr_n_fft, padding=nr_padding,
                                    win_length=nr_win_length, hop_length=nr_hop_length, use_torch=nr_torch)
                wave = wave.transpose()
    else:
        outdata[:] = audio
        return
    
#     # for bug testing, save wave here for each block, with file name according to block number
#     # block_num = block_num + 1
#     # scipy.io.wavfile.write("out/" + str(block_num) + ".wav", noise_sr, wave)

    if context_cut is None:
        wave_left = wave.shape[0] - frames - context_blend_size
        wave_right = wave.shape[0] - context_blend_size
        context_cut = wave[wave_right:, ...].squeeze()
        wave = wave[wave_left:wave_right, ...]
    else:
        wave, context_cut = next_buffer(wave, context_cut, frames)
    outdata[:] = wave


audio_stream = sd.Stream(samplerate=noise_sr, blocksize=sd_blocksize, 
                                     channels=sd_channels, latency=sd_latency, callback=callback)
audio_devices = sd.query_devices()
audio_hostapis = sd.query_hostapis()
# Add in the index into each host API dictionary
for index in range(len(audio_hostapis)):
    audio_hostapis[index]['index'] = index

# Separate audio devices into a nested list with outer list corresponding to which host API it uses, 
# inner list is a list of input devices or list of output devices,
# according to whether the device has any input or output channels
input_devices = []
output_devices = []
for list_API in audio_hostapis:
    input_devices_for_this_api = []
    output_devices_for_this_api = []
    for device_ID in list_API['devices']:
        if audio_devices[int(device_ID)]['max_input_channels'] > 0: # this device is an input
            # add a key and value to each device dictionary with its original index in sd.query_devices(), which sounddevice uses as an ID
            audio_devices[int(device_ID)]['index'] = int(device_ID)
            input_devices_for_this_api.append(audio_devices[int(device_ID)])
        if audio_devices[int(device_ID)]['max_output_channels'] > 0: # this device is an output
            # add a key and value to each device dictionary with its original index in sd.query_devices(), which sounddevice uses as an ID
            audio_devices[int(device_ID)]['index'] = int(device_ID)
            output_devices_for_this_api.append(audio_devices[int(device_ID)])
    input_devices.append(input_devices_for_this_api)
    output_devices.append(output_devices_for_this_api)


audio_device_menu_layout = ['', [
    'Host API', [str(api['name']) + '::ID  ' + str(api['index']) + '_API' for api in audio_hostapis],
    'Input Device', [str(device['name']) + '::ID  ' + str(device['index']) + '_INPUT' for device in input_devices[sd.default.hostapi]],
    'Output Device', [str(device['name']) + '::ID  ' + str(device['index']) + '_OUTPUT' for device in output_devices[sd.default.hostapi]]]]


# TODO: add option to browse files for a recorded noise
layout = [[sg.Button('Noise Reduction Disabled', key='NR_ON', button_color="gray"),
            # sg.Button('Record Noise', key='NR_REC'),
        sg.Checkbox('Stationary', default=nr_stationary, key='NR_STAT', enable_events=True)],
        [sg.Text('Proportion Noise Decrease:'), 
            sg.Slider(range=(0.0, 1.0), resolution=0.01, default_value=nr_prop_decrease, key='NR_PROP_DECREASE', orientation='h', change_submits=True)],
        [sg.Text('Stationary Noise Threshold (σ):'),
            sg.Slider(range=(0.0, 6.0), resolution=0.01, default_value=nr_s_n_std_thresh, key='NR_S_N_STD_THRESH', orientation='h', change_submits=True)],
        [sg.Text('Mask Smoothing [Frequency (Hz), Time (ms)]:')],
        [sg.Slider(range=(noise_sr / (nr_n_fft / 2), 20000), resolution=10, default_value=nr_freq_mask_smooth_hz, key='NR_FREQ_MASK_SMOOTH_HZ', orientation='h', change_submits=True),
         sg.Slider(range=(math.ceil((nr_hop_length / noise_sr) * 1000), 5000), resolution=1, default_value=nr_time_mask_smooth_ms, key='NR_TIME_MASK_SMOOTH_MS', orientation='h', change_submits=True)],
        [sg.Text('Chunk Size (samples):'),
            sg.Slider(range=(10000, 120000), resolution=1000, default_value=nr_chunk_size, key='NR_CHUNK_SIZE', orientation='h', change_submits=True)],
        [sg.Text('# FFTs:'),
            sg.Slider(range=(200, 8192), resolution=2, default_value=nr_n_fft, key='NR_N_FFT', orientation='h', change_submits=True)],
        [sg.Text('Padding:'),
            sg.Slider(range=(10000, 60000), resolution=1000, default_value=nr_padding, key='NR_PADDING', orientation='h', change_submits=True)],
        [sg.Text('Window Length:'),
            sg.Slider(range=(1, 1024), resolution=2, default_value=nr_win_length, key='NR_WIN_LENGTH', orientation='h', change_submits=True)],
        [sg.Text('Hop Length:'),
            sg.Slider(range=(1, 512), resolution=1, default_value=nr_hop_length, key='NR_HOP_LENGTH', orientation='h', change_submits=True)],
        [sg.ButtonMenu('Device Settings', audio_device_menu_layout, key='DEVICE_SETTINGS')]]

window = sg.Window('Noise Reducer', layout)

# Start processing audio.
audio_stream.start()

# Start the GUI
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        audio_stream.abort()
        break
    elif event == 'NR_ON':
        if enable_noisereduction:
            window['NR_ON'].update('Noise Reduction Disabled', button_color="gray")
            enable_noisereduction = False
        else:
            window['NR_ON'].update('Noise Reduction Enabled', button_color="blue")
            enable_noisereduction = True
    elif event == 'NR_STAT':
        nr_stationary = values['NR_STAT']
    elif event == 'NR_REC':
        noise[:] = context_buffer
        # TODO: re-add way to record noise
        # noise_threshold = torch.max(log_norm(preprocess(noise)))
    elif event == 'NR_PROP_DECREASE':
        nr_prop_decrease = float(values['NR_PROP_DECREASE'])
    elif event == 'NR_S_N_STD_THRESH':
        nr_s_n_std_thresh = float(values['NR_S_N_STD_THRESH'])
    elif event == 'NR_FREQ_MASK_SMOOTH_HZ':
        nr_freq_mask_smooth_hz = int(values['NR_FREQ_MASK_SMOOTH_HZ'])
        # constraint (see NR_N_FFT, TODO: noise update): nr_freq_mask_smooth_hz >= noise_sr / (nr_n_fft / 2)
    elif event == 'NR_TIME_MASK_SMOOTH_MS':
        # constraint (see NR_HOP_LENGTH, NR_WIN_LENGTH, NR_FFT, TODO: noise update): nr_time_mask_smooth_ms >= (nr_hop_length / noise_sr) * 1000
        nr_time_mask_smooth_ms = int(values['NR_TIME_MASK_SMOOTH_MS'])
    elif event == 'NR_CHUNK_SIZE':
        nr_chunk_size = int(values['NR_CHUNK_SIZE'])
    elif event == 'NR_N_FFT':
        nr_n_fft = int(values['NR_N_FFT'])
        # constraint: 0 < nr_win_length <= n_fft
        if nr_n_fft < nr_win_length:
            nr_win_length = nr_n_fft
            window['NR_WIN_LENGTH'].update(nr_win_length, range=(1, nr_n_fft))
            # inherited constraint: 0 < hop_length <= win_length
            if nr_hop_length > nr_win_length:
                nr_hop_length = nr_win_length
                window['NR_HOP_LENGTH'].update(nr_hop_length, range=(1, nr_win_length))
                # inherited2 constraint: nr_time_mask_smooth_ms >= (nr_hop_length / noise_sr) * 1000
                if nr_time_mask_smooth_ms < (nr_hop_length / noise_sr) * 1000:
                    nr_time_mask_smooth_ms = math.ceil((nr_hop_length / noise_sr) * 1000)
                    window['NR_TIME_MASK_SMOOTH_MS'].update(nr_time_mask_smooth_ms, range=(math.ceil((nr_hop_length / noise_sr) * 1000), 5000))
                else:
                    window['NR_TIME_MASK_SMOOTH_MS'].update(range=(math.ceil((nr_hop_length / noise_sr) * 1000), 5000))
            else:
                window['NR_HOP_LENGTH'].update(range=(1, nr_win_length))
        else:
            window['NR_WIN_LENGTH'].update(range=(1, nr_n_fft))
        # constraint: nr_freq_mask_smooth_hz >= noise_sr / (nr_n_fft / 2)
        if nr_freq_mask_smooth_hz < noise_sr / (nr_n_fft / 2):
            nr_freq_mask_smooth_hz = noise_sr / (nr_n_fft / 2)
            window['NR_FREQ_MASK_SMOOTH_HZ'].update(nr_freq_mask_smooth_hz, range=(noise_sr / (nr_n_fft / 2), 20000))
        else:
            window['NR_FREQ_MASK_SMOOTH_HZ'].update(range=(noise_sr / (nr_n_fft / 2), 20000))
    elif event == 'NR_PADDING':
        nr_padding = int(values['NR_PADDING'])
    elif event == 'NR_WIN_LENGTH':
        # constraint (see NR_N_FFT): 0 < win_length <= n_fft
        nr_win_length = int(values['NR_WIN_LENGTH'])
        # constraint: 0 < hop_length <= win_length
        if nr_hop_length > nr_win_length:
            nr_hop_length = nr_win_length
            window['NR_HOP_LENGTH'].update(nr_hop_length, range=(1, nr_win_length))
            # inherited constraint: nr_time_mask_smooth_ms >= (nr_hop_length / noise_sr) * 1000
            if nr_time_mask_smooth_ms < (nr_hop_length / noise_sr) * 1000:
                nr_time_mask_smooth_ms = math.ceil((nr_hop_length / noise_sr) * 1000)
                window['NR_TIME_MASK_SMOOTH_MS'].update(nr_time_mask_smooth_ms, range=(math.ceil((nr_hop_length / noise_sr) * 1000), 5000))
            else:
                window['NR_TIME_MASK_SMOOTH_MS'].update(range=(math.ceil((nr_hop_length / noise_sr) * 1000), 5000))
        else:
            window['NR_HOP_LENGTH'].update(range=(1, nr_win_length))
    elif event == 'NR_HOP_LENGTH':
        # constraint (see NR_WIN_LENGTH, NR_FFT): 0 < hop_length <= win_length
        nr_hop_length = int(values['NR_HOP_LENGTH'])
        # constraint: nr_time_mask_smooth_ms >= (nr_hop_length / noise_sr) * 1000
        if nr_time_mask_smooth_ms < (nr_hop_length / noise_sr) * 1000:
            nr_time_mask_smooth_ms = math.ceil((nr_hop_length / noise_sr) * 1000)
            window['NR_TIME_MASK_SMOOTH_MS'].update(nr_time_mask_smooth_ms, range=(math.ceil((nr_hop_length / noise_sr) * 1000), 5000))
        else:
            window['NR_TIME_MASK_SMOOTH_MS'].update(range=(math.ceil((nr_hop_length / noise_sr) * 1000), 5000))
    elif event == 'DEVICE_SETTINGS':
        # make sure the value of DEVICE_SETTINGS is a string before comparing it to key strings
        if not isinstance(values['DEVICE_SETTINGS'], str):
            print("Value of DEVICE_SETTINGS event is not a string, when it should be.")
            break
        
        # Now, check which menu was selected from Host API, Input Device, or Output Device.
        if values['DEVICE_SETTINGS'][-4:] == '_API':
            # Not great implementation, but the menu requires values to be string
            api_ID = int(values['DEVICE_SETTINGS'][-7:-4])
            # When the Host API is changed, the Input Device and Output Device menus need to be updated.
            audio_device_menu_layout = ['', [
                'Host API', [str(api['name']) + '::ID  ' + str(api['index']) + '_API' for api in audio_hostapis],
                'Input Device', [str(device['name']) + '::ID  ' + str(device['index']) + '_INPUT' for device in input_devices[api_ID]],
                'Output Device', [str(device['name']) + '::ID  ' + str(device['index']) + '_OUTPUT' for device in output_devices[api_ID]]]]
            window['DEVICE_SETTINGS'].update(audio_device_menu_layout)
            # The current input and output devices should be changed to the default for that Host API?
            input_ID = int(audio_hostapis[api_ID]['default_input_device'])
            output_ID = int(audio_hostapis[api_ID]['default_output_device'])
            # For some reason some of the default devices are invalid. Try to instead find a valid device for that host API if it's invalid.
            try:
                sd.check_input_settings(device = input_ID, samplerate=noise_sr, blocksize=sd_blocksize, 
                                     channels=sd_channels, latency=sd_latency)
            except:
                found_valid = False
                for device in input_devices[api_ID]:
                    try:
                        input_ID = device['index']
                        sd.check_input_settings(device = input_ID, samplerate=noise_sr, blocksize=sd_blocksize, 
                                     channels=sd_channels, latency=sd_latency)
                    except:
                        # print('input device', input_ID, 'causes exception')
                        continue
                    else: # If one of the devices work, then keep it and stop searching.
                        # print('input device', input_ID, 'found that does not cause exception')
                        found_valid = True
                    if found_valid:
                        break
                if not found_valid:
                    print('No valid input device found for this host API :(')
            
            try:
                sd.check_output_settings(device = output_ID, samplerate=noise_sr, blocksize=sd_blocksize, 
                                     channels=sd_channels, latency=sd_latency)
            except:
                found_valid = False
                for device in output_devices[api_ID]:
                    try:
                        output_ID = device['index']
                        sd.check_output_settings(device = output_ID, samplerate=noise_sr, blocksize=sd_blocksize,
                                     channels=sd_channels, latency=sd_latency)
                    except:
                        # print('output device', output_ID, 'causes exception')
                        continue
                    else: # If one of the devices work, then keep it and stop searching.
                        # print('output device', output_ID, 'found that does not cause exception')
                        found_valid = True
                    if found_valid:
                        break
                if not found_valid:
                    print('No valid output device found for this host API :(')
                # print('output device works fine')
            # print(input_ID, '|', output_ID)
            
            # Stop the audio stream and replace it with a new one that uses the specified input device, then start it back.
            audio_stream.stop()
            audio_stream = sd.Stream(device=(input_ID, output_ID), samplerate=noise_sr, blocksize=sd_blocksize, 
                                     channels=sd_channels, latency=sd_latency, callback=callback)
            audio_stream.start()

        elif values['DEVICE_SETTINGS'][-6:] == '_INPUT':
            # Get the index of the device from the key string in values via a hardcoded position splice
            input_ID = int(values['DEVICE_SETTINGS'][-9:-6])
            # Get the current output device index to keep after stream is restarted
            output_ID = audio_stream.device[1]
            
            # Stop the audio stream and replace it with a new one that uses the specified input device, then start it back.
            audio_stream.stop()
            audio_stream = sd.Stream(device=(input_ID, output_ID), samplerate=noise_sr, blocksize=sd_blocksize,  
                                     channels=sd_channels, latency=sd_latency, callback=callback)
            audio_stream.start()

        elif values['DEVICE_SETTINGS'][-7:] == '_OUTPUT':
            # Get the index of the device from the key string in values via a hardcoded position splice
            output_ID = int(values['DEVICE_SETTINGS'][-10:-7])
            # Get the current input device index to keep after stream is restarted
            input_ID = audio_stream.device[0]
            
            # Stop the audio stream and replace it with a new one that uses the specified input device, then start it back.
            audio_stream.stop()
            audio_stream = sd.Stream(device=(input_ID, output_ID), samplerate=noise_sr, blocksize=sd_blocksize, 
                                     channels=sd_channels, latency=sd_latency, callback=callback)
            audio_stream.start()
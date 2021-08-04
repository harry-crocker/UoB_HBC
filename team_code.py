#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of the required functions, remove non-required functions, and add your own functions.

################################################################################
#
# Imported functions and variables
#
################################################################################

# Import functions. These functions are not required. You can change or remove them.
from helper_code import *
import numpy as np, os, sys, joblib


import dill
import numpy as np, os, sys
import tensorflow as tf

from data_funcs import *
from model_funcs import *

# import wandb
# from wandb.keras import WandbCallback

# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_configurations = (twelve_leads, six_leads, four_leads, three_leads, two_leads)

# twelve_lead_model_filename = '12_lead_model'
# six_lead_model_filename = '6_lead_model'
# four_lead_model_filename = '4_lead_model'
# three_lead_model_filename = '3_lead_model'
# two_lead_model_filename = '2_lead_model'
# model_filenames = (twelve_lead_model_filename, six_lead_model_filename, four_lead_model_filename, three_lead_model_filename, two_lead_model_filename) 


####################################s############################################
#
# Training model function
#
################################################################################

# Train your model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract classes from dx_mapping_scored.csv file as want to have same classes for all models
    print('Extracting classes...')
    classes= get_classes()
    num_classes = len(classes)

    # Extract features and labels from dataset.
    print('Finding Files...')
    # In the real submission all training files are in a single folder
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)
    sequence = []
    for i in range(num_recordings):
        if i % 20 == 0:
            sequence.append(i)

    header_files = [header_files[i] for i in sequence]
    recording_files = [recording_files[i] for i in sequence]

    train_header_files, train_recording_files, val_header_files, val_recording_files = train_val_split(header_files, recording_files, config.val_split)

    print(num_recordings, 'Files found')
    print(len(train_recording_files), 'for training; ', len(val_recording_files), 'for threshold calculation')
    if not num_recordings:
        raise Exception('No data within:', data_directory.split('/')[-1])

    # Get data
    _, train_labels_list, train_recording_list, train_ecg_lengths = load_data(train_header_files, train_recording_files, twelve_leads, classes)
    _, val_labels_list, val_recording_list, val_ecg_lengths = load_data(val_header_files, val_recording_files, twelve_leads, classes)

    # Model configuration file defined in model_funcs.py
    config.classes = classes
    config.num_classes = num_classes
        
    #############
    # Loop through each  model and train
    ############
    for model_leads in lead_configurations:
        model_filename = get_model_filename(model_leads)
        print('Training', model_filename)
        print(model_leads)
        # Add lead-specific model configurations
        config.leads = model_leads
        config.num_leads = len(config.leads)
        config.input_shape = [config.Window_length, config.num_leads]
        config.thresholds = [0.5]*num_classes   # Reset this
        config.lead_indexes = lead_indexes(twelve_leads, config.leads)

        # run = wandb.init(project='FinalModels', allow_val_change=True)  ###########################
        # wandb.config.update(vars(config), allow_val_change=True) ###########################

        cbs = []
        steps = config.SpE * np.ceil(len(train_recording_files) / config.batch_size) * config.epochs
        lr_schedule = OneCycleScheduler(config.lr, steps, wd=config.wd, mom_min=0.85, mom_max=0.95)
        cbs.append(lr_schedule) 
        # cbs.append(WandbCallback()) ###########################

        # Build Model
        model = Build_InceptionTime(config.input_shape, config.num_classes, config.num_modules, config.lr, config.wd, config.optimizer, config.loss_func, 
                                    config.Window_length, config.lap, config.filters, config.kernel_sizes, config.head_nodes)

        # Train model
        history = model.fit(train_generator(train_labels_list, train_recording_list, train_ecg_lengths, config), 
                        steps_per_epoch= steps // config.epochs,
                        epochs=config.epochs, 
                        batch_size=config.batch_size,
                        # validation_data=train_generator(val_labels_list, val_recording_list, val_ecg_lengths, config, val=True), ###################
                        # validation_steps=len(val_header_files)//config.batch_size,  ################################
                        callbacks=cbs)

        #############################################
        print('Calculating Thresholds')
        predictions = []
        labels = []
        num_val = len(val_recording_files)
        # Package model
        val_model = (model, config)
        # Loop through all validation set and calculate predictions (probabilities)
        # This code block is similar to in test_model.py
        for i in range(num_val):
            if i % 100 == 0:
                print('    {}/{}...'.format(i+1, num_val))

            # Load header and recording.
            header = load_header(val_header_files[i])
            recording = load_recording(val_recording_files[i])
            leads = get_leads(header)

            # Apply model to recording.
            if all(lead in leads for lead in model_leads):
                _, _, probabilities = run_model(val_model, header, recording)

            predictions.append(probabilities)

            label = one_hot_encode_labels(header, classes)
            labels.append(label)

        # Use probabilities to find classwise thresholds
        thresholds = find_thresholds(np.array(labels), np.array(predictions))
        config.thresholds = thresholds
        # wandb.config.update(vars(config), allow_val_change=True) ##############################

        # Save model
        filename = os.path.join(model_directory, model_filename)
        model.save_weights(filename)
        save_object(config, filename+'Config.pkl')
        # run.join() ################################

################################################################################
#
# Running trained model function
#
################################################################################

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def run_model(model, header, recording):
    # Unpack model
    model, config = model
    thresholds = config.thresholds

    # Preprocess recording
    _, _, recording = get_features(header, recording, config.leads)
    recording = np.swapaxes(recording, 0, 1)    # Needs to be of form (num_samples, num_channels)
    # Get sampling data from header
    frequency = get_frequency(header)
    num_samples = get_num_samples(header)
    # Downsample the recording
    recording = downsample_recording(recording, frequency, num_samples)
    recording = np.expand_dims(recording, 0)    # Needs to be of form (num_recordings, num_samples, num_channels)

    # Predict
    outputs = model.compute_predictions(recording)[0] # Only a single prediction (opposite of above)

    # Predict labels and probabilities.
    labels = [0]*config.num_classes
    for i in range(config.num_classes):
        if outputs[i] > thresholds[i]:
            labels[i] = 1

    probabilities = list(np.array(outputs))

    # Remove normal label if other problems found
    # normal_class = '426783006'
    # norm_idx = classes.index(normal_class)
    # if labels[norm_idx] == 1 and sum(labels) > 1:
    #   labels[norm_idx] == 0

    return config.classes, labels, probabilities

################################################################################
#
# File I/O functions
#
################################################################################

# I created these 2 functions
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        dill.dump(obj, output)

def load_object(filename):
    with open(filename, 'rb') as file:  # Overwrites any existing file.
        return dill.load(file)

# Load a trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def load_model(filename):
    # Load config file, create fresh model, load weights into model
    config = load_object(filename+'Config.pkl')
    model = Build_InceptionTime(config.input_shape, config.num_classes, config.num_modules, config.lr, config.wd, config.optimizer, 
                                config.loss_func, config.Window_length, config.lap, config.filters, config.kernel_sizes, config.head_nodes)
    model.load_weights(filename)
    return (model, config)

# Define the filename(s) for the trained models. This function is not required. You can change or remove it.
def get_model_filename(leads):
    sorted_leads = sort_leads(leads)
    return 'model_' + '-'.join(sorted_leads)

################################################################################
#
# Feature extraction function
#
################################################################################

# # Extract features from the header and recording. This function is not required. You can change or remove it.
# def get_features(header, recording, leads):
#     # Extract age.
#     age = get_age(header)
#     if age is None:
#         age = float('nan')

#     # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
#     sex = get_sex(header)
#     if sex in ('Female', 'female', 'F', 'f'):
#         sex = 0
#     elif sex in ('Male', 'male', 'M', 'm'):
#         sex = 1
#     else:
#         sex = float('nan')

#     # Reorder/reselect leads in recordings.
#     recording = choose_leads(recording, header, leads)

#     # Pre-process recordings.
#     adc_gains = get_adc_gains(header, leads)
#     baselines = get_baselines(header, leads)
#     num_leads = len(leads)
#     for i in range(num_leads):
#         recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

#     # Compute the root mean square of each ECG lead signal.
#     rms = np.zeros(num_leads)
#     for i in range(num_leads):
#         x = recording[i, :]
#         rms[i] = np.sqrt(np.sum(x**2) / np.size(x))

#     return age, sex, rms

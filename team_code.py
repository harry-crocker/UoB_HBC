#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

import dill
import numpy as np, os, sys
import tensorflow as tf

from helper_code import *
from model_funcs import *
from data_funcs import *

# To change when found save model
twelve_lead_model_filename = '12_lead_model'
six_lead_model_filename = '6_lead_model'
three_lead_model_filename = '3_lead_model'
two_lead_model_filename = '2_lead_model'
model_filenames = (twelve_lead_model_filename, six_lead_model_filename, three_lead_model_filename, two_lead_model_filename)
lead_configurations = (twelve_leads, six_leads, three_leads, two_leads)	# Defined in helper_code.py

# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#
################################################################################
#
# Training function
#
################################################################################

# Train your model. This function is *required*. Do *not* change the arguments of this function.
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
	train_header_files, train_recording_files, val_header_files, val_recording_files = train_val_split(header_files, recording_files, config.val_split)

	num_recordings = len(recording_files)
	print(num_recordings, 'Files found')
	print(len(train_recording_files), 'for training; ', len(val_recording_files), 'for threshold calculation')
	if not num_recordings:
		raise Exception('No data within:', data_directory.split('/')[-1])

	# Model configuration file defined in model_funcs.py
	config.classes = classes
	config.num_classes = num_classes
		
	#############
	# Loop through each  model and train
	############
	for model_leads,  model_filename in zip(lead_configurations, model_filenames):
		print('Training', model_filename)
		print(model_leads)
		# Add lead-specific model configurations
		config.leads = model_leads
		config.num_leads = len(config.leads)
		config.input_shape = [config.Window_length, config.num_leads]
		config.thresholds = [0.5]*num_classes	# Reset this
		config.epochs = 30 + 3*config.num_leads

		cbs = []
		steps = config.SpE * np.ceil(len(train_recording_files) / config.batch_size) * config.epochs
		lr_schedule = OneCycleScheduler(config.lr, steps, wd=config.wd, mom_min=0.85, mom_max=0.95)
		cbs.append(lr_schedule)

		# Build Model
		model = Build_InceptionTime(config.input_shape, config.num_classes, config.num_modules, config.lr, config.wd, config.optimizer, config.loss_func, 
									config.Window_length, config.lap, config.filters, config.kernel_sizes, config.head_nodes)

		# Train model
		history = model.fit(train_generator(train_header_files, train_recording_files, config), 
						steps_per_epoch= steps // config.epochs,
						epochs=config.epochs, 
						batch_size=config.batch_size,
						# validation_data=train_generator(val_header_files, val_recording_files, config),
						# validation_steps=len(val_header_files)//config.batch_size,
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

		# Save model
		filename = os.path.join(model_directory, model_filename)
		model.save_weights(filename)
		save_object(config, filename+'Config.pkl')


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


# Save your trained models.
## Dont use this
def save_model(filename, classes, leads, imputer, classifier):
	# Construct a data structure for the model and save it.
	d = {'classes': classes, 'leads': leads, 'imputer': imputer, 'classifier': classifier}
	joblib.dump(d, filename, protocol=0)

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
	filename = os.path.join(model_directory, twelve_lead_model_filename)
	return load_model(filename)

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
	filename = os.path.join(model_directory, six_lead_model_filename)
	return load_model(filename)

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
	filename = os.path.join(model_directory, three_lead_model_filename)
	return load_model(filename)

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
	filename = os.path.join(model_directory, two_lead_model_filename)
	return load_model(filename)

# Generic function for loading a model.
def load_model(filename):
	# Load config file, create fresh model, load weights into model
	config = load_object(filename+'Config.pkl')
	model = Build_InceptionTime(config.input_shape, config.num_classes, config.num_modules, config.lr, config.wd, config.optimizer, 
								config.loss_func, config.Window_length, config.lap, config.filters, config.kernel_sizes, config.head_nodes)
	model.load_weights(filename)
	return (model, config)

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
	return run_model(model, header, recording)

# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
	return run_model(model, header, recording)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
	return run_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
	return run_model(model, header, recording)

# Generic function for running a trained model.
## Change this to accept a single recording 
## Classes should use my function get_classes()
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
	# 	labels[norm_idx] == 0

	return config.classes, labels, probabilities

################################################################################
#
# Other functions
#
################################################################################

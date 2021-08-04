import os, sys
import numpy as np
import pandas as pd
from scipy import interpolate

from helper_code import *


# Not used and broken
# Files of different lengths cannot be in the same numpy array
def get_data(header_files, recording_files):
	# Returns the X, Z and y data for this dataset

	data = [] 
	labels = np.zeros((num_recordings, num_classes), dtype=np.bool) # One-hot encoding of classes

	for i in range(num_recordings):
		if i%100 ==1:   
			print('    {}/{}...'.format(i+1, num_recordings))

		# Load header and recording.
		header = load_header(header_files[i])
		recording = load_recording(recording_files[i])

		# Get age, sex and correctly ordered leads
		# twelve_leads here is from helper_code.py
		age, sex, recording = get_features(header, recording, twelve_leads)
		data.append(recording)
	
		# One hot encode based on header file and position in classes list
		current_labels = get_labels(header)
		for label in current_labels:
			if label in classes:
				j = classes.index(label)
				labels[i, j] = 1

	X = np.stack(data)
	y = labels
	print('Recordings shape:', X.shape, y.shape)
	return X, y

'''
def get_classes():
	path_to_classes = os.path.join(sys.path[0], 'dx_mapping_scored.csv')
	df = pd.read_csv(path_to_classes)
	SNOMED_CT_Codes = list(df['SNOMEDCTCode'])
	SNOMED_CT_Codes = [str(item) for item in SNOMED_CT_Codes]
	equivalent_classes = {'713427006': '59118001', '284470004': '63593006', '427172004': '17338001', '164909002': '733534002'}
	# Remove one of equivalent classes
	for label in equivalent_classes.keys():
		SNOMED_CT_Codes.remove(label)
		if not equivalent_classes[label] in SNOMED_CT_Codes:
			print('Equivalent class not in classes')
	# class_abbreviations = list(df['Abbreviation'])
	return SNOMED_CT_Codes
	'''

# from eval_helper_code import *
from evaluate_model import load_weights

def get_classes():
	weights_file = os.path.join(sys.path[0], 'weights.csv')
	classes, _ = load_weights(weights_file)
	# Unpack duplicate classes
	classes = [list(class_set)[0] for class_set in classes]
	return classes


def one_hot_encode_labels(header, classes):
	equivalent_classes = {'713427006': '59118001', '284470004': '63593006', '427172004': '17338001', '164909002': '733534002'}
	num_classes = len(classes)
	labels = np.zeros(num_classes, dtype=np.bool) # One-hot encoding of classes
	current_labels = get_labels(header)
	for label in current_labels:
		if label in equivalent_classes.keys():	# Remove one of equivalent classes
			label = equivalent_classes[label]
		if label in classes:
			j = classes.index(label)
			labels[j] = 1
	return labels


# downsample to 100hz for the model to accept
def downsample_recording(recording, frequency, num_samples):
	ecg_len = num_samples/frequency  # Length of time in seconds
	num_samples = int(num_samples)
	t = np.linspace(0, ecg_len, num=num_samples)
	# Create an interpolation class to return the values at a given t_new
	interp_func = interpolate.interp1d(t, recording, kind='linear', axis=0, assume_sorted=True)

	freq_new = 100
	num_samples_new = int(ecg_len*freq_new)
	# New times to evaluate function at (same length but fewer samples)
	t_new = np.linspace(0, ecg_len, num=num_samples_new)
	recording_new = interp_func(t_new)
	return recording_new


def train_val_split(header_files, recording_files, percent):
	num_files = len(header_files)
	rands = np.random.random(num_files)

	train_header_files = []
	val_header_files = []
	train_recording_files = []
	val_recording_files = []

	for i in range(num_files):
		if rands[i] < percent:
			val_header_files.append(header_files[i])
			val_recording_files.append(recording_files[i])
		else:
			train_header_files.append(header_files[i])
			train_recording_files.append(recording_files[i])

	return train_header_files, train_recording_files, val_header_files, val_recording_files


# Extract features from the header and recording.
# Important for selecting leads
def get_features(header, recording, leads, wide_features=False, preprocessing=False):
	# Extract age.
	if wide_features:
		age = get_age(header)
		if age is None:
			age = float('nan')

		# Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
		sex = get_sex(header)
		if sex in ('Female', 'female', 'F', 'f'):
			sex = 0
		elif sex in ('Male', 'male', 'M', 'm'):
			sex = 1
		else:
			sex = float('nan')
	else:
		age=0
		sex=0

	# Reorder/reselect leads in recordings.
	# Need to get this into training generator
	available_leads = get_leads(header)
	indices = list()
	for lead in leads:
		i = available_leads.index(lead)
		indices.append(i)
	recording = recording[indices, :]


	# Pre-process recordings.
	if preprocessing:
		adc_gains = get_adcgains(header, leads)
		baselines = get_baselines(header, leads)
		num_leads = len(leads)
		for i in range(num_leads):
			recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

	return age, sex, recording


def correct_leads(header, recording, leads):
	available_leads = get_leads(header)
	indices = list()
	for lead in leads:
		i = available_leads.index(lead)
		indices.append(i)
	recording = recording[indices, :]
	return recording

def lead_indexes(twelve_leads, leads):
	indexes = np.zeros(12, dtype=bool)
	for lead in leads:
		if lead in twelve_leads:
			idx = twelve_leads.index(lead)
			indexes[idx] = 1
		else:
			raise Exception(lead, 'not in', twelve_leads)

	return indexes









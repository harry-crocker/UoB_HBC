# Not used
def recordings_to_keep(header_files, recording_files, data_directory, training):
	# Returns the indexes of the recordings to keep based on training and dataset
	# Get the folder name of the dataset being processed
	dataset = data_directory.split('/')[-1]
	print('Processing:', dataset)

	if dataset == 'WFDB_PTBXL':
		# Load csv file
		path = os.path.join(sys.path[0], 'ptbxl_database.csv')
		Y = pd.read_csv(path, index_col='ecg_id')
		if training:
			to_keep = np.where(Y.strat_fold < 9)
		else:
			to_keep = np.where(Y.strat_fold >= 9)

		header_files = header_files[to_keep]
		recording_files = recording_files[to_keep]


	return header_files, recording_files

# custom model functions
	# Not needed (got another modified version)
	def find_thresholds(self, y_labels, y_hat):
		best_thresh = [0]*y_labels.shape[1]
		best_thresh_f1 = [0]*y_labels.shape[1]

		for i in range(y_labels.shape[1]):
			thresh = 0
			increment = 1e-3
			y = y_labels[:, i]
			while thresh < 1:
				thresh += increment
				y_pred = np.where(y_hat[:, i] > thresh, 1, 0)
				tp = np.count_nonzero(y_pred * y, axis=0)
				fp = np.count_nonzero(y_pred * (1 - y), axis=0)
				fn = np.count_nonzero((1 - y_pred) * y, axis=0)
				f1 = 2*tp / (2*tp + fn + fp + 1e-16)

				# If new F1 score is better than previous then update threshold
				if f1 > best_thresh_f1[i]:
					best_thresh_f1[i] = f1
					best_thresh[i] = thresh

		print('F1 Score on Validation:', np.mean(best_thresh_f1))
		return best_thresh
	
	# Also not needed (to be used elsewhere outside of model)
	def macro_f1(self, y, y_hat, thresh=0.5, thresholds=None):
		"""Compute the macro F1-score on a batch of observations (average F1 across labels)

		Args:
			y (int32 Tensor): labels array of shape (BATCH_SIZE, N_LABELS)
			y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
			thresh: probability value above which we predict positive

		Returns:
			macro_f1 (scalar Tensor): value of macro F1 for the batch
		"""
		y_pred = np.zeros(y.shape)

		for i in range(y.shape[1]):
			if thresholds:
				thresh = thresholds[i]
			y_pred[:, i] = np.where(y_hat[:, i] > thresh, 1, 0)

		tp = np.count_nonzero(y_pred * y, axis=0)
		fp = np.count_nonzero(y_pred * (1 - y), axis=0)
		fn = np.count_nonzero((1 - y_pred) * y, axis=0)
		f1 = 2*tp / (2*tp + fn + fp + 1e-16)
		macro_f1 = f1.mean()
		return macro_f1
	
	# NOT NEEDED (cannot be used with files of different lengths)
	def eval_on_test(self, X_val, y_val, X_test, y_test):
		# Function identifies the optimal thresholds based on the validation set and then provides the F1 score for both sets
		wind=self.wind
		lap=self.lap
		y_hat = self.compute_predictions(X_val, wind, lap)
		self.best_thresh = self.find_thresholds(y_val, y_hat)
		
		y_hat = self.compute_predictions(X_test, wind, lap)
		test_F1 = self.macro_f1(y_test, y_hat, thresholds=self.best_thresh)
		test_F1_nothresh = self.macro_f1(y_test, y_hat)
		print('F1 Score on Test:', test_F1)
		print('F1 Score on Test with thresh=0.5:', test_F1_nothresh)
		return max([test_F1,test_F1_nothresh])
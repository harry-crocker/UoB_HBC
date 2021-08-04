# Contains model functions for building InceptionTime model
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_addons as tfa
import transformers
# import time, sys

from helper_code import *
from data_funcs import *


######
# TO DO LIST
# - replace get_features with a single func to preprocess recording for use in generator and predictions
# - Change find_thresh to maximise challenge score rather than F1 score
######


# Class for empty config file
class Config_file():
    pass


dev_mode = False ################################
if dev_mode: 
    import tensorflow_addons as tfa
    import wandb 
    # # Metrics
    auroc = tf.keras.metrics.AUC()
    # Also change the file names in team_code
    # Also change metrics 
    # Also change update thresholds
    # Also change load model in test_model.py


# Create all configuration files
config = Config_file()
config.num_modules = 6 # 6
config.lr = 3e-3  # 1e-2
config.batch_size = 128  # PTB-XL = 128
config.optimizer='AdamWeightDecay'
config.wd = 1e-2 # Float
config.Window_length = 250 # 250
config.lap = 0.5
config.loss_func = 'BC'   # BC Or F1
config.SpE = 1 # 1
config.filters = 32
config.kernel_sizes = [3, 7, 17] #[9, 23, 49]
config.head_nodes = 2048
config.val_split = 0.1
config.epochs = 3


def load_data(header_files, recording_files, leads, classes):
    header_list = []
    recording_list = []
    labels_list = []
    ecg_lengths = []
    print('Preparing Samples')
    for i, (header_file, recording_file) in enumerate(zip(header_files, recording_files)):
        # Load from file
        header = load_header(header_file)
        recording = load_recording(recording_file)
        # Preprocess recording
        recording = correct_leads(header, recording, leads)
        recording = np.swapaxes(recording, 0, 1)    # Needs to be of form (num_samples, num_channels)
        # Downsample recording
        frequency = get_frequency(header)
        num_samples = get_num_samples(header)
        recording = downsample_recording(recording, frequency, num_samples)
        # Get labels
        labels = one_hot_encode_labels(header, classes)
        # Get ecg length in seconds
        ecg_len = num_samples/frequency
        # Store in lists
        ecg_lengths.append(ecg_len)
        header_list.append(header)
        labels_list.append(labels)
        recording_list.append(recording)
        if i % 1000 ==1:
            print(i, '/', len(recording_files)) 

    return header_list, labels_list, recording_list, ecg_lengths

# Generator functions for producing segments of ECG
def train_generator(labels_list, recording_list, ecg_lengths, config, val=False):
    wind = config.Window_length
    bs = config.batch_size
    num_recordings = len(labels_list)
    probs = np.array(ecg_lengths)/np.sum(ecg_lengths)
    if val:
        probs=None

    # Need to reset these every batch
    inputs = []
    targets = []
    bc = 0  # Batch count increments after every recording
    # Select ecgs indexes for this batch
    file_idxs = np.random.choice(range(num_recordings), size=bs, p=probs)
    
    while True:
        # Get the current ecg index
        file_idx = file_idxs[bc]
        # Get data from list
        labels = labels_list[file_idx]
        recording = recording_list[file_idx]

        # Get segement 
        max_start_idx = recording.shape[0] - wind
        t_idx = np.random.randint(0, max_start_idx)
        segment = recording[t_idx:t_idx+wind, config.lead_indexes] # SELECT LEADS HERE
        # Append outputs to list 
        inputs.append(segment)
        targets.append(labels)
        
        bc += 1
        if bc >= bs:
            # End of batch, output and reset
            retX = np.array(inputs)
            rety = np.array(targets)
            yield (retX, rety)
            # Generator will resume here after yield
            inputs = []
            targets = []
            bc = 0  # Batch count increments after every recording
            # Select ecgs indexes for this batch
            file_idxs = np.random.choice(range(num_recordings), size=bs, p=probs)

# Callback functions
class CosineAnnealer:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0
        
    def step(self):
        self.n += 1
        cos = np.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos

    
class OneCycleScheduler(Callback):
    """ `Callback` that schedules the learning rate on a 1cycle policy as per Leslie Smith's paper(https://arxiv.org/pdf/1803.09820.pdf).
    If the model supports a momentum parameter, it will also be adapted by the schedule.
    The implementation adopts additional improvements as per the fastai library: https://docs.fast.ai/callbacks.one_cycle.html, where
    only two phases are used and the adaptation is done using cosine annealing.
    In phase 1 the LR increases from `lr_max / div_factor` to `lr_max` and momentum decreases from `mom_max` to `mom_min`.
    In the second phase the LR decreases from `lr_max` to `lr_max / (div_factor * 1e4)` and momemtum from `mom_max` to `mom_min`.
    By default the phases are not of equal length, with the phase 1 percentage controlled by the parameter `phase_1_pct`.
    """

    def __init__(self, lr_max, steps, wd=1e-2, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25.):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor
        final_lr = lr_max / (div_factor * 1e4)
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps
        self.wd = wd
        
        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0
        
        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)], 
                 [CosineAnnealer(lr_max, final_lr, phase_2_steps), CosineAnnealer(mom_min, mom_max, phase_2_steps)]]
        
        self.lrs = []
        self.moms = []

    def on_train_begin(self, logs=None):
        self.phase = 0
        self.step = 0

        self.set_lr(self.lr_schedule().start)
        self.set_momentum(self.mom_schedule().start)
        
    def on_train_batch_begin(self, batch, logs=None):
        self.lrs.append(self.get_lr())
        self.moms.append(self.get_momentum())

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1
        if self.step >= self.phase_1_steps:
            self.phase = 1
            
        self.set_lr(self.lr_schedule().step())
        self.set_momentum(self.mom_schedule().step())
        
    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            return None
        
    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self.model.optimizer.momentum)
        except AttributeError:
            return None
        
    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            # tf.keras.backend.set_value(self.model.optimizer.weight_decay, self.wd)
        except AttributeError:
            pass # ignore
        
    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self.model.optimizer.momentum, mom)
        except AttributeError:
            pass # ignore

    def lr_schedule(self):
        return self.phases[self.phase][0]
    
    def mom_schedule(self):
        return self.phases[self.phase][1]
    
    def plot(self):
        ax = plt.figure()
        plt.plot(self.lrs)
        plt.plot(self.mom)


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr


# Custom loss function
def macro_double_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    This version uses the computation of soft-F1 for both positive and negative class for each label.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    tn = tf.reduce_sum((1 - y_hat) * (1 - y), axis=0)
    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-16)
    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-16)
    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost


# Building model functions
class CustomModel(keras.Model):
    def __init__(self, wind, lap, **kwargs):
        super().__init__(**kwargs)
        self.wind = wind
        self.lap=lap

    # def test_step(self, data):
    #   # Overwrite what happens in model.evaluate by 
    #   # Unpack the data
    #   x, y = data
    #   # Compute predictions
    #   y_pred = self.compute_predictions(x)
    #   # Updates the metrics tracking the loss
    #   self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    #   # Update the metrics.
    #   self.compiled_metrics.update_state(y, y_pred)
    #   # Return a dict mapping metric names to current value.
    #   # Note that it will include the loss (tracked in self.metrics).
    #   return {m.name: m.result() for m in self.metrics}

    def compute_predictions(self, X):
        wind = self.wind
        lap = self.lap
        sample_length = X.shape[1]
        max_start_idx = sample_length - wind # Exclusive
        y_pred = []
        
        step = int(wind*(1-lap))
        tidx = 0
        while tidx <= max_start_idx:
            segments = X[:, tidx:tidx+wind, :]
            segment_preds = self(segments, training=False)
            y_pred.append(segment_preds)
            tidx += step
        y_pred = tf.stack(y_pred, axis=2)
        y_pred = tf.math.reduce_max(y_pred, axis=2)
        return y_pred
    

def InceptionModule(input_tensor, num_filters=32, bottleneck_size=32, activation='linear', strides=1, bias=False, kernel_sizes=None):
    reg = None
    # From the input apply a bottleneck to condense features
    input_inception = layers.Conv1D(filters=bottleneck_size, 
                                    kernel_size=1, 
                                    padding='same', 
                                    activation=activation, 
                                    use_bias=bias)(input_tensor)
    # input_inception = layers.Dropout(0.5)(input_inception)
    # Add parallel Conv layers following this with different kernel lengths
    # Can choose below which kernel configurations to choose
    # kernel_sizes = [3, 5, 8, 11, 17]
    # kernel_sizes = [15, 31, 59]
    # kernel_sizes = [7, 15, 31]
    # kernel_sizes = [3, 7, 11]
    conv_list = []
    for KS in kernel_sizes:
        conv_branch = layers.Conv1D(filters=num_filters, 
                                            kernel_size=KS,
                                            strides=strides, 
                                            padding='same', 
                                            activation=activation, 
                                            use_bias=bias, 
                                            kernel_regularizer=reg)(input_inception)
        # conv_branch = layers.Dropout(0.3)(conv_branch)
        conv_list.append(conv_branch)
        
    # Parallel to the above have a maxpooling layer followed by a conv layer
    max_pool = layers.MaxPool1D(pool_size=3, 
                                    strides=1, 
                                    padding='same')(input_tensor)

    max_pool = keras.layers.Conv1D(filters=num_filters, 
                                    kernel_size=1,
                                    padding='same', 
                                    activation=activation, 
                                    use_bias=bias,
                                    kernel_regularizer=reg)(max_pool)
    # max_pool = layers.Dropout(0.3)(max_pool)
    conv_list.append(max_pool)
    
    # Join the parallel branches into 1, along a new 'feature' axis bringing to 3D again
    # axis 0 is each batch, axis 1 is time
    x = keras.layers.Concatenate(axis=2)(conv_list)
    x = keras.layers.BatchNormalization(axis=1)(x)
    x = keras.layers.Activation(activation='relu')(x)
    return x


def Build_InceptionTime(input_shape, num_classes, num_modules, learning_rate, wd, opt, loss_func, wind, lap, filters, kernel_sizes, head_nodes):
    bias=False
    input_layer = layers.Input(shape=input_shape)

    x = input_layer
    shortcut_start = input_layer

    for d in range(num_modules):
        x = InceptionModule(x, num_filters=filters, bottleneck_size=filters, activation='linear', strides=1, bias=False, kernel_sizes=kernel_sizes)
        # Add shortcut every 3 layers
        if d % 3 == 2:
            shortcut = layers.Conv1D(filters=int(x.shape[-1]), 
                                             kernel_size=1,
                                             padding='same', 
                                             use_bias=bias)(shortcut_start)
            shortcut = layers.BatchNormalization(axis=1)(shortcut)

            x = keras.layers.Add()([shortcut, x])
            x = keras.layers.Activation('relu')(x)
            shortcut_start = x

    x = layers.GlobalAveragePooling1D()(x)
    # x2 = layers.GlobalMaxPooling1D()(x)

    # x = layers.Concatenate(axis=1)([x1, x2])
    
    # x = layers.BatchNormalization()(x)
    # x = layers.Dropout(0.25)(x)
    
    # x = layers.Dense(head_nodes, activation='relu')(x)
    # x = layers.Dropout(0.5)(x)

    output_layer = layers.Dense(num_classes, activation='sigmoid')(x)

    model = CustomModel(wind, lap, inputs=input_layer, outputs=output_layer)
    
    # Choose loss function
    if loss_func == 'BC':
        loss = 'binary_crossentropy'
    elif loss_func == 'F1':
        loss = macro_double_soft_f1
        
    # Choose optimizer
    if opt == 'Adam':
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate) # ,amsgrad=True)
    elif opt == 'AdamW':
        optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=wd, beta_2=0.99, epsilon=1e-5)
    elif opt == 'AdamWeightDecay':
        optimizer = transformers.AdamWeightDecay(learning_rate=learning_rate, weight_decay_rate=wd )#, beta_2=0.99, epsilon=1e-5)
    
    lr_metric = get_lr_metric(optimizer)
    # F1 = tfa.metrics.F1Score(num_classes=num_classes, threshold=0.5, average='macro') #######################

    model.compile(loss=loss, 
                  optimizer=optimizer,
                  metrics=['accuracy', lr_metric, 
                  # auroc, F1 #####################
                  ])

    return model


# Threshold funnctions
# def find_thresholds(y_labels, y_hat):
#     best_thresh = [0.5]*y_labels.shape[1]
#     best_thresh_f1 = [0]*y_labels.shape[1]

#     for i in range(y_labels.shape[1]):
#         thresh = 0
#         increment = 1e-2
#         y = y_labels[:, i]
#         while thresh < 1:
#             thresh += increment
#             y_pred = np.where(y_hat[:, i] > thresh, 1, 0)
#             tp = np.count_nonzero(y_pred * y, axis=0)
#             fp = np.count_nonzero(y_pred * (1 - y), axis=0)
#             fn = np.count_nonzero((1 - y_pred) * y, axis=0)
#             f1 = 2*tp / (2*tp + fn + fp + 1e-16)

#             # If new F1 score is better than previous then update threshold
#             if f1 > best_thresh_f1[i]:
#                 best_thresh_f1[i] = f1
#                 best_thresh[i] = thresh

#     print('F1 Score on Validation:', np.mean(best_thresh_f1))
#     return best_thresh

from evaluate_model import *

def find_thresholds(y_labels, y_hat):

    y_labels, y_hat = convert_labels(y_labels, y_hat)

    best_thresh = [0.5]*y_labels.shape[1]
    best_thresh_CM = [0]*y_labels.shape[1]

    weights_file = 'weights.csv'
    sinus_rhythm = set(['426783006'])
    classes, weights = load_weights(weights_file)

    labels
    binary_outputs


    for i in range(y_labels.shape[1]):
        thresh = 0
        increment = 1e-2
        y = y_labels[:, i]
        while thresh < 1:
            thresh += increment

            y_pred = np.where(y_hat[:, i] > thresh, 1, 0)


            challenge_metric = compute_challenge_metric(weights, y_labels, binary_outputs, classes, sinus_rhythm)

            # If new F1 score is better than previous then update threshold
            if challenge_metric > best_thresh_CM[i]:
                best_thresh_CM[i] = challenge_metric
                best_thresh[i] = thresh

        print('Challenge Metric: ', challenge_metric)
    print('Challenge Metric on Validation Set:', challenge_metric)
    return best_thresh





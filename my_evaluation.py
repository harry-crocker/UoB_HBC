from evaluate_model import *
# from team_code import save_object
import dill

import sys


def my_compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(   np.sum(np.any(  (labels[i, :], outputs[i, :]), axis=0)  ), 1   ))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        if j==k or not (labels[i, k] and outputs[i, j]):
                            A[j, k] += 1.0/normalization
    return A


def compute_big_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(   np.sum(np.any(  (labels[i, :], outputs[i, :]), axis=0)  ), 1   ))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1  #.0/normalization
    # print(A)
    return A


def run_evaluation(label_directory, output_directory, workspace):
    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    weights_file = 'weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    # Load the scored classes and the weights for the Challenge metric.
    print('Loading weights...')
    classes, weights = load_weights(weights_file, equivalent_classes)

    # Load the label and output files.
    print('Loading label and output files...')
    label_files, output_files = find_challenge_files(label_directory, output_directory)
    labels = load_labels(label_files, classes, equivalent_classes)
    binary_outputs, scalar_outputs = load_outputs(output_files, classes, equivalent_classes)


    A = compute_confusion_matrices(labels, binary_outputs)
    np.save(workspace+'/normal_confusion_matrix', A)

    A = compute_big_confusion_matrix(labels, binary_outputs)
    np.save(workspace+'/big_confusion_matrix', A)

    save_object(classes, workspace+'/classes')


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        dill.dump(obj, output)


if __name__ == '__main__':
    label_directory = sys.argv[1]
    output_directory = sys.argv[2]
    workspace = sys.argv[3]
    print(workspace)

    if not os.path.isdir(workspace):
        os.mkdir(workspace)

    run_evaluation(label_directory, output_directory, workspace)

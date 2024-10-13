import os, math, multiprocessing
from os.path import join
from copy import copy
from skimage.io import imread
import opts
from copy import deepcopy
import numpy as np
from PIL import Image
from pathlib import Path
import imageio

from visual_words import extract_filter_responses, get_visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    
    hist, _ = np.histogram(wordmap, bins=np.arange(K+1), range=(0, K))
    
    
    hist = hist.astype(float)
    hist_sum = hist.sum()
    
    if hist_sum > 0:
        hist /= hist_sum
    
    return hist
    

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    

    H, W = wordmap.shape

    def compute_histograms(layer_wordmap, num_cells):
        '''
        Compute histograms for a given layer.
        
        [input]
        * layer_wordmap: numpy.ndarray of shape (H/L, W/L), the downscaled wordmap for the layer
        * num_cells: number of cells in the layer
        
        [output]
        * layer_hist: numpy.ndarray of shape (K*num_cells), histogram for the layer
        '''
        hist, _ = np.histogram(layer_wordmap, bins=np.arange(K+1), range=(0, K))
        return hist

    hist_layers = []
    num_cells = 1  # Start with 1 cell for the finest layer

    for layer in range(L + 1):
        # Compute the wordmap for this layer
        layer_wordmap = wordmap[::2**layer, ::2**layer]
        hist_layer = compute_histograms(layer_wordmap, num_cells)
        hist_layers.append(hist_layer)
        num_cells *= 4  # Increase the number of cells for the next layer

    # Concatenate histograms
    hist_all = np.concatenate(hist_layers)

    # Weights for each layer
    weights = np.array([2**(-L) if i == 0 else 2**(-L) for i in range(L+1)])
    weighted_hist = np.zeros_like(hist_all)
    
    start_idx = 0
    num_cells = 1
    for layer in range(L + 1):
        end_idx = start_idx + K * num_cells
        weighted_hist[start_idx:end_idx] = hist_layers[layer] * weights[layer]
        start_idx = end_idx
        num_cells *= 4

    # Normalize the final histogram
    hist_sum = weighted_hist.sum()
    if hist_sum > 0:
        hist_all = weighted_hist / hist_sum

    return hist_all

    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)

    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    # ----- RAW FILE PATH INSERTION BLOCK -----
    # MAKE SURE TO USE A RAW STRING FOR THE FILE PATH
    # EXAMPLE: r"C:\path\to\your\file.txt" OR convert_path FUNCTION
    
    

    img = imageio.imread(str(img_path))

    wordmap = get_visual_words(opts, img, dictionary)

    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    
    return feature
    

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''
    def convert_path(filepath):
    # Convert the Path object or string path to a string
        filepath_str = str(filepath)
    
    # Replace backslashes with forward slashes
        filepath_str = filepath_str.replace('\\', '/')
    
    # Return the modified string path (can use Path(filepath_str) if needed)
        return filepath_str

    data_dir = Path('C:/Users/abhis/OneDrive/Desktop/Abhishek/Computer Vision/HW1/data/')
    out_dir = Path('C:/Users/abhis/OneDrive/Desktop/Abhishek/Computer Vision/HW1/')
    SPM_layer_num = opts.L

    # ----- RAW FILE PATH INSERTION BLOCK -----
    # ENSURE THAT THE FILEPATH TO THE TRAIN FILE IS PROVIDED IN RAW STRING FORMAT
    # EXAMPLE: r"C:\path\to\train_files.txt"
    
    train_files_path = data_dir / 'train_files.txt'
    train_labels_path = data_dir / 'train_labels.txt'
    dictionary_path = out_dir / 'dictionary.npy'

    train_files_path = convert_path(train_files_path)
    train_labels_path = convert_path(train_labels_path)
    dictionary_path = convert_path(dictionary_path)

    print(f"train_files_path: {train_files_path}")
    print(f"train_labels_path: {train_labels_path}")
    print(f"dictionary_path: {dictionary_path}")


# Check file existence
    # if train_files_path.exists() and train_labels_path.exists() and dictionary_path.exists():
    train_files = train_files_path.read_text().splitlines()
    train_labels = train_labels_path.read_text().splitlines()
    dictionary = np.load(dictionary_path)
    #else:
    #    print('Error at lines 159-167')
    # train_files = r"C:/Users/abhis/OneDrive/Desktop/Abhishek/Computer Vision/HW1/data/train_files.txt"
    # train_labels = r"C:/Users/abhis/OneDrive/Desktop/Abhishek/Computer Vision/HW1/data/train_labels.txt"
    # dictionary = r"C:/Users/abhis/OneDrive/Desktop/Abhishek/Computer Vision/HW1/code/dictionary.npy"

    # ----- TODO -----

    features = []
    
    # Extract features for each training image
    for img_path in train_files:
        feature = get_image_feature(opts, img_path, dictionary)
        features.append(feature)

    # Convert features list to numpy array
    features = np.array(features)

    # Example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    similarity = np.minimum(word_hist, histograms).sum(axis=1)
    
    # Compute distance as one minus similarity
    hist_dist = 1.0 - similarity
    
    return hist_dist

    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = r'C:/Users/abhis/OneDrive/Desktop/Abhishek/Computer Vision/HW1/data'
    out_dir = r'C:/Users/abhis/OneDrive/Desktop/Abhishek/Computer Vision/HW1'

    # ----- RAW FILE PATH INSERTION BLOCK -----
    # USE RAW STRING FORMAT FOR FILE PATHS WHEN LOADING THE TRAINED SYSTEM OR READING DATA
    # EXAMPLE: r"C:\path\to\trained_system.npz"

    trained_system_path = Path('C:/Users/abhis/OneDrive/Desktop/Abhishek/Computer Vision/HW1/code/trained_system.npz')

    if trained_system_path.exists():
        trained_system = np.load(trained_system_path)
        dictionary = trained_system['dictionary']
        train_features = trained_system['features']
        train_labels = trained_system['labels']
    else:
        print('Error at lines 235')
    dictionary = trained_system['dictionary']
    train_features = trained_system['features']
    train_labels = trained_system['labels']

    # Using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_file_dir = r"C:/Users/abhis/OneDrive/Desktop/Abhishek/Computer Vision/HW1/data"
    test_label_dir = r'C:/Users/abhis/OneDrive/Desktop/Abhishek/Computer Vision/HW1/data'

    test_files_path = Path(test_file_dir) / 'test_files.txt'
    test_labels_path = Path(test_label_dir) / 'test_labels.txt'

    if test_files_path.exists() and test_labels_path.exists():
        test_files = test_files_path.read_text().splitlines()
        test_labels = test_labels_path.read_text().splitlines()
    else:
        print('Error at lines 253-261')
    # ----- TODO -----

    num_classes = 8
    conf = np.zeros((num_classes, num_classes), dtype=np.int32)

    # Function to compute distances between test and training features
    def compute_distances(test_feature):
        return distance_to_set(test_feature, train_features)
    
    # Evaluate each test image
    for i, img_path in enumerate(test_files):
        # Extract features for the test image
        test_feature = get_image_feature(test_opts, img_path, dictionary)
        
        # Compute distances to all training features
        distances = compute_distances(test_feature)
        
        # Find the closest training image
        predicted_label = train_labels[np.argmin(distances)]
        actual_label = test_labels[i]
        
        # Update confusion matrix
        conf[actual_label, predicted_label] += 1

    # Calculate accuracy
    accuracy = np.trace(conf) / np.sum(conf)

    return conf, accuracy

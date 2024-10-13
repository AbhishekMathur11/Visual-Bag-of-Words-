import os, multiprocessing
from os.path import join, isfile
import opts
import numpy as np
from PIL import Image
import scipy.ndimage as scnd
from skimage import color
from skimage.io import imread
from sklearn.cluster import KMeans
import glob
from scipy.spatial.distance import cdist


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    # ----- TODO -----
    
    if img.dtype != np.float32:
        img.astype('float32') / 255.0
    
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)
    
    img_lab = color.rgb2lab(img)
    print(f"Shape of img_lab: {img_lab.shape}")
    img_lab = np.squeeze(img_lab)

    if len(img_lab.shape) != 3 or img_lab.shape[2] != 3:
        raise ValueError(f"Unexpected shape for img_lab: {img_lab.shape}")

    H, W, _ = img_lab.shape
    F = len(opts.filter_scales) * 4
    filter_responses = np.zeros((H, W, 3*F), dtype = np.float32)

    index = 0

    for scale in opts.filter_scales:

        for i in range(3):
            filter_responses[:, :, index] = scnd.gaussian_filter(img_lab[:, :, i], sigma=scale)
            index += 1

        for i in range(3):
            filter_responses[:, :, index] = scnd.gaussian_laplace(img_lab[:, :, i], sigma=scale)
            index += 1
        for i in range(3):
            dx = np.array([[1,0,-1], [1,0,-1], [1,0,-1]])
            filter_responses[:, :, index] = scnd.gaussian_filter(img_lab[:, :,i],sigma=scale, order=(0,1))
            index += 1

        for i in range(3):
            dy = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])
            filter_responses[:,:, index] = scnd.gaussian_filter(img_lab[:, :, i], sigma=scale, order=(1,0))
            index += 1
    return filter_responses

def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    try:
        opts, image_path = args
        print(f"Processing image: {image_path}")

        img = imread(image_path)
        print(f"Image loaded successfully: {image_path}")

        filter_responses = extract_filter_responses(opts, img)
        print(f"Extracted filter responses for image: {image_path}")

        H, W, _= filter_responses.shape
        alpha = opts.alpha

        total_pixels = H * W
        sampled_indices = np.random.choice(total_pixels, min(alpha, total_pixels), replace=False)
        sampled_responses = filter_responses.reshape(-1, filter_responses.shape[2])[sampled_indices, :]

        print(f"Sampled filter responses from image: {image_path}")

        temp_filename = os.path.basename(image_path).replace('.jpg', '.npy')
        temp_filepath = join(opts.feat_dir, temp_filename)

       
        if not os.path.exists(opts.feat_dir):
            os.makedirs(opts.feat_dir)
            print(f"Created directory: {opts.feat_dir}")

        
        np.save(temp_filepath, sampled_responses)
        print(f"Saved sampled responses to: {temp_filepath}")

        
        if not os.path.exists(temp_filepath):
            print(f"Error: File not found after saving: {temp_filepath}")

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def compute_dictionary(opts, n_worker):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''
    print("Starting dictionary computation...") 
    data_dir = r'C:/Users/abhis/OneDrive/Desktop/Abhishek/Computer Vision/HW1/data'
    feat_dir = r'C:/Users/abhis/OneDrive/Desktop/Abhishek/Computer Vision/HW1/feat'
    out_dir = r'C:/Users/abhis/OneDrive/Desktop/Abhishek/Computer Vision/HW1'
    K = opts.K

    train_files = open(os.path.join(data_dir, 'train_files.txt')).read().splitlines()

    print(f"Number of training files: {len(train_files)}")
    # ----- TODO -----
    pool = multiprocessing.Pool(processes=n_worker)
    args = [(opts, join(data_dir, file)) for file in train_files]
    pool.map(compute_dictionary_one_image, args)
    pool.close()
    pool.join()

    filter_responses = []
    print("Starting k-means clustering...")

    for file in train_files:
        temp_filename = os.path.basename(file).replace('.jpg', '.npy')
        temp_filepath = join(feat_dir, temp_filename)
        try:
            responses = np.load(temp_filepath)
            filter_responses.append(responses)
        except FileNotFoundError:
            print(f"Temporary file not found: {temp_filepath}")
        except Exception as e:
            print(f"Error loading temporary file {temp_filepath}: {e}")

    if len(filter_responses) == 0:
        raise ValueError("No filter responses loaded; check if all images are processed correctly.")

    filter_responses = []

    for file in train_files:
        temp_filename = os.path.basename(file).replace('.jpg', '.npy')
        temp_filepath = join(feat_dir, temp_filename)
        try:
            responses = np.load(temp_filepath)
            filter_responses.append(responses)
        except FileNotFoundError:
            print(f"Temporary file not found: {temp_filepath}")
        except Exception as e:
            print(f"Error loading temporary file {temp_filepath}: {e}")

    if len(filter_responses) == 0:
        raise ValueError("No filter responses loaded; check if all images are processed correctly.")

    filter_responses = np.vstack(filter_responses)
    print(filter_responses.shape)
    
    print("Starting k-means clustering...")

    kmeans = KMeans(n_clusters=35, n_init=10, random_state=0) 

    dictionary = kmeans.fit(filter_responses).cluster_centers_
    print("K-means clustering completed.")

    np.save(join(out_dir, 'dictionary.npy'), dictionary)
    print(f"Dictionary saved at {join(out_dir, 'dictionary.npy')}")
    

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    filter_responses = extract_filter_responses(opts, img)


    H, W, F = filter_responses.shape
    print(filter_responses.shape)
    print(dictionary.shape)

    filter_responses = filter_responses.reshape(H*W, F)

    distances = cdist(filter_responses, dictionary, metric = 'euclidean')

    wordmap = np.argmin(distances, axis = 1)

    wordmap = wordmap.reshape(H, W)

    return wordmap


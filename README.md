# Visual-Bag-of-Words-Scene-Detection

This project implements a scene classification system using the bag-of-visual-words approach with spatial pyramid matching. It can classify images into 8 different scene categories like aquarium, desert, highway, kitchen, etc.
Key Features
Extracts filter responses using a multi-scale filter bank
Builds a visual word dictionary using k-means clustering
Represents images as histograms of visual words
Uses spatial pyramid matching to capture spatial information
Classifies scenes with a nearest neighbor approach
Usage
The main script to run the system is main.py:
text
python main.py

This will:
Extract filter responses from training images
Build the visual word dictionary
Compute features for training and test images
Train the classifier
Evaluate on the test set
Configuration
Hyperparameters can be modified in opts.py, including:
Filter scales
Number of visual words (K)
Number of spatial pyramid levels
etc.
Files
visual_words.py: Functions for extracting visual words
visual_recog.py: Core recognition system
util.py: Utility functions
test.py: For running tests
Results
The confusion matrix and accuracy results are saved to:
confmat.csv
accuracy.txt
Extending
The visual_recog_mk2.py file contains an improved version of the recognition system that can be used to experiment with enhancements.

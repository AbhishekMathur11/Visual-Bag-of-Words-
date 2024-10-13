# Visual Bag-of-Words Scene Detection

## Overview
This project implements a **Visual Bag-of-Words (BoW) model** for scene detection using image feature extraction, clustering, and classification techniques. The system recognizes and classifies scenes based on visual patterns in images.

## Key Features
- **Feature Extraction:** Uses SIFT or ORB for keypoint descriptor extraction.
- **K-Means Clustering:** Clusters features to form a visual vocabulary.
- **Histogram Representation:** Transforms images into histograms of visual words.
- **Scene Classification:** Classifies scenes using the BoW model.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/AbhishekMathur11/Visual-Bag-of-Words-Scene-Detection.git
    cd Visual-Bag-of-Words-Scene-Detection
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How It Works

1. **Feature Extraction:** Keypoints are detected using feature extractors (SIFT/ORB).
2. **Clustering:** K-Means clustering groups similar features into visual words.
3. **Histogram Creation:** Represents each image as a histogram of visual word occurrences.
4. **Classification:** Machine learning model is trained on these histograms to classify scenes.

## Usage

To train and test the model, run the following:
```bash
python main.py --train_path <path_to_training_data> --test_path <path_to_testing_data>

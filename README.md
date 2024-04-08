# Image Feature Extraction and Dimensionality Reduction

## Overview
This Jupyter notebook demonstrates feature extraction and dimensionality reduction techniques for a collection of images. It extracts features using various methods such as color histogram, Local Binary Patterns (LBP), Histogram of Oriented Gradients (HOG), and features from a pre-trained VGG16 convolutional neural network. The extracted features are then combined and reduced using Principal Component Analysis (PCA).

## Prerequisites
- Python 3.x
- Jupyter Notebook environment (or Google Colab)
- Libraries:
  - Pillow (PIL)
  - OpenCV (cv2)
  - tqdm
  - NumPy
  - scikit-image
  - TensorFlow (for VGG16)
  - scikit-learn

## Usage
1. Clone or download the notebook file (`Feature_extraction_and_dimensionality_reduction.ipynb`).
2. Open the notebook in a Jupyter Notebook environment.
3. Ensure the necessary libraries are installed.
4. Execute each cell in the notebook to run the code sequentially.
5. Follow the instructions/comments provided within the notebook.
6. Customize parameters and methods as needed.

## Features Extraction Methods
1. **Color Histogram**:
   - Calculates histogram features for each color channel (B, G, R) of the image.
   
2. **Local Binary Patterns (LBP)**:
   - Computes LBP features using the uniform method.

3. **Histogram of Oriented Gradients (HOG)**:
   - Extracts HOG features using a specified set of parameters.

4. **VGG16 Features**:
   - Utilizes a pre-trained VGG16 model to extract features from a specified layer.

## Dimensionality Reduction
- **Principal Component Analysis (PCA)**:
  - Performs dimensionality reduction on the combined feature matrix.
  - Allows customization of the number of components.

## Output
- The notebook provides insights into the extracted features and the dimensionality reduction process through printed statistics and visualizations.

## Acknowledgments
- The notebook utilizes pre-trained models and techniques from various libraries, including OpenCV, scikit-image, TensorFlow, and scikit-learn.

## License
This project is licensed under the [MIT License](LICENSE).

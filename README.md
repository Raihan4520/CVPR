# Computer Vision and Pattern Recognition (CVPR) Course

This repository contains Python implementations and Jupyter Notebooks from the **Computer Vision and Pattern Recognition (CVPR)** course at American International University - Bangladesh (AIUB). It includes various projects and exercises demonstrating key concepts in computer vision, such as image processing, feature detection, and machine learning for vision tasks.

## Table of Contents
- [Overview](#overview)
- [Highlighted Projects](#highlighted-projects)
  - [Face Mask Detection](#face-mask-detection)
  - [Emotion Detection](#emotion-detection)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Contact](#contact)

## Overview

This repository focuses on practical implementations of computer vision algorithms using **Python** and **Jupyter Notebooks**. It includes multiple projects, with a particular emphasis on real-world applications such as detecting face masks and recognizing emotions from facial expressions. The repository also includes foundational exercises in image processing, feature extraction, and object detection, which are key components of computer vision systems.

### Course Information
For more details on the course, refer to the [AIUB Undergraduate Course Catalog](https://www.aiub.edu/faculties/fst/ug-course-catalog).  
*Note: Search for "Computer Vision & Pattern Recognition" for specific course information.*

## Highlighted Projects

### Face Mask Detection

- **Description:** A project aimed at detecting whether individuals in an image or video feed are wearing a face mask. This project uses convolutional neural networks (CNNs) to classify images into two categories: "Mask" and "No Mask."
- **Key Concepts:**
  - Preprocessing images for model training.
  - CNN-based architecture for classification.
  - Real-time detection using OpenCV.
- **Technologies Used:** TensorFlow/Keras, OpenCV, Python.

### Emotion Detection

- **Description:** This project detects emotions (such as happiness, sadness, anger, surprise, etc.) from facial expressions using a deep learning model. The model is trained on a dataset of labeled facial expressions to classify the emotion displayed by a person in an image.
- **Key Concepts:**
  - Image preprocessing (data augmentation, resizing).
  - Transfer learning (Pre trained model: MobileNet).
  - CNN architecture for emotion classification.
  - Visualization of results using Matplotlib.
- **Technologies Used:** TensorFlow/Keras, OpenCV, Python.

## Technologies Used

- **Programming Language:** Python
- **Libraries and Tools:**
  - **OpenCV:** For image processing and real-time detection.
  - **TensorFlow/Keras:** For building deep learning models.
  - **NumPy and Pandas:** For data manipulation and processing.
  - **Matplotlib and Seaborn:** For visualizing data and model performance.
  - **Jupyter Notebook:** For running and documenting code interactively.

## How to Run

To run the code in this repository, follow these steps:

### Prerequisites

- Install Python 3.x and Jupyter Notebook.
- Install the required libraries by running the following command:
  ```bash
  pip install <library_name>

### Running the Notebooks
1. Clone the repository:
   ```bash
   git clone https://github.com/Raihan4520/CVPR.git
2. Open Jupyter Notebook:
   ```bash
   jupyter notebook
3. Navigate to the desired notebook (e.g., Face_Mask_Detection.ipynb or Emotion_Detection.ipynb) and run the cells interactively.
4. Follow the instructions within each notebook to provide the required input (e.g., images or video feed) and view the results.

### Example
To run the Face Mask Detection project:

1. Open Face_Mask_Detection.ipynb in Jupyter Notebook.
2. Ensure that the dataset is available in the specified folder or update the file paths in the notebook.
3. Run all cells to train the model or use a pre-trained model to detect face masks in real-time.

## Contact

If you have any questions or suggestions, feel free to reach out through the repository's issues or contact me directly.

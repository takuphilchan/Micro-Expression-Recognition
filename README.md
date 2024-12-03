# 3D CNN Micro Expression Recognition Algorithm

This repository contains a 3D Convolutional Neural Network (3D CNN) implementation for micro expression recognition. The algorithm processes video datasets, performs recognition tasks, and offers both GUI and code-based methods for training and testing.

## Folder Structure

### `data_processing_gui/`
The `data_processing_gui` folder provides a **Graphical User Interface (GUI)** designed for processing video datasets and testing the micro expression recognition algorithm. It allows users to:
- Process and preprocess video files into usable formats.
- Dynamically select frames for training and evaluation.
- Test the algorithm on processed datasets with ease.

The GUI simplifies dataset management and testing, offering an interactive experience for users without requiring direct coding.

### `training/`
The `training` folder contains two approaches for training the 3D CNN model:

1. **Google Colab Notebook**  
   - The Colab notebook provides an easy-to-use interface for training the model in the cloud without needing a local setup.
   - You can run the notebook to train the model, monitor the progress, and evaluate its performance.
   - This method is recommended for users with limited hardware resources or those who prefer not to manage a local setup.

2. **Python Files**  
   - The Python files in this folder allow for local model training and fine-tuning. 
   - You can directly run Python scripts to train the model on your local machine or server.
   - This method is suitable for users who prefer more control over the training process and have access to adequate hardware resources.

## Requirements

Before running the code, ensure you have the necessary dependencies installed. You can install them by running:

```bash
pip install -r requirements.txt

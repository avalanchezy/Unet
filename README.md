# Continual Learning for Medical Image Segmentation

Welcome to the repository for our project "Continual Learning for Medical Image Segmentation," where we focus on developing and refining deep learning models capable of segmenting medical images with increasing accuracy over time. The CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation) dataset served as the starting point for training our initial U-Net model. We are poised to incorporate a second dataset (SUMAC) to facilitate continuous learning and model improvement.

## Overview

The objective of this project is to establish a robust machine learning framework that can continually learn from new data, thereby improving its performance in segmenting medical images. By employing continual learning strategies, our models are designed to adapt to new patterns in data without forgetting previously learned information.

## Project Origin

This project is inspired by and builds upon the work found at [CAMUS](https://github.com/creatis-myriad/camus-hands-on.git) by team Myriad of CREATIS Laboratory. Our approach extends the methodologies and techniques presented there, focusing on continual learning and adaptation to new datasets in medical image segmentation.

## System Requirements

To run the models and training scripts in this repository, you will need a system with the following:

- [CUDA 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive)
- [cuDNN 8.1](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.1.33/11.2_20210301/cudnn-11.2-windows-x64-v8.1.1.33.zip)

Please ensure that you have these versions installed and properly configured on your system. They are essential for leveraging GPU acceleration with TensorFlow.

## TensorFlow Environment

The code in this repository is developed for TensorFlow. It is recommended to use a virtual environment for Python to manage dependencies. After setting up CUDA and cuDNN, install the required Python packages using:

- Python 3.10 in a conda environment

## Repository Contents

- `Segmentation_TF2_Unet.ipynb`: A Jupyter notebook containing the pipeline for training our first U-Net model on the CAMUS dataset.
- `Segmentation_TF2_test_on_new.ipynb`: A Jupyter notebook testing performance of the model on the SUMAC dataset.
- `Segmentation_TF2_test_on_old.ipynb`: A Jupyter notebook testing performance of the model on the CAMUS dataset.
- `models/`: Directories containing the trained model weights and architectures for deployment and testing.
- `utils/`: Utility scripts for data handling, augmentation, and metric calculations.
- `results/`: Evaluation results and comparison plots showcasing the model's segmentation performance on various datasets.
- `requirements.txt`: A list of Python dependencies required to run the projects in this repository.

## Model Performance

The initial U-Net model was trained on the CAMUS dataset. As we expand our dataset collection for continuous learning, we aim to refine our models to achieve better and more generalized performance.

## Getting Started

To begin working with the models in this repository:

```bash
git clone https://github.com/creatis-myriad/camus-hands-on.git
git clone https://github.com/avalanchezy/Unet.git
```
Use the Keras folder and Jupyter notebook file mentioned in this Repo to replace the original file.

Please download [the trained model](https://drive.google.com/file/d/13mlaFQDDrcIzwMrgFWHL02xHTaNZwtVM/view?usp=sharing).

## Contribution

We welcome contributions that help advance the project, whether by improving existing models, adding new datasets, or implementing continual learning methods. Feel free to fork the repository, make your enhancements, and create a pull request.

## License

This project is open-sourced under the MIT License. See the LICENSE file for full details.

## Contact

For questions or feedback, please open an issue in this repository, and we will address it promptly.



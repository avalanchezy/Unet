# Continuous Learning for Medical Image Segmentation

Welcome to the repository for our project "Continuous Learning for Medical Image Segmentation," where we focus on developing and refining deep learning models capable of segmenting medical images with increasing accuracy over time. The CAMUS (Cardiac Acquisitions for Multi-structure Ultrasound Segmentation) dataset served as the starting point for training our initial U-Net model. We are poised to incorporate a second dataset to facilitate continuous learning and model improvement.

## Overview

The objective of this project is to establish a robust machine learning framework that can continually learn from new data, thereby improving its performance in segmenting medical images. By employing continuous learning strategies, our models are designed to adapt to new patterns in data without forgetting previously learned information.

## System Requirements

To run the models and training scripts in this repository, you will need a system with the following:

- CUDA 11.8
- cuDNN 8.1

Please ensure that you have these versions installed and properly configured on your system. They are essential for leveraging GPU acceleration with TensorFlow.

## TensorFlow Environment

The code in this repository is developed for TensorFlow. It is recommended to use a virtual environment for Python to manage dependencies. After setting up CUDA and cuDNN, install the required Python packages using:

## Repository Contents

- `Segmentation_TF2_Unet.ipynb`: A Jupyter notebook containing the pipeline for training our first U-Net model on the CAMUS dataset.
- `models/`: Directories containing the trained model weights and architectures for deployment and testing.
- `utils/`: Utility scripts for data handling, augmentation, and metric calculations.
- `results/`: Evaluation results and comparison plots showcasing the model's segmentation performance on various datasets.
- `requirements.txt`: A list of Python dependencies required to run the projects in this repository.

## Model Performance

The initial U-Net model was trained on the CAMUS dataset. As we expand our dataset collection for continuous learning, we aim to refine our models to achieve better and more generalized performance.

## Getting Started

To begin working with the models in this repository:

```bash
git clone https://github.com/avalanchezy/Unet/blob/main/Segmentation_TF2_Unet.ipynb
cd continuous-learning-medical-segmentation
pip install -r requirements.txt
```

## Contribution

We welcome contributions that help advance the project, whether by improving existing models, adding new datasets, or implementing continuous learning methods. Feel free to fork the repository, make your enhancements, and create a pull request.

## License

This project is open-sourced under the MIT License. See the LICENSE file for full details.

## Contact

For questions or feedback, please open an issue in this repository, and we will address it promptly.



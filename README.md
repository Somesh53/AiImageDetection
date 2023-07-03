# AI-Generated Image Detection

## Problem Statement
With the advent of generative AI, it has become increasingly difficult to separate real data from AI-generated content. The goal of this project is to develop a model that can accurately identify fake photos created by AI. The model aims to contribute to the development of more secure and reliable online services by detecting and mitigating identity fraud.

## Dataset
The dataset used for training and evaluation is the "140k Real and Fake Faces" dataset, which can be found [here](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces). Due to limitations in computational power working with Images, a subset of 20,000 images from the test data of the above data set was used for this project. The dataset contains both real and fake images, and each image is labelled accordingly.

## Solution
The project was implemented using Python and the Keras deep learning library. The solution consists of the following steps:

1. Data Preparation:
   - The dataset was loaded from the provided CSV file, which contains image paths and corresponding labels.
   - Images were resized to a standard size of 224x224 pixels.
   - Images and labels were stored in arrays for further processing.
   - This was a challenging task because I had to take Images from one folder and use it corresponding to the label in the CSV file.

2. Model Architecture:
   - The VGG16 pre-trained model was used as the base model, with the pre-trained weights initialized from the "ImageNet" dataset.
   - The last few layers of the VGG16 model were modified by adding global average pooling and fully connected layers.
   - The final layer was designed for binary classification, as the task is to distinguish between real and fake images.

3. Model Training:
   - The dataset was split into training, validation, and testing sets.
   - The model was trained using the training set and evaluated on the validation set.
   - The Adam optimizer and binary cross-entropy loss function were used during training.
   - The model was trained for 10 epochs with a batch size of 32.

4. Model Evaluation:
   - The trained model was used to predict labels for the test set.
   - F1 score, a metric commonly used in binary classification tasks, was calculated to evaluate the model's performance.

## Results
After trying different combinations of batch size and complexity of fully connected layers, my model achieved an F1 score of 0.808 on the test set, indicating its ability to accurately detect AI-generated images.

## Dependencies
The following dependencies are required to run the project:
- Python (version 3.11.4)
- Keras (version 2.12.0)
- pandas (version 2.0.2)
- numpy (version 1.23.5)
- matplotlib (version 3.7.1)
- scikit-learn (version 1.2.2)
- PIL (version 9.5.0)


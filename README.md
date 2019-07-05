Deep Learning with TensorFlow - Tutorials
================
Codes courtesy from a course by deeplearning.ai, modified by Tran Nguyen

-   [DEEP LEARNING WITH TENSORFLOW](#deep-learning-with-tensorflow)

DEEP LEARNING WITH TENSORFLOW
-----------------------------

Quick notes from the course + codes to run in Mac terminal. If you want to learn more about TensorFlow, check out the great course ""Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning" by deeplearning.ai on Coursera.

The materials below were generated based on the tutorials from the course, with some modifications by me.

#### 1. Simple neural network

-   Codes: TF\_HelloWorld.py
-   What you will learn: (i) Creating a simple neural network using tensorflow to predict a simple y = ax + b function.

#### 2. Image classifier: Fashion dataset

-   Codes: TF\_fashion\_mnist.py, data is in the package keras.datasets.fashion\_mnist
-   What you will learn: (i) Creating a neural network to classify 10 different classes of fashion data (clothes, shoes, bags, etc.) (ii) Using callback to stop training when the accuracy reaches certain threshold. (iii) Using convolution for better training. (iv) Visualizing the effect of convolution on training.

#### 3. Basic ideas about Convolution

-   Codes: Convolutions\_basic.py
-   What you will learn: (i) Basic idea about convolution. (ii) Display images using the matplotlib.pyplot package. (iii) Trying different filters for convolution.

#### 4. The neural network that recognizes handwriting digit

-   Codes: TF\_hand\_writing\_recognition.py, data is in the package keras.datasets.mnist
-   What you will learn: Practicing all the concepts in session (3) with the handwriting data.

#### 5. Image classifier: Horse versus human

-   Codes: TF\_Human\_horse.py, data: `data/horse-or-human-data` folder
-   An image classifier to distinguish human or horse image using Convolutional Neural Networks
-   What you will learn: (i) Two-class classification problem. (ii) Training with more complex data (images 300x300 with 3 bytes color) (iii) Using ImageDataGenerator to process the data

#### 6. Image classifier: Happy versus sad faces

-   Codes: TF\_HappyVsSad\_Face.py, data: `data/happy-or-sad-data` folder
-   An image classifier to distinguish sad and happy icons with the accuracy greater than 99.9%
-   What you will learn: Practicing all the concepts in session (5) with the happy-or-sad face data.

#### 7. TF\_Saving\_Loading\_Model.py

-   Codes: TF\_Saving\_Loading\_Model.py,
-   What you will learn: How to save and load the model using the h5py package

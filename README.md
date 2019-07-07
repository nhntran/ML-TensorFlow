Deep Learning with TensorFlow - Tutorials
================
Codes courtesy from TensorFlow in Practice Specialization by deeplearning.ai on Coursera, modified by Tran Nguyen

-   [BASIC ABOUT DEEP LEARNING](#basic-about-deep-learning)
-   [MORE ON CONVOLUTIONAL NEURAL NETWORKS](#more-on-convolutional-neural-networks)

Quick notes from the courses + codes and data to run in Mac terminal. If you want to learn more about TensorFlow, check out the great courses in the "TensorFlow in Practice Specialization" by deeplearning.ai on Coursera.

The codes work well with TensorFlow 2.0

``` bash
pip install tensorflow==2.0.0-alpha0
```

BASIC ABOUT DEEP LEARNING
-------------------------

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

MORE ON CONVOLUTIONAL NEURAL NETWORKS
-------------------------------------

#### 8. Image classifier: Dogs versus Cats (Kaggle dataset)

-   Codes: TF\_Dogs\_Cats.py, data: `data/dog-or-cat-data` folder
-   An image classifier to distinguish dog and cat
-   What you will learn: (i) Practicing again all the concepts above. (ii) Plotting the training/validation accuracy and loss to evaluate the model. (iii) Learning about overfitting based on the training/validation accuracy and loss plots.

<img src="./img/training_validation_acc_loss_plot.png" width="50%" height="50%" style="display: block; margin: auto;" />

#### 9. Visualizing Intermediate Representations

-   Codes: TF\_Dogs\_Cats\_Visualization.py, need to run the TF\_Dogs\_Cats.py file first to obtain the model.
-   What you will learn: See how input gets transformed when going through the convnet =&gt; investigating features that the convnet has learned.
-   Output interpretation: Each ouput is generated from a random image in the set. Each colum: an image of a specific filter. Each row is the output of a layer.

<img src="./img/dog_intermediate representation.png" width="1171" style="display: block; margin: auto;" />

"As you can see we go from the raw pixels of the images to increasingly abstract and compact representations. The representations downstream start highlighting what the network pays attention to, and they show fewer and fewer features being "activated"; most are set to zero. This is called "sparsity." Representation sparsity is a key feature of deep learning.

These representations carry increasingly less information about the original pixels of the image, but increasingly refined information about the class of the image. You can think of a convnet (or a deep network in general) as an information distillation pipeline." (Note from the Course notebook <https://colab.sandbox.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%202%20-%20Lesson%202%20-%20Notebook.ipynb>)

#### 10. Data Processing:

-   Codes: TF\_DataProcessing\_for\_CNN\_DogCatDataset.py, the zip file can be downloaded via the link in the codes.
-   What you will learn: Randomly splitting the data into training and testing dataset =&gt; generate suitable input for ImageGenerator


 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-

### Learning TensorFlow with Laurence Moroney, Google Brain on Coursera
### /Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/\
###/MachineLearning/TensorFlow/TensorFlowCodes

### import modules used here -- sys is a very standard one
import tensorflow as tf
import matplotlib.pyplot as plt


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.4):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training=True

def fashion_mnist_basic():
    ##load the data from keras
    mnist = tf.keras.datasets.fashion_mnist
    
    ##getting training and test data
    (training_images, training_labels),(test_images, test_labels)\
    = mnist.load_data()
    
    ##checking the data by visualization using matplotlib
    #plt.imshow(training_images[59])
    #print(training_labels[59])
    #print(training_images[59])
    #print(len(training_labels), len(training_images))
    #print(training_images[59])
    
    ##normalize the data
    training_images = training_images / 255.0
    test_images = test_images / 255.0

    ##create a model
    #Only 2 layers. For this simple data, adding more layer is not necessary
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),\
        tf.keras.layers.Dense(128, activation=tf.nn.relu),\
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)])
    #Flatten: the first layer in nn should be the same shape as the data.
    #The data:28x28 images, and 28 layers of 28 neurons would be infeasible,
    # =>'flatten' that 28,28 into a 784x1. 
    #=> add the Flatten() layer at the begining, 
    #when the arrays are loaded into the model => automatically be flattened

    #Dense: add a layer of neurons, number of neurons=128
    #128 dense layers
    #=> Increase the number of neurons: 512, 1024, ... 
    #=> Training takes longer, but it more accurate
    #activation: activation function
    #Relu effectively means "If X>0 return X, else return 0"
    #10: 10 neurons = 10 different classifications the data has

    #Softmax takes a set of values, and effectively picks the biggest one
    #Ex: [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05]
    #=> [0,0,0,0,1,0,0,0,0]

    ## build the model
    model.compile(optimizer = tf.train.AdamOptimizer(),\
        loss= 'sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=10)
    #epochs: number of reiteration, more epochs can make "overfitting"

    ## evaluate the model on the test set
    loss, accuracy = model.evaluate(test_images, test_labels)
    

    ## predict the classification on the test set
    classifications = model.predict(test_images)
    print(classifications[5])
    print(test_labels[5])

    return loss, accuracy

def fashion_mnist_callback():
    callbacks = myCallback()
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels)\
    = mnist.load_data()
    training_images = training_images/255.0
    test_images = test_images/255.0
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation = tf.nn.relu),
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    ])
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy')
    model.fit(training_images, training_labels, epochs=5, callbacks = [callbacks])

def fashion_mnist_convolution():
    
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels)\
    = mnist.load_data()
    #60,000 images, 28x28 size, 1 byte to store

    training_images = training_images.reshape(60000,28,28,1)
    #the first convolution expects a single tensor containing everything, 
    #so instead of 60,000 28x28x1 items in a list, we have a single 4D list 
    #that is 60,000x28x28x1, and the same for the test images. 
    #If you don't do this, you'll get an error when training

    training_images = training_images/255.0
    test_images = test_images.reshape(10000,28,28,1)
    test_images = test_images/255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(3,3), activation = 'relu', 
            input_shape = (28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    #The size of the Convolution, in this case a 3x3 grid
    #The number of convolutions you want to generate: 
    #good to start with something in the order of 32
    #MaxPooling2D(2,2): creates a 2x2 array of pixels, 
    #and picks the biggest one, thus turning 4 pixels into 1. 
    #=> effectively reducing the image by 25%.

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'])
    model.summary()
    #model.summary() to see the size and shape of the network
    model.fit(training_images, training_labels, epochs=5)
    test_loss = model.evaluate(test_images, test_labels)

def convolution_visualizing():
    
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels)\
    = mnist.load_data()
    #60,000 images, 28x28 size, 1 byte to store

    training_images = training_images.reshape(60000,28,28,1)
    #the first convolution expects a single tensor containing everything, 
    #so instead of 60,000 28x28x1 items in a list, we have a single 4D list 
    #that is 60,000x28x28x1, and the same for the test images. 
    #If you don't do this, you'll get an error when training

    training_images = training_images/255.0
    test_images = test_images.reshape(10000,28,28,1)
    test_images = test_images/255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(3,3), activation = 'relu', 
            input_shape = (28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])

    f, axarr = plt.subplots(3,4)
    FIRST_IMAGE=0
    SECOND_IMAGE=7
    THIRD_IMAGE=26
    CONVOLUTION_NUMBER = 1
    from tensorflow.keras import models
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
    for x in range(0,4):
      f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
      axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
      axarr[0,x].grid(False)
      f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
      axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
      axarr[1,x].grid(False)
      f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
      axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
      axarr[2,x].grid(False)

########################################################################
# The main() function
def main():
    
    #print(tf.__version__)
    #loss, accuracy=fashion_mnist_basic()
    #print(loss, accuracy)
    #fashion_mnist_callback()
    #fashion_mnist_convolution()
    convolution_visualizing()

#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()


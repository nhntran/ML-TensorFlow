
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
        if(logs.get('acc')>0.98):
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training=True


def mnist_callback():
    callbacks = myCallback()
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    ##checking the data by visualization using matplotlib
    plt.imshow(x_train[59])
    print("x_train:",x_train[59])
    print("y_train:",y_train[59])
    
    x_train = x_train/255.0
    x_test = x_test/255.0


    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(512, activation = tf.nn.relu),
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    ])
    model.compile(optimizer = 'adam', 
        loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs = 10, callbacks = [callbacks])

    #evaluation
    loss, accuracy = model.evaluate(x_test, y_test)

    #prediction
    classifications = model.predict(x_test)
    print(classifications[3])
    print(y_test[5])

    return loss, accuracy

def mnist_callback_convol():
    callbacks = myCallback()
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    ##checking the data by visualization using matplotlib
    #plt.imshow(x_train[59])
    print(len(x_train))
    print(len(y_train))
    
    x_train = x_train.reshape(60000,28,28,1)
    x_test = x_test.reshape(10000,28,28,1)

    #the first convolution expects a single tensor containing everything,
    #so need to reshape

    x_train = x_train/255.0
    x_test = x_test/255.0


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(3,3), activation = 'relu',
            input_shape = (28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(218, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'softmax')
    ])
    model.compile(optimizer = 'adam', 
        loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, epochs = 5, callbacks = [callbacks])

    #evaluation
    loss, accuracy = model.evaluate(x_test, y_test)

    #prediction
    #classifications = model.predict(x_test)
    return loss, accuracy


########################################################################
# The main() function
def main():
    
    #print(tf.__version__)
    loss, accuracy=mnist_callback_convol()
    print("Loss and accuracy:", loss, accuracy)
    

#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()


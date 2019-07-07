
 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-

### Learning TensorFlow with Laurence Moroney, Google Brain on Coursera
### /Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/\
###/MachineLearning/TensorFlow/TensorFlowCodes

### Dogs vs Cats - Kaggle Problem, the two-class classification problem
# Input: 2,000 images 300x300 with 3 bytes color: input_shape=(300, 300, 3)

#Download the data for the test (around 1k images)
#!wget --no-check-certificate \
#  https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip \
#  -O /tmp/cats_and_dogs_filtered.zip

#Download the real data  (around 25k images)
#!wget --no-check-certificate \
    # "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" \
    # -O "/tmp/cats-and-dogs.zip"

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
#import the RMSprop optimization algorithm
from tensorflow.keras.optimizers import RMSprop
#data processing using ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import h5py

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training=True

def check_display_data(local_path1):
    
    ### checking the data
    train_cat_dir = os.path.join(local_path1 + 'train/cats')
    train_cat_names = os.listdir(train_cat_dir)
    train_dog_dir = os.path.join(local_path1 + 'train/dogs')
    train_dog_names = os.listdir(train_dog_dir)

    validation_cat_dir = os.path.join(local_path1 + 'validation/cats')
    validation_cat_names = os.listdir(train_cat_dir)
    validation_dog_dir = os.path.join(local_path1 + 'validation/dogs')
    validation_dog_names = os.listdir(train_dog_dir)

    print('Total training cat images:',len(train_cat_names))
    print('Total training dog images:', len(train_dog_names))
    print('Total validation cat images:',len(validation_cat_names))
    print('Total validation dog images:', len(validation_dog_names))
    ### display the data
    #need matplotlib
    #Ouput images in a 4x4 configuration
    nrows = 4
    ncols = 4
    #Set up matplotlib figure 
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)
    pic_index = 0
    pic_index +=8
    next_cat_pic = [os.path.join(train_cat_dir,fname)
                        for fname in train_cat_names[pic_index-8:pic_index]]
    next_dog_pic = [os.path.join(train_dog_dir,fname)
                        for fname in train_dog_names[pic_index-8:pic_index]]

    for i, img_path in enumerate(next_cat_pic + next_dog_pic):
        sp = plt.subplot(nrows, ncols, i+1)
        #sp.axis('Off') # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()
    #plt.savefig("myplotdoesnotshowup.png")

def generating_data(local_path1):
    train_datagen = ImageDataGenerator(rescale = 1/255)
    validation_datagen = ImageDataGenerator(rescale = 1/255)

    train_gen = train_datagen.flow_from_directory(
        local_path1 + 'train/',
        target_size = (150,150),
        batch_size = 20, #training image in batch of 128 images
        class_mode = 'binary'
        )

    validation_gen = validation_datagen.flow_from_directory(
        local_path1 + 'validation/',
        target_size = (150,150),
        batch_size = 20, #training image in batch of 32 images
        class_mode = 'binary'
        )
    return train_gen, validation_gen

def building_model(local_path1):
    callbacks = myCallback()
    model = tf.keras.models.Sequential([
        ## 5 levels of convolutions - 1st convolution
        tf.keras.layers.Conv2D(16, (3,3), activation = 'relu',
            input_shape = (150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        ## 2nd convolution
        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        ## 3rd convolution
        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        ## Flatten the result
        tf.keras.layers.Flatten(),
        ## DNN
        tf.keras.layers.Dense(512, activation = 'relu'),
        #output layer (classification)
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    model.summary()

    #compile the model using RMSprop optimization algorithm
    model.compile(loss = 'binary_crossentropy',
                    optimizer = RMSprop(lr = 0.001), #learning rate: 0.001
                    metrics = ['acc'])
    
    train_generator, validation_generator=generating_data(local_path1)

    history = model.fit_generator(
        train_generator, steps_per_epoch = 100,
        epochs = 10, verbose = 2,
        validation_data = validation_generator,
        validation_steps = 50,
        callbacks = [callbacks]
        )
    #verbose=2: Note the values per epoch (loss, accuracy, 
    #validation loss, validation accuracy)
    ## retrieve values
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    #evaluate the model
    model_evaluation(acc, val_acc, loss, val_loss)
    ## save model
    model.save('TF_Dogs_Cats.h5') #after training, saving the model into the .h5 file

    return model

def model_evaluation(acc, val_acc, loss, val_loss):
    epochs = range(len(acc)) #get number of epochs

    #plot training and validation accuracy and loss
    plt.plot(epochs,acc)
    plt.plot(epochs,val_acc)
    plt.title("Training and validation accuracy")

    plt.plot(epochs,loss)
    plt.plot(epochs,val_loss)
    plt.title("Training and validation loss")


def prediction_cat_dog(local_path2, model):
    prediction_dir = os.path.join(local_path2)
    prediction_names = os.listdir(prediction_dir)

    for fn in prediction_names:
     
      img = image.load_img(local_path2+fn, target_size=(150, 150))
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)

      images = np.vstack([x])
      classes = model.predict(images, batch_size=10)
      print(classes[0])

      if classes[0]>0.5:
        print(fn + " is a dog")
      else:
        print(fn + " is a cat")



########################################################################
# The main() function
def main():
    
    #print(tf.__version__)
    local_path1 = 'data/dog-cat-data/cats_and_dogs_filtered/'
    local_path2 = 'data/dog-cat-data/cat_dog_prediction/'

    # First time running: training model
    check_display_data(local_path1)
    model = building_model(local_path1)
    prediction_cat_dog(local_path2, model)

    # Second time running: Loading the model again
    # new_model = tf.keras.models.load_model('TF_Dogs_Cats.h5')
    # prediction_cat_dog(local_path2, new_model)

#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()


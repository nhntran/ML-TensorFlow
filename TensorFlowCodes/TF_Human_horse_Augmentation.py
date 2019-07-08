
 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-

#### *****
# For Image augmentation, everything will stay the same 
# except the ImageDataGenerator process
# check the function "generating_data(local_path1, local_path2)""
#### *****


### Learning TensorFlow with Laurence Moroney, Google Brain on Coursera
### /Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/\
###/MachineLearning/TensorFlow/TensorFlowCodes

### Two-class classification problem: identifying horse versus human
# Input: images 300x300 with 3 bytes color: input_shape=(300, 300, 3)

#Download the data for the test
#!wget --no-check-certificate \
#    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \
#    -O /tmp/horse-or-human.zip

#Download validation data
#!wget --no-check-certificate \
#    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip \
#    -O /tmp/validation-horse-or-human.zip
### import modules used here -- sys is a very standard one

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

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.98):
            print("\nReached 98% accuracy so cancelling training!")
            self.model.stop_training=True

def check_display_data(local_path1, local_path2):
    
    ### checking the data
    train_horse_dir = os.path.join(local_path1 + 'horses')
    train_horse_names = os.listdir(train_horse_dir)
    train_human_dir = os.path.join(local_path1 + 'humans')
    train_human_names = os.listdir(train_human_dir)

    validation_horse_dir = os.path.join(local_path2 + 'horses')
    validation_horse_names = os.listdir(train_horse_dir)
    validation_horse_dir = os.path.join(local_path2 + 'human')
    validation_human_names = os.listdir(train_human_dir)

    print('Total training horse images:',len(train_horse_names))
    print('Total training human images:', len(train_human_names))
    print('Total validation horse images:',len(validation_horse_names))
    print('Total validation human images:', len(validation_human_names))
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
    next_horse_pic = [os.path.join(train_horse_dir,fname)
                        for fname in train_horse_names[pic_index-8:pic_index]]
    next_human_pic = [os.path.join(train_human_dir,fname)
                        for fname in train_human_names[pic_index-8:pic_index]]

    for i, img_path in enumerate(next_horse_pic + next_human_pic):
        sp = plt.subplot(nrows, ncols, i+1)
        #sp.axis('Off') # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()

def generating_data(local_path1, local_path2):

    #train_datagen = ImageDataGenerator(rescale = 1/255)
    validation_datagen = ImageDataGenerator(rescale = 1/255)

    train_gen = train_datagen.flow_from_directory(
        local_path1,
        target_size = (300,300),
        batch_size = 128, #training image in batch of 128 images
        class_mode = 'binary'
        )

    validation_gen = validation_datagen.flow_from_directory(
        local_path2,
        target_size = (300,300),
        batch_size = 32, #training image in batch of 32 images
        class_mode = 'binary'
        )
    return train_gen, validation_gen

def generating_data_augmentation(local_path1, local_path2):
    # Augmentation on the training data only
    #train_datagen = ImageDataGenerator(rescale = 1/255)
    train_datagen = ImageDataGenerator(
        rescale = 1/255,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest')

    validation_datagen = ImageDataGenerator(rescale = 1/255)

    train_gen = train_datagen.flow_from_directory(
        local_path1,
        target_size = (300,300),
        batch_size = 128, #training image in batch of 128 images
        class_mode = 'binary'
        )

    validation_gen = validation_datagen.flow_from_directory(
        local_path2,
        target_size = (300,300),
        batch_size = 32, #training image in batch of 32 images
        class_mode = 'binary'
        )
    return train_gen, validation_gen

def building_model(local_path1, local_path2):
    callbacks = myCallback()
    model = tf.keras.models.Sequential([
        ## 5 levels of convolutions - 1st convolution
        tf.keras.layers.Conv2D(16, (3,3), activation = 'relu',
            input_shape = (300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        ## 2nd convolution
        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        ## 3rd convolution
        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        ## 4th convolution
        tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        ## 5th convolution
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
                    #can also use: lr = 1e-4
                    metrics = ['acc'])
    
    train_generator, validation_generator=generating_data_augmentation(local_path1, local_path2)

    history = model.fit_generator(
        train_generator, steps_per_epoch = 8,
        epochs = 15, verbose = 2,
        validation_data = validation_generator,
        validation_steps = 8,
        callbacks = [callbacks]
        )

    #verbose = 2: If regularization mechanisms are used, verbose is turned on to log acc and loss
    #to avoid overfitting
    
    #evaluate the model
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']   
    model_evaluation(acc, val_acc, loss, val_loss)

    return model

def model_evaluation(acc, val_acc, loss, val_loss):
    epochs = range(len(acc)) #get number of epochs

    #plot training and validation accuracy and loss
    plt.plot(epochs,acc)
    plt.plot(epochs,val_acc)
    plt.title("Training and validation accuracy")
    #Another way of plot
    # plt.plot(epochs,acc, 'r', "Training Accuracy")
    # plt.plot(epochs,val_acc, 'b')
    # plt.title("Training and validation accuracy")

    plt.plot(epochs,loss)
    plt.plot(epochs,val_loss)
    plt.title("Training and validation loss")

def prediction_human_horse(local_path3, model):
    prediction_dir = os.path.join(local_path3)
    prediction_names = os.listdir(prediction_dir)

    for fn in prediction_names:
     
      img = image.load_img(local_path3+fn, target_size=(300, 300))
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)

      images = np.vstack([x])
      classes = model.predict(images, batch_size=10)
      print(classes[0])

      if classes[0]>0.5:
        print(fn + " is a human")
      else:
        print(fn + " is a horse")

########################################################################
# The main() function
def main():
    
    #print(tf.__version__)
    local_path1 = 'data/horse-human-data/horse-or-human/'
    local_path2 = 'data/horse-human-data/validation-horse-or-human/'
    local_path3 = 'data/horse-human-data/prediction/'
    check_display_data(local_path1, local_path2)
    model = building_model(local_path1, local_path2)
    prediction_human_horse(local_path3, model)

#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()


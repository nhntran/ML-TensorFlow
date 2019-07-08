
 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-

### Learning TensorFlow with Laurence Moroney, Google Brain on Coursera
### /Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/\
###/MachineLearning/TensorFlow/TensorFlowCodes

#Download the model for transfer learning
#!wget --no-check-certificate \
#   https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
#   -O /Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/MachineLearning/TensorFlow/TensorFlowCodes/\
#data/learning/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
### !!!! If error occurs: 
#  File "h5py/h5f.pyx", line 88, in h5py.h5f.open
#OSError: Unable to open file (file signature not found)
# => the .h5 file may be getting corrupted during download 
# => download it manually would solve the problem

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
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

def transfer_learning_model(localpath1):
    #create and load pre trained model:
    pre_trained_model = InceptionV3(input_shape = (150, 150, 3),
                                    include_top = False,
                                    weights = None)
    pre_trained_model.load_weights(localpath1)

    for layer in pre_trained_model.layers:
        layer.trainable = False
    #pre_trained_model.summary()
    # really long model summary output

    last_layer = pre_trained_model.get_layer('mixed7')
    print("Last layer:", last_layer.output_shape)
    last_output = last_layer.output
    return last_output, pre_trained_model

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
    train_datagen = ImageDataGenerator(rescale = 1/255,
                                        rotation_range = 40,
                                        width_shift_range = 0.2,
                                        height_shift_range = 0.2,
                                        shear_range = 0.2,
                                        zoom_range = 0.2,
                                        horizontal_flip = True)

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

def building_model(local_path1, last_output, pre_trained_model):
    callbacks = myCallback()

    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation = 'relu')(x)
    #Dropouts step
    x = layers.Dropout(0.2)(x) #dropout 20%
    x = layers.Dense(1,activation = 'sigmoid')(x)
    model = Model(pre_trained_model.input,x)
    model.summary()
    model.compile(optimizer = RMSprop(lr=1e-04),
                    loss = 'binary_crossentropy',
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
    model.save('TF_Dogs_Cats_TransferLearning0.h5') #after training, saving the model into the .h5 file

    return model

def model_evaluation(acc, val_acc, loss, val_loss):
    epochs = range(len(acc)) #get number of epochs

    #plot training and validation accuracy and loss
    plt.plot(epochs,acc, 'r', "Training Accuracy")
    plt.plot(epochs,val_acc, 'b')
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
    local_path0 = 'data/learning/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    local_path1 = 'data/dog-cat-data/cats_and_dogs_filtered/'
    local_path2 = 'data/dog-cat-data/cat_dog_prediction/'

    # First time running: training model
    check_display_data(local_path1)
    last_output, pre_trained_model = transfer_learning_model(local_path0)
    model = building_model(local_path1, last_output, pre_trained_model)
    prediction_cat_dog(local_path2, model)

    # Second time running: Loading the model again
    # new_model = tf.keras.models.load_model('TF_Dogs_Cats_TransferLearning0.h5')
    # prediction_cat_dog(local_path2, new_model)

#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()



 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-

### Learning TensorFlow with Laurence Moroney, Google Brain on Coursera
### /Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/\
###/MachineLearning/TensorFlow/TensorFlowCodes

### Two-class classification problem: identifying happy versus sad face
# Input: images 300x300 with 3 bytes color: input_shape=(300, 300, 3)

#Download the data for the test
#!wget --no-check-certificate \
#    "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
#    -O "/tmp/happy-or-sad.zip"

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
        desired_accuracy = 0.999
        if(logs.get('acc')>desired_accuracy):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training=True

def check_display_data(local_path):
    
    ### checking the data
    train_happy_dir = os.path.join(local_path + 'happy')
    train_happy_names = os.listdir(train_happy_dir)
    train_sad_dir = os.path.join(local_path + 'sad')
    train_sad_names = os.listdir(train_sad_dir)
    
    print('Total training happy images:',len(train_happy_names))
    print('Total training sad images:', len(train_sad_names))
   
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
    next_happy_pic = [os.path.join(train_happy_dir,fname)
                        for fname in train_happy_names[pic_index-8:pic_index]]
    next_sad_pic = [os.path.join(train_sad_dir,fname)
                        for fname in train_sad_names[pic_index-8:pic_index]]

    for i, img_path in enumerate(next_happy_pic + next_sad_pic):
        sp = plt.subplot(nrows, ncols, i+1)
        #sp.axis('Off') # Don't show axes (or gridlines)

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()

def generating_data(local_path1):
    train_datagen = ImageDataGenerator(rescale = 1/255)

    train_gen = train_datagen.flow_from_directory(
        local_path1,
        target_size = (150,150),
        batch_size = 10, #training image in batch of 128 images
        class_mode = 'binary'
        )

    return train_gen

def building_model(local_path1):
    callbacks = myCallback()
    model = tf.keras.models.Sequential([
        ## 3 levels of convolutions - 1st convolution
        tf.keras.layers.Conv2D(16, (3,3), activation = 'relu',
            input_shape = (150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        ## 2nd convolution
        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        ## 3rd convolution
        tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
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
    
    train_generator = generating_data(local_path1)

    history = model.fit_generator(
        train_generator, steps_per_epoch =2 ,
        epochs = 15, verbose = 1#,
        #callbacks = [callbacks]
        )
    #saving model
    model.save('TF_HappyVsSad_Face.h5')
    return model

def prediction_happy_sad(local_path3, model):
    prediction_dir = os.path.join(local_path3)
    prediction_names = os.listdir(prediction_dir)

    for fn in prediction_names:
     
      img = image.load_img(local_path3+fn, target_size=(150, 150))
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)

      images = np.vstack([x])
      classes = model.predict(images, batch_size=10)
      print(classes[0])

      if classes[0]>0.5:
        print(fn + " is a happy face")
      else:
        print(fn + " is a sad face")

########################################################################
# The main() function
def main():
    
    #print(tf.__version__)
    local_path1 = 'data/happy-sad-data/happy-or-sad/'
    local_path2 = 'data/happy-sad-data/happy-or-sad-prediction/'
    check_display_data(local_path1)
    model = building_model(local_path1)
    print("Predict1")
    prediction_happy_sad(local_path2, model)

    #Loading the model again
    new_model = tf.keras.models.load_model('TF_HappyVsSad_Face.h5')
    new_model.summary()
    print("Predict2")
    prediction_happy_sad(local_path2, new_model)

#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()



 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-

### Learning TensorFlow with Laurence Moroney, Google Brain on Coursera
### /Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/\
###/MachineLearning/TensorFlow/TensorFlowCodes

import os
import numpy as np
import random
# Check matplotlib version to make sure that the images will be displayed
#python -c 'import matplotlib; print(matplotlib.__version__, matplotlib.__file__)'
import matplotlib.pyplot as plt
import tensorflow as tf
from   tensorflow.keras.preprocessing.image import img_to_array, load_img

def visualization(local_path,model):
    
    # Let's define a new Model that will take an image as input, and will output
    # intermediate representations for all layers in the previous model after the first.
    # The model.layers API allows to inspect the impact of convolutions on the images.
    successive_outputs = [layer.output for layer in model.layers[1:]]

    #visualization_model = Model(img_input, successive_outputs)
    visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

    train_cat_dir = os.path.join(local_path + 'train/cats')
    train_cat_names = os.listdir(train_cat_dir)
    train_dog_dir = os.path.join(local_path + 'train/dogs')
    train_dog_names = os.listdir(train_dog_dir)

    # Let's prepare a random input image of a cat or dog from the training set.
    cat_img_files = [os.path.join(train_cat_dir, f) for f in train_cat_names]
    dog_img_files = [os.path.join(train_dog_dir, f) for f in train_dog_names]

    img_path = random.choice(cat_img_files + dog_img_files)
    img = load_img(img_path, target_size=(150, 150))  # this is a PIL image

    x   = img_to_array(img)                           # Numpy array with shape (150, 150, 3)
    x   = x.reshape((1,) + x.shape)                   # Numpy array with shape (1, 150, 150, 3)

    # Rescale by 1/255
    x /= 255.0

    # Let's run our image through our network, thus obtaining all
    # intermediate representations for this image.
    successive_feature_maps = visualization_model.predict(x)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]

    # -----------------------------------------------------------------------
    # Now let's display our representations
    # -----------------------------------------------------------------------
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
      
      if len(feature_map.shape) == 4:
        
        #-------------------------------------------
        # Just do this for the conv / maxpool layers, not the fully-connected layers
        #-------------------------------------------
        n_features = feature_map.shape[-1]  # number of features in the feature map
        size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)
        
        # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))
        
        #-------------------------------------------------
        # Postprocess the feature to be visually palatable
        #-------------------------------------------------
        for i in range(n_features):
            x  = feature_map[0, :, :, i]
            x -= x.mean()
            #print(x.std())
            if(x.std()==0.0):
                x==x
            else:
                x /= x.std() 
            x *=  64
            x += 128
            x  = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size : (i + 1) * size] = x # Tile each filter into a horizontal grid

        #-----------------
        # Display the grid
        #-----------------

        scale = 20. / n_features
        plt.figure( figsize=(scale * n_features, scale) )
        plt.title ( layer_name )
        plt.grid  ( True )
        plt.imshow( display_grid, aspect='auto', cmap='viridis' ) 


########################################################################
# The main() function
def main():
    
    #print(tf.__version__)
    local_path = 'data/dog-cat-data/cats_and_dogs_filtered/'
    model = tf.keras.models.load_model('TF_Dogs_Cats.h5')
    visualization(local_path, model)

#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()

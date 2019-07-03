
 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-

### Learning TensorFlow with Laurence Moroney, Google Brain on Coursera
### /Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/\
###/MachineLearning/TensorFlow/TensorFlowCodes

### import modules used here -- sys is a very standard one
import cv2
#install the cv3 package by the command: pip3 install opencv-python
import numpy as np 
from scipy import misc
import matplotlib.pyplot as plt

def show_image(i):
    plt.grid(False)
    plt.gray()
    plt.axis('off')
    plt.imshow(i)
    plt.show()

def filter_3x3():
    # This filter detects edges nicely
    # It creates a convolution that only passes through sharp edges and straight lines.

    #filter = [ [0, 0, 0], [0, 1, 0], [0, 0, 0]] #nothing, return the original image
    #Experiment with different values for fun effects.
    #filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]

    # A couple more filters to try for fun!
    #filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    #filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

    #sharper
    #filter = [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]
    #show edges excessively
    filter = [[1, 1, 1], [1, -7, 1], [1, 1, 1]]
    # If all the digits in the filter don't add up to 0 or 1, you 
    # should probably do a weight to get it to do so
    # so, for example, if your weights are 1,1,1 1,2,1 1,1,1
    # They add up to 10, so you would set a weight of .1 if you want to normalize them
    return filter


########################################################################
# The main() function
def main():
    i = misc.ascent()
    show_image(i)

    i_transformed = np.copy(i)
    #get the dimensions of the image so we can loop over it later
    size_x = i_transformed.shape[0]
    size_y = i_transformed.shape[1]

    filter = filter_3x3()
    weight = 1
    for x in range(1,size_x-1):
        for y in range(1,size_y-1):
          convolution = 0.0
          convolution = convolution + (i[x - 1, y-1] * filter[0][0])
          convolution = convolution + (i[x, y-1] * filter[0][1])
          convolution = convolution + (i[x + 1, y-1] * filter[0][2])
          convolution = convolution + (i[x-1, y] * filter[1][0])
          convolution = convolution + (i[x, y] * filter[1][1])
          convolution = convolution + (i[x+1, y] * filter[1][2])
          convolution = convolution + (i[x-1, y+1] * filter[2][0])
          convolution = convolution + (i[x, y+1] * filter[2][1])
          convolution = convolution + (i[x+1, y+1] * filter[2][2])
          convolution = convolution * weight
          if(convolution<0):
            convolution=0
          if(convolution>255):
            convolution=255
          i_transformed[x, y] = convolution
    
    show_image(i_transformed)
    #ascent_convolution()

#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()


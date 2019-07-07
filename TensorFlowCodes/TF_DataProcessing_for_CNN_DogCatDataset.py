
 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-

### Learning TensorFlow with Laurence Moroney, Google Brain on Coursera
### /Users/trannguyen/TranData/WORK/BioinformaticsSpecialization_Tran_2019/\
###/MachineLearning/TensorFlow/TensorFlowCodes

### Dogs vs Cats - Kaggle Problem, the two-class classification problem
# Input: 2,000 images 300x300 with 3 bytes color: input_shape=(300, 300, 3)

#Download the real data  (around 25k images)
#!wget --no-check-certificate \
    # "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" \
    # -O "/tmp/cats-and-dogs.zip"

import os
import random
import zipfile
from shutil import copyfile


def data_processing_for_cnn(localpath, split_size):
    
    #unzip the data
    # zip_ref = zipfile.ZipFile('data/cats-and-dogs.zip','r')
    # zip_ref.extractall('localpath')
    # zip_ref.close()

    #make dir
    try:
        os.mkdir(localpath+"cat-dog-full-data")
        #for testing
        os.mkdir(localpath+"/cat-dog-full-data/training")
        os.mkdir(localpath+"/cat-dog-full-data/training/cats")
        os.mkdir(localpath+"/cat-dog-full-data/training/dogs")

        #for training
        os.mkdir(localpath+"/cat-dog-full-data/testing")
        os.mkdir(localpath+"/cat-dog-full-data/testing/cats")
        os.mkdir(localpath+"/cat-dog-full-data/testing/dogs")

    except OSError:
        pass

    cat_path = localpath+"PetImages/Cat/"
    cat_testing = localpath+"cat-dog-full-data/testing/cats/"
    cat_training = localpath+"cat-dog-full-data/training/cats/"

    dog_path = localpath+"PetImages/Dog/"
    dog_testing = localpath+"cat-dog-full-data/testing/dogs/"
    dog_training = localpath+"cat-dog-full-data/training/dogs/"

    split_data(cat_path, cat_training, cat_testing, split_size)
    split_data(dog_path, dog_training, dog_testing, split_size)

    #Check size:
    print("Cat data: Training -", len(os.listdir(cat_training)), 
        "Testing - ", len(os.listdir(cat_testing)))

    print("Dog data: Training -", len(os.listdir(dog_training)), 
        "Testing - ", len(os.listdir(dog_testing)))

def split_data(mpath, training, testing, split_size):
    files = []
    for filename in os.listdir(mpath):
        file = mpath+filename
        if os.path.getsize(file)>0:
            files.append(filename)
        else: 
            print(filename + " is zero length => ignoring this file")

    train_length = int(len(files) * split_size)
    test_length = int(len(files) - train_length)
    random_set = random.sample(files, len(files))
    train_set = random_set[0:train_length]
    test_set = random_set[-test_length:]

    for filename in train_set:
        copyfile(mpath+filename, training + filename)

    for filename in test_set:
        copyfile(mpath+filename, testing + filename)

########################################################################
# The main() function
def main():
    localpath='data/'
    split_size = 0.9 # 90% training, 10% testing
    data_processing_for_cnn(localpath, split_size)

#######################################################################
# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()


'''
Download dataset here:
https://pjreddie.com/projects/cifar-10-dataset-mirror/

Cifar 10 home page:
https://www.cs.toronto.edu/~kriz/cifar.html
'''
import cv2
import os
import numpy as np
import pickle 

def image_loader(image_path, image_size):
    '''
    Load an image from a disk.
    
    :param image_path: String, path to the image
    :param image_size: tuple, size of an output image Example: image_size=(32, 32)
    '''
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size, cv2.INTER_CUBIC)
    return image


def dataset_preprocessing(dataset_path, labels_file_path, image_size, image_paths_pickle):
    '''
    Loads images and labels from dataset folder.
    
    :param dataset_path: String, path to the train/test dataset folder
    :param labels_file_path: String, path to the .txt file where classes names are written
    :param image_size: tuple, single image size
    :param image_paths_pickle: String, name of a pickle file where all image paths will be saved
    '''
    
    with open(labels_file_path, 'r') as f:
        classes = f.read().split('\n')[:-1]
        
    
    images = []
    labels = []
    image_paths = []
    
    for image_name in os.listdir(dataset_path):
        try:
            image_path = os.path.join(dataset_path, image_name)
            images.append(image_loader(image_path, image_size))
            image_paths.append(image_path)
            for idx in range(len(classes)):
                if classes[idx] in image_name: #Example: 0_frog.png
                    labels.append(idx)
        except:
            pass
    
    with open(image_paths_pickle + ".pickle", 'wb') as f:
        pickle.dump(image_paths, f)
    
    assert len(images) == len(labels)
    return np.array(images), np.array(labels)
import pickle
import numpy as np 
import tensorflow as tf
import config as cfg

from utils.utils import *
from utils.dataset import image_loader

from model import ImageSearchModel

def simple_inference(client,
                     train_set_vectors, 
                     uploaded_image_path, 
                     image_size, 
                     distance='hamming'):
    
    '''
    Doing simple inference for single uploaded image.
    
    :param client: Tensorflow serving client
    :param train_set_vectors: loaded training set vectors
    :param uploaded_image_path: string, path to the uploaded image
    :param image_size: tuple, single image (height, width)
    :param dsitance: string, type of distance to be used, 
                             this parameter is used to choose a way how to prepare vectors
    '''
    
    image = image_loader(uploaded_image_path, image_size)

    #Define model inputs 
    req_data = [{'in_tensor_name': 'images:0', 'in_tensor_dtype': 'DT_FLOAT', 'data': np.array([image]).astype(float)},
            {'in_tensor_name': 'drop:0', 'in_tensor_dtype': 'DT_FLOAT', 'data': 0.0}]
    
    #Send request data and get model predictions
    dense_2_features, dense_4_features = client.predict(req_data)
    
    closest_ids = None
    if distance == 'hamming':
        dense_2_features = np.where(dense_2_features < 0.5, 0, 1)
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)
        
        uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))
        
        closest_ids = hamming_distance(train_set_vectors, uploaded_image_vector)
    elif distance == 'cosine':
        uploaded_image_vector = np.hstack((dense_2_features, dense_4_features))
        
        closest_ids = cosine_distance(train_set_vectors, uploaded_image_vector)
        
    return closest_ids
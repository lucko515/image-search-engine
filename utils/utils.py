import numpy as np
from scipy.spatial.distance import hamming, cosine, euclidean


def compare_color(color_vectors, 
                  uploaded_image_colors, 
                  ids):
    '''
    Comparing color vectors of closest images from the training set with a color vector of a uploaded image (query image).
    
    :param color_vectors: color features vectors of closest training set images to the uploaded image
    :param uploaded_image_colors: color vector of the uploaded image
    :param ids: indices of training images being closest to the uploaded image (output from a distance function) 
    '''
    color_distances = []
    
    for i in range(len(color_vectors)):
        color_distances.append(euclidean(color_vectors[i], uploaded_image_colors))
        
    #The 15 is just an random number that I have choosen, you can return as many as you need/want
    return ids[np.argsort(color_distances)[:15]] 

def cosine_distance(training_set_vectors, query_vector, top_n=50):
    '''
    Calculates cosine distances between query image (vector) and all training set images (vectors).
    
    :param training_set_vectors: numpy Matrix, vectors for all images in the training set
    :param query_vector: numpy vector, query image (new image) vector
    :param top_n: integer, number of closest images to return
    '''
    
    distances = []
    
    for i in range(len(training_set_vectors)): #For Cifar 10 -> 50k images
        distances.append(cosine(training_set_vectors[i], query_vector[0]))
        
    return np.argsort(distances)[:top_n]


def hamming_distance(training_set_vectors, query_vector, top_n=50):
    '''
    Calculates hamming distances between query image (vector) and all training set images (vectors).
    
    :param training_set_vectors: numpy Matrix, vectors for all images in the training set
    :param query_vector: numpy vector, query image (new image) vector
    :param top_n: Integer, number of closest images to return
    '''
     
    distances = []
    
    for i in range(len(training_set_vectors)): #For Cifar 10 -> 50k images
        distances.append(hamming(training_set_vectors[i], query_vector[0]))
        
    return np.argsort(distances)[:top_n]


def sparse_accuracy(true_labels, predicted_labels):
    '''
    Calculates accuracy of a model based on softmax outputs.
    
    :param true_labels: numpy array, real labels of each sample. Example: [1, 2, 1, 0, 0]
    :param predicted_labels: numpy matrix, softmax probabilities. Example [[0.2, 0.1, 0.7], [0.9, 0.05, 0.05]]
    '''
    
    assert len(true_labels) == len(predicted_labels)
    
    correct = 0
    
    for i in range(len(true_labels)):
        
        if np.argmax(predicted_labels[i]) == true_labels[i]:
            correct += 1
            
    return correct / len(true_labels)
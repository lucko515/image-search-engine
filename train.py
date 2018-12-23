import numpy as np
import tensorflow as tf
import os
import pickle

import config as cfg
from model import ImageSearchNetwork

def train(model, 
          epochs,
          drop_rate,
          batch_size, 
          data, 
          save_dir, 
          saver_delta=0.15):
    
    '''
    The core training function, use this function to train a model.
    
    :param model: CNN model
    :param epochs: integer, number of epochs
    :param drop_rate: float, dropout_rate
    :param batch_size: integer, number of samples to put through the model at once
    :param data: tuple, train-test data Example(X_train, y_train, X_test, y_test)
    :param save_dir: string, path to a folder where model checkpoints will be saved
    :param saver_delta: float, used to prevent overfitted model to be saved
    '''
    
    X_train, y_train, X_test, y_test = data
    
    #start session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    #define saver
    saver = tf.train.Saver()
    
    best_test_accuracy = 0.0
    #start training loop
    for epoch in range(epochs):
        
        train_accuracy = []
        train_loss = []
        
        for ii in tqdm(range(len(X_train) // batch_size)):
            start_id = ii*batch_size
            end_id = start_id + batch_size
            
            X_batch = X_train[start_id:end_id]
            y_batch = y_train[start_id:end_id]
            
            feed_dict = {model.inputs:X_batch, 
                         model.targets:y_batch, 
                         model.dropout_rate:drop_rate}
            
            _, t_loss, preds_t = session.run([model.optimizer, model.loss, model.predictions], feed_dict=feed_dict)
            
            train_accuracy.append(sparse_accuracy(y_batch, preds_t))
            train_loss.append(t_loss)
            
        print("Epoch: {}/{}".format(epoch, epochs),  
              " | Training accuracy: {}".format(np.mean(train_accuracy)), 
              " | Training loss: {}".format(np.mean(train_loss)) )
        
        test_accuracy = []
        
        for ii in tqdm(range(len(X_test) // batch_size)):
            start_id = ii*batch_size
            end_id = start_id + batch_size
            
            X_batch = X_test[start_id:end_id]
            y_batch = y_test[start_id:end_id]
            
            feed_dict = {model.inputs:X_batch, 
                         model.dropout_rate:0.0}
            
            preds_test = session.run(model.predictions, feed_dict=feed_dict)
            test_accuracy.append(sparse_accuracy(y_batch, preds_test))
            
        print("Test accuracy: {}".format(np.mean(test_accuracy)))
        
        #saving the model
        if np.mean(train_accuracy) > np.mean(test_accuracy): #to prevent underfitting
            if np.abs(np.mean(train_accuracy) - np.mean(test_accuracy)) <= saver_delta: #to prevent overfit
                if np.mean(test_accuracy) >= best_test_accuracy:
                    best_test_accuracy = np.mean(test_accuracy)
                    saver.save(session, "{}/model_epoch_{}.ckpt".format(save_dir, epoch))
                    
    session.close()


def create_training_set_vectors(model, 
                                X_train, 
                                y_train,
                                batch_size,
                                checkpoint_path, 
                                image_size, 
                                distance='hamming'):
    
    '''
    Creates training set vectors and saves them in a pickle file.
    
    :param model: CNN model
    :param X_train: numpy array, loaded training set images
    :param y_train: numpy array,loaded training set labels
    :param batch_size: integer, number of samples to put trhough the model at once
    :param checkpoint_path: string, path to the model checkpoint
    :param image_size: tuple, single image (height, width)
    :param distance: string, type of distance to be used, 
                             this parameter is used to choose a way how to prepare and save training set vectors
    '''
            
    #Define session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    #restore session
    saver = tf.train.Saver()
    saver.restore(session, checkpoint_path)
    
    dense_2_features = []
    dense_4_features = []
    
    #iterate through training set
    for ii in tqdm(range(len(X_train) // batch_size)):
        start_id = ii*batch_size
        end_id = start_id + batch_size

        X_batch = X_train[start_id:end_id]

        feed_dict = {model.inputs:X_batch, 
                     model.dropout_rate:0.0}
        
        dense_2, dense_4 = session.run([model.dense_2_features, model.dense_4_features], feed_dict=feed_dict)
        
        dense_2_features.append(dense_2)
        dense_4_features.append(dense_4)
        
    dense_2_features = np.vstack(dense_2_features)
    dense_4_features = np.vstack(dense_4_features)
    #hamming distance - vectors processing
    if distance == 'hamming':
        dense_2_features = np.where(dense_2_features < 0.5, 0, 1) #binarize vectors
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)
        
        training_vectors = np.hstack((dense_2_features, dense_4_features))
        
        with open('hamming_train_vectors.pickle', 'wb') as f:
            pickle.dump(training_vectors, f)
            
    #cosine distance - vectors processing
    elif distance == 'cosine':
        training_vectors = np.hstack((dense_2_features, dense_4_features))
        
        with open('cosine_train_vectors.pickle', 'wb') as f:
            pickle.dump(training_vectors, f)


def create_training_set_vectors_with_colors(model, 
                                            X_train, 
                                            y_train,
                                            batch_size,
                                            checkpoint_path, 
                                            image_size, 
                                            distance='hamming'):
    
    '''
    Creates training set vectors and saves them in a pickle file.
    
    :param model: CNN model
    :param X_train: numpy array, loaded training set images
    :param y_train: numpy array,loaded training set labels
    :param batch_size: integer, number of samples to put trhough the model at once
    :param checkpoint_path: string, path to the model checkpoint
    :param image_size: tuple, single image (height, width)
    :param distance: string, type of distance to be used, 
                             this parameter is used to choose a way how to prepare and save training set vectors
    '''
            
    #Define session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    #restore session
    saver = tf.train.Saver()
    saver.restore(session, checkpoint_path)
    
    dense_2_features = []
    dense_4_features = []
    
    ##########################################################################
    ### Calculate color feature vectors for each image in the training set ###
    color_features = []
    for img in X_train:
        channels = cv2.split(img)
        features = []
        for chan in channels:
            hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
            features.append(hist)
            
        color_features.append(np.vstack(features).squeeze())
    ##########################################################################
    
    #iterate through training set
    for ii in tqdm(range(len(X_train) // batch_size)):
        start_id = ii*batch_size
        end_id = start_id + batch_size

        X_batch = X_train[start_id:end_id]

        feed_dict = {model.inputs:X_batch, 
                     model.dropout_rate:0.0}
        
        dense_2, dense_4 = session.run([model.dense_2_features, model.dense_4_features], feed_dict=feed_dict)
        
        dense_2_features.append(dense_2)
        dense_4_features.append(dense_4)
        
    dense_2_features = np.vstack(dense_2_features)
    dense_4_features = np.vstack(dense_4_features)
    #hamming distance - vectors processing
    if distance == 'hamming':
        dense_2_features = np.where(dense_2_features < 0.5, 0, 1) #binarize vectors
        dense_4_features = np.where(dense_4_features < 0.5, 0, 1)
        
        training_vectors = np.hstack((dense_2_features, dense_4_features))
        with open('hamming_train_vectors.pickle', 'wb') as f:
            pickle.dump(training_vectors, f)
            
    #cosine distance - vectors processing
    elif distance == 'cosine':
        training_vectors = np.hstack((dense_2_features, dense_4_features))
        training_vectors = np.hstack((training_vectors, color_features[:len(training_vectors)]))
        with open('cosine_train_vectors.pickle', 'wb') as f:
            pickle.dump(training_vectors, f)
            
    #########################################################################
    ### Save training set color feature vectors to a separate pickle file ###
    with open('color_vectors.pickle', 'wb') as f:
        pickle.dump(color_features[:len(training_vectors)], f)
    #########################################################################


'''
Training Example:

epochs = 20
batch_size = 128
learning_rate = 0.001
dropout_probs = 0.6
image_size = (32, 32)
X_train, y_train = dataset_preprocessing('dataset/train/', 'dataset/labels.txt', image_size=image_size, image_paths_pickle="train_images_pickle")
X_test, y_test = dataset_preprocessing('dataset/test/', 'dataset/labels.txt', image_size=image_size, image_paths_pickle="test_images_pickle")

#define the model
model = ImageSearchModel(learning_rate, image_size)

data = (X_train, y_train, X_test, y_test)

train(model, epochs, dropout_probs, batch_size, data, 'saver')

'''
#!/usr/bin/env python3

import numpy as np
import math
import sklearn.base as skb
import sklearn.metrics as skm
import sklearn.model_selection as sms
import tensorflow as tf
from tensorflow import keras
from sklearn.externals import joblib
import subprocess

'''Usage: ./ClassificationModel.py'''

'''creating numpy arrays with annotated data'''
subprocess.call('~/taylorRotation/preprocessing/tensorMatrixPD.py --outfile savedMatrices.npz --IDEAScalls ~/taylorRotation/preprocessing/data/ideas*getfa.bed --RNAseq ~/taylorRotation/preprocessing/data/scriptseq3.v3.filter4ChrLocAvgkw2.bed --ATACseq ~/taylorRotation/preprocessing/data/VISIONmusHem_ccREs_filter2kw.txt')

'''loading saved numpy arrays of annotated data'''
npzfile = np.load("savedMatrices.npz")
#print(npzfile.files) #>['cellTypeIndex', 'labels', 'sequences', 'RNA_seq', 'ATAC_seq']

cellTypeIndex = npzfile['cellTypeIndex']
labels = npzfile['labels']
sequences = npzfile['sequences']
RNA_seq = npzfile['RNA_seq']
ATAC_seq = npzfile['ATAC_seq']

'''splitting data into training and test arrays'''
sss = sms.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(labels, cellTypeIndex):
    X_labels_train_full, X_labels_test = labels[train_index], labels[test_index]
    X_sequences_train_full, X_sequences_test = sequences[train_index], sequences[test_index]
    X_RNAseq_train_full, X_RNAseq_test = RNA_seq[train_index], RNA_seq[test_index]
    X_ATACseq_train_full, X_ATACseq_test = ATAC_seq[train_index], ATAC_seq[test_index]
    Y_cellTypeIndex_train_full, Y_cellTypeIndex_test = cellTypeIndex[train_index], cellTypeIndex[test_index]

'''merging training arrays into a single 2D array'''
seq2d_train = np.reshape(X_sequences_train_full,(int(X_sequences_train_full.shape[0]), -1))
ATAC2d_train = np.reshape(X_ATACseq_train_full, (int(X_sequences_train_full.shape[0]),1))
merged_train_arrays = np.concatenate((seq2d_train,X_RNAseq_train_full,ATAC2d_train), axis=1).astype(np.int32)

#print(X_sequences_train_full.shape) #> (X, 4, 42400)
#print(seq2d_train.shape) #> (X, 169600)
#print(ATAC2d_train.shape) #> (X, 1)
#print(merged_train_arrays.shape) #> (X,169604)

'''merging test arrays into a single 2D array'''
seq2d_test = np.reshape(X_sequences_test, (int(X_sequences_test.shape[0]),-1))
ATAC2d_test = np.reshape(X_ATACseq_test, (int(X_sequences_test.shape[0]),1))
merged_test_arrays = np.concatenate((seq2d_test,X_RNAseq_test,ATAC2d_test), axis=1).astype(np.int32)

#print(X_sequences_test.shape) #> (Z,4,42400)
#print(seq2d_test.shape) #> (Z,169600)
#print(ATAC2d_test.shape) #> (Z,1)
#print(merged_test_arrays.shape) #>(Z,169604)

'''verify labels are int32 type'''
X_labels_train_full = X_labels_train_full.astype(np.int32)
X_labels_test = X_labels_test.astype(np.int32)

'''save test and train arrays'''
f = open('testArrays.npz', 'wb')
np.savez(f, Y_cellTypeIndex_test = Y_cellTypeIndex_test, X_labels_test = X_labels_test, merged_test_arrays = merged_test_arrays)
f.close()
f = open('trainArrays.npz', 'wb')
np.savez(f, Y_cellTypeIndex_train_full = Y_cellTypeIndex_train_full, X_labels_train_full = X_labels_train_full, merged_train_arrays_full = merged_train_arrays)
f.close()

'''Tuning the model using RandomizedSearchCV while employing placeholders, an initializable iterator, and fit_generator to input large amounts of data'''
#default params for nn_wrap class
params = {'alpha':1, 
          'filters':300, 
          'kernel_size':7, 
          'pool_size': 3, 
          'strides':2, 
          'activation': 'elu', 
          'epochs': 6,
          'batchSize': 1500}

#a hyperparameter grid for RandomizedSearchCV to sample from
param_grid = {'alpha': [0.1,0.2,0.3,0.4,0.5,1,2,3,4,5,6,7,8,9,10], 
              'filters': [10,30,60,90,150,300,600], 
              'kernel_size': [2,3,4,5,6,7,8,9,24,36,48,60],
              'pool_size': [1,2,3,4,5,6,9,12], 
              'strides': [1,2,3,4,5,6] ,
              'activation':['relu','elu'], 
              'epochs':[2,3,4,5,6,7,8,9,10,20,30],
              'batchSize':[32,64,128,256,500,750,1000,1500,2000]}
              
#wrapper class of custom sklearn estimator              
#needs get_params and set_params methods which it inherits from sklearn.base.BaseEstimator; can inherit sklearn.base.ClassifierMixin score method if not defined
class nn_wrap(skb.BaseEstimator, skb.ClassifierMixin):
    def __init__(self, alpha, filters, kernel_size, activation, pool_size, strides, epochs, batchSize):
        self.padding='same'
        self.alpha = alpha
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.pool_size = pool_size
        self.strides = strides
        self.epochs = epochs
        self.needsCompiled = True
        self.inputNodes = 0
        self.outputNodes = 27
        self.hiddenNodes = 0
        self.batchSize = batchSize #I was using 1500 for my small data training runs; should I increase for when I am using all of the training data?
        self.steps_per_epoch = 1 #redefined in getInputArrayShape method = rounded up number of batches
    
    '''This function runs and convolution and pooling methods on the input and returns the reduced array
        run prior to this function: instantiate tensor variables
        input: tensor
        output: tensor'''
    def convAndPool(self, X):
        
        convInput = tf.layers.conv1d(X, 
                                     self.filters, 
                                     self.kernel_size, 
                                     padding = self.padding, 
                                     activation= self.activation)
        
        convInput = tf.layers.max_pooling1d(convInput, 
                                            self.pool_size, 
                                            self.strides, 
                                            padding= self.padding)
        return (convInput)
    
    '''This function computes the number of nodes so that the model can be instantiated before fully generating the data
        run prior to this funciton: self.getInputArrayShape()
        input: X,y (numpy arrays - data and labels respectively)
        output: self.inputNodes and self.hiddenNodes are assigned values != 0
        '''
    def computeNodes(self, X, y):
        input_numpyArray_shape = self.getInputArrayShape(X,y)
        #print(input_numpyArray_shape)
        
        self.inputNodes = int(input_numpyArray_shape[1])
        self.hiddenNodes = math.ceil(int(input_numpyArray_shape[0])/
                                     (self.alpha*(self.inputNodes+self.outputNodes)))
        
        #print(self.inputNodes)
        #print(self.hiddenNodes)
    
    '''This function instantiates and compiles the model; notice, inputShape=numberOfFeatures, must be given to the input layer because I use the fit_generator function later
        run prior to this function: self.getInputArrayShape() and self.computeNodes()
        input: no direct input
        output: compiled nn or Sequential model
        '''
        
    def model(self):
        nn = tf.keras.Sequential()
        nn.add(keras.layers.Dense(self.inputNodes, 
                                  activation=self.activation, 
                                  input_shape=(int(self.inputShape[1]),)))
        nn.add(keras.layers.Dense(self.hiddenNodes, 
                                  activation=self.activation))
        nn.add(keras.layers.Dense(self.hiddenNodes, 
                                  activation=self.activation))
        nn.add(keras.layers.Dense(self.outputNodes, 
                                  activation=tf.nn.softmax))
        
        nn.compile(optimizer=tf.train.AdamOptimizer(), 
                   loss='sparse_categorical_crossentropy', 
                   metrics = ['accuracy'])
        return (nn)
    
    '''This function runs the data through a barebones version of the generator function to ascertain what the inputArrayShape will be so that the model can be instantiated
        input: X,y (numpy arrays - data and labels respectively)
        output: defines self.steps_per_epoch (i.e. #_ofSamples/batchSize)
                defines and returns self.inputShape
        '''
    #This method is by far the most inefficient thing I coded, but I have to mimic the generator before actually running the generator - any suggestions for how I can make this more efficient?
    def getInputArrayShape(self,X,y):
        sess = tf.InteractiveSession()
        
        placeholder = tf.placeholder(np.int32, shape=X.shape)
        dataset = tf.data.Dataset.from_tensor_slices((placeholder, y)).batch(self.batchSize)
        
        self.steps_per_epoch = int(np.ceil(int(X.shape[0])/self.batchSize))
        
        iterator = dataset.make_initializable_iterator()
        nextOutput = iterator.get_next()
        iterator.initializer.run(feed_dict = {placeholder:X})
        
        tf.initialize_all_variables().run()
        features,labels = tf.get_default_session().run(nextOutput)


        seq_array = np.reshape(features[:,:-4], (int(features.shape[0]),4,-1))
        RNAseq_array = features[:,-4:-1]
        ATACseq_array = features[:,-1:]

        seq_tensor = tf.convert_to_tensor(seq_array, dtype=tf.float32)

        init_g = tf.global_variables_initializer()
        init_g.run()

        convInput = self.convAndPool(seq_tensor)
        convInput2 = self.convAndPool(convInput)
        flattenSeq = tf.layers.Flatten()(convInput2)

        tf.initialize_all_variables().run()
        flattenSeqNA = flattenSeq.eval()

        input_numpyArray = np.concatenate((flattenSeqNA, 
                                           RNAseq_array, 
                                           ATACseq_array), 
                                          axis=1).astype(np.int32)
    
        sess.close()
        self.inputShape = input_numpyArray.shape
        #print(self.inputShape)
        return(input_numpyArray.shape)
    
    '''This function generates the batched data for each epoch of the fit process
        run during fit_generator method
        input: X,y (numpy arrays, data and labels respectively)
        output: yields the batched input_numpyArray and its corresponding labels_array
        '''
        
    def generator(self, X, y):
        #print("generator 1", X.shape, "\n", y.shape)
        sess = tf.InteractiveSession() #interactive session becomes default session
        
        placeholder = tf.placeholder(np.int32, shape=X.shape) #define a placeholder for the data
        dataset = tf.data.Dataset.from_tensor_slices((placeholder, y)).batch(self.batchSize) #define a dataset for the data and its labels batched according to batchSize
        
        iterator = dataset.make_initializable_iterator() #define iterator
        nextOutput = iterator.get_next() #nextOutput will be a tensor with the data (nextOuput[0]) and the labels (nextOuput[1])
        iterator.initializer.run(feed_dict = {placeholder:X}) #initialize the iterator, feeding X to the placeholder in the dataset
        for epochs in range(self.epochs): #need to feed the batched data for each epoch
            try:
                while True: #will repeat for each step in the epoch
                    tf.initialize_all_variables().run() #deprecated command, but the only way that I could successfully evaluate the tensors to get features, labels numpy arrays was to use this command coupled with the next. If I feed the model the tensors instead of the numpy arrays - then the predict() ouput size issue occurs 
                    features,labels = tf.get_default_session().run(nextOutput)

                    #print(features.shape)
                    #print(type(features)) #>numpy_array
                    #print(labels.shape)
                    #print(type(labels)) #>numpy_array
                    #print("-----Break-----")

                    labels_array = labels
                    seq_array = np.reshape(features[:,:-4], (int(labels_array.shape[0]),4,-1))
                    RNAseq_array = features[:,-4:-1]
                    ATACseq_array = features[:,-1:]

                    seq_tensor = tf.convert_to_tensor(seq_array, dtype=tf.float32) #convolution and pooling require that they are fed a tensor, will reconvert to numpy array

                    init_g = tf.global_variables_initializer()
                    init_g.run()

                    convInput = self.convAndPool(seq_tensor)
                    convInput2 = self.convAndPool(convInput)
                    flattenSeq = tf.layers.Flatten()(convInput2)

                    tf.initialize_all_variables().run() #with these two commands, return the flattened sequence tensor to a numpyArray
                    flattenSeqNA = flattenSeq.eval()

                    input_numpyArray = np.concatenate((flattenSeqNA, 
                                                       RNAseq_array, 
                                                       ATACseq_array), 
                                                      axis=1).astype(np.int32)
                    
                    #print("generator 2", input_numpyArray.shape, "\n", labels_array.shape)
                    yield(input_numpyArray, labels_array)

            except tf.errors.OutOfRangeError:
                iterator.initializer.run(feed_dict = {placeholder:X}) #re-initialize so that data can be rebatched for next epoch
        
        sess.close()
        
    '''This method works as the fit method
        runs: self.computeNodes(), self.model() (only once in order to compile the model) and nn.fit_generator
        input: X,y (numpy arrays - data and labels respectively)
        output: self.estimator is updated to equal fitted nn
                returns self'''
    def fit(self, X, y):
        #print("fit print 1: ", X.shape, "\n", y.shape)
        
        self.computeNodes(X,y)
        
        if self.needsCompiled == True:

            nn = self.model()
            self.needsCompiled = False
        
        nn.fit_generator(self.generator(X,y), 
                         epochs=self.epochs, 
                         steps_per_epoch=self.steps_per_epoch, 
                         workers = 0)
        self.estimator = nn
                
        return(self)
    '''This function is used for prediction and scoring of the estimator; it contains another sort of barebones data generator (doesn't have the for loop as no epochs)
        input: X,y (numpy arrays, data and labels respectivley)
        output: defines self.estimator.score = precision
                returns this precision score as well
        '''    
    def score(self, X, y):
        #print("Score print 1: ", X.shape)
        all_pred_y = []
        all_true_y = []
        instance_len = []
        total_instances = 0
        
        sess = tf.InteractiveSession()
        
        placeholder = tf.placeholder(np.int32, shape=X.shape)
        dataset = tf.data.Dataset.from_tensor_slices((placeholder, y)).batch(self.batchSize)
        
        iterator = dataset.make_initializable_iterator()
        nextOutput = iterator.get_next()
        iterator.initializer.run(feed_dict = {placeholder:X})
        try:
            while True:
                tf.initialize_all_variables().run()
                features,labels = tf.get_default_session().run(nextOutput)

                #print(features.shape)
                #print(type(features))
                #print(labels.shape)
                #print(type(labels))
                #print("-----Break-----")

                labels_array = labels
                seq_array = np.reshape(features[:,:-4], (int(labels_array.shape[0]),4,-1))
                RNAseq_array = features[:,-4:-1]
                ATACseq_array = features[:,-1:]

                seq_tensor = tf.convert_to_tensor(seq_array, dtype=tf.float32)

                init_g = tf.global_variables_initializer()
                init_g.run()

                convInput = self.convAndPool(seq_tensor)
                convInput2 = self.convAndPool(convInput)
                flattenSeq = tf.layers.Flatten()(convInput2)

                tf.initialize_all_variables().run()
                flattenSeqNA = flattenSeq.eval()

                input_numpyArray2 = np.concatenate((flattenSeqNA, 
                                                   RNAseq_array, 
                                                   ATACseq_array), 
                                                  axis=1).astype(np.int32)

                #print("Score print 2", input_numpyArray2.shape, "\n", labels_array.shape)

                pred_y = self.estimator.predict(input_numpyArray2)
                
                total_instances += int(pred_y.shape[0])
                instance_len.append(int(pred_y.shape[0]))
                all_pred_y.append(pred_y)
                all_true_y.append(labels)
                
                #print("Score print 3: ", pred_y.shape)
        
        except tf.errors.OutOfRangeError:
            pass
        
         
        '''compute precision'''
        
        correct_pred = 0
        for i in range(len(all_pred_y)):
            for j in range(instance_len[i]):
                predicted = np.argmax(all_pred_y[i][j,:])
                if predicted == int(all_true_y[i][j]):
                    correct_pred += 1

        precision = correct_pred/total_instances
        #print("score print 4: ", precision)
        self.estimator.score = precision    
        
        sess.close()
        return(self.estimator.score)

train_full_nn=nn_wrap(**params)
rsearch = sms.RandomizedSearchCV(estimator = train_full_nn, param_distributions=param_grid, n_iter=100) 
rsearch.fit(merged_train_arrays, X_labels_train_full)

filename = 'VISION_classification_RSCV_best'
joblib.dump(rsearch.best_estimator_, filename)
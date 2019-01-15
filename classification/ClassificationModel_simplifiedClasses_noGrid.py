#!/usr/bin/env python3

import numpy as np
import math
import itertools
import sklearn.base as skb
import sklearn.metrics as skm
import sklearn.preprocessing as skp
import sklearn.model_selection as sms
from scipy import interp
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.externals import joblib

'''Usage: ./ClassificationModel_simplifiedClasses.py
        to be run following tensorMatrix_simplifyClasses.py'''

'''loading saved numpy arrays of annotated data'''
with np.load("savedMatrices_simplifiedClasses.npz") as npzfile:
print(npzfile.files) #>['cellTypeIndex', 'labels', 'sequences', 'RNA_seq', 'ATAC_seq']

    cellTypeIndex = npzfile['cellTypeIndex']
    labels = npzfile['labels']
    sequences = npzfile['sequences']
    RNA_seq = npzfile['RNA_seq']
    ATAC_seq = npzfile['ATAC_seq']

'''subsampling data'''
scaling_factor = 1e3
print("Scaling_factor for subsampling: ", scaling_factor)
indices = np.random.choice(np.arange(int(cellTypeIndex.shape[0])), size=int(scaling_factor))
cellTypeIndex = cellTypeIndex[indices]
labels = labels[indices]
sequences = sequences[indices]
RNA_seq = RNA_seq[indices]
ATAC_seq = ATAC_seq[indices]
print("subsampled")

'''save subsampling indices'''
f=open("subsampling_indices_noGrid_scaling_factor1e3.npz", 'wb')
np.savez(f, indices = indices)
f.close()
print("indices for subsampling saved")

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

print("X_sequences_train_full_shape: ", X_sequences_train_full.shape) #> (X, 4, 42400)
print("seq2d_train_shape: ", seq2d_train.shape) #> (X, 169600)
print("ATAC2d_train_shape: ", ATAC2d_train.shape) #> (X, 1)
print("merged_train_shape: ", merged_train_arrays.shape) #> (X,169604)

'''merging test arrays into a single 2D array'''
seq2d_test = np.reshape(X_sequences_test, (int(X_sequences_test.shape[0]),-1))
ATAC2d_test = np.reshape(X_ATACseq_test, (int(X_sequences_test.shape[0]),1))
merged_test_arrays = np.concatenate((seq2d_test,X_RNAseq_test,ATAC2d_test), axis=1).astype(np.int32)

print("X_sequences_test_shape: ", X_sequences_test.shape) #> (Z,4,42400)
print("seq2d_test_shape: ",seq2d_test.shape) #> (Z,169600)
print("ATAC2d_test_shape: ",ATAC2d_test.shape) #> (Z,1)
print("merged_test_shape: ",merged_test_arrays.shape) #>(Z,169604)

'''verify labels are int32 type'''
X_labels_train_full = X_labels_train_full.astype(np.int32)
X_labels_test = X_labels_test.astype(np.int32)

'''save test and train arrays'''
f = open('testArrays_simplifiedClasses_noGrid.npz', 'wb')
np.savez(f, Y_cellTypeIndex_test = Y_cellTypeIndex_test, X_labels_test = X_labels_test, merged_test_arrays = merged_test_arrays)
f.close()

print("test arrays should be saved")

'''Tuning the model using RandomizedSearchCV while employing placeholders, an initializable iterator, and fit_generator to input large amounts of data'''
#default params for nn_wrap class
params = {'alpha':0.3, 
          'filters':30, 
          'kernel_size':6, 
          'pool_size': 4, 
          'strides':2, 
          'activation': 'elu', 
          'epochs': 5,
          'batchSize': 32}

#a hyperparameter grid for RandomizedSearchCV to sample from
param_grid = {'alpha': [0.2,0.5,1,2], 
              'filters': [20,30,60,90], 
              'kernel_size': [2,3,4,5,6,7,8,9,24,36,48,60],
              'pool_size': [1,2,3,4,5,6,9,12], 
              'strides': [1,2,3,4,5,6] ,
              'activation':['relu','elu'], 
              'epochs':[4,5,6,7,8,9,10],
              'batchSize':[32,64,128]}
              
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
        self.inputNodes = 0 #num input features
        self.outputNodes = 5 #num_classes
        self.hiddenNodes = 0
        self.batchSize = batchSize #higher accuracy when smaller
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
        print("computeNodes() print1: ",input_numpyArray_shape)
        
        self.inputNodes = int(input_numpyArray_shape[1])
        self.hiddenNodes = math.ceil(int(input_numpyArray_shape[0])/
                                     (self.alpha*(self.inputNodes+self.outputNodes)))
        
        print("inputNodes: ",self.inputNodes)
        print("hiddenNodes: ",self.hiddenNodes)
    
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
        print("inputShape: ",self.inputShape)
        return(input_numpyArray.shape)
    
    '''This function generates the batched data for each epoch of the fit process
        run during fit_generator method
        input: X,y (numpy arrays, data and labels respectively)
        output: yields the batched input_numpyArray and its corresponding labels_array
        '''
        
    def generator(self, X, y):
        print("generator 1", X.shape, "\n", y.shape)
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
                    
                    print("generator 2", input_numpyArray.shape, "\n", labels_array.shape)
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
        print("fit print 1: ", X.shape, "\n", y.shape)
        
        self.computeNodes(X,y)
        
        if self.needsCompiled == True:

            nn = self.model()
            self.needsCompiled = False
        
        checkpoint_path = '/home-3/kweave23@jhu.edu/work/users/kweave23/out/checkpoints/scaleDown1_simplifiedClasses_noGrid.ckpt'
        
        '''Create checkpoint callback'''
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
        
        
        nn.fit_generator(self.generator(X,y), 
                         epochs=self.epochs, 
                         steps_per_epoch=self.steps_per_epoch, 
                         workers = 0,
                         callbacks = [cp_callback])
                         
        self.estimator = nn
                
        return(self)
    '''This function is used for prediction and scoring of the estimator; it contains another sort of barebones data generator (doesn't have the for loop as no epochs)
        input: X,y (numpy arrays, data and labels respectivley)
        output: defines self.estimator.score = precision
                returns this precision score as well
        '''    
    def score(self, X, y):
        print("Score print 1: ", X.shape)
	print("self.Parameters:\nalpha: ",self.alpha,"\nfilters: ",self.filters,"\nkernel_size: ",self.kernel_size,"\nactivation: ", self.activation, "\npool_size: ", self.pool_size, "\nstrides: ", self.strides, "\nepochs: ", self.epochs,"\ninputNodes: ", self.inputNodes, "\nhiddenNodes: ", self.hiddenNodes, "\nbatchSize: ", self.batchSize)
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

                print("features: ",features.shape)
                print(type(features))
                print("labels: ",labels.shape)
                print(type(labels))
                print("-----Break-----")

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

                print("Score print 3", input_numpyArray2.shape, "\n", labels_array.shape)

                pred_y = self.estimator.predict(input_numpyArray2)
                
                total_instances += int(pred_y.shape[0])
                instance_len.append(int(pred_y.shape[0]))
                all_pred_y.append(pred_y)
                all_true_y.append(labels)
                
                print("Score print 4: ", pred_y.shape)
        
        except tf.errors.OutOfRangeError:
            pass
        
         
        '''compute accuracy'''
        y_pred = []
	y_true = []
        correct_pred = 0
        for i in range(len(all_pred_y)):
            for j in range(instance_len[i]):
                predicted = np.argmax(all_pred_y[i][j,:])
                y_pred.append(predicted)
		true = int(all_true_y[i][j])
		y_true.append(true)
		if predicted == true:
                    correct_pred += 1

        accuracy = (correct_pred/total_instances)*100
        print("score print 5: ", accuracy)
        self.estimator.score = accuracy
	self.y_pred = y_pred
	self.y_true = y_true    
        

        sess.close()
        return(self.estimator.score)

    def predict(self, X, y):
	accuracy = self.score(X,y)
	
	'''confusion_matrix'''
	confusion_matrix = skm.confusion_matrix(self.y_true, self.y_pred)
	
	'''classification_report'''
	classification_report = skm.classification_report(self.y_true, self.y_pred)
	
	'''binarize the predictions'''
	y_true = skp.label_binarize(self.y_true, classes=np.arange(5))
	y_pred = skp.label_binarize(self.y_pred, classes=np.arange(5))

	'''precision, recall, average_precision, false_positive_rate, true_positive_rate, roc_auc'''
	num_classes = int(np.max(self.y_true)+1)
	print("num_classes = 5: ", num_classes == 5)
	precision = dict()
	recall = dict()
	average_precision = dict()
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(num_classes):
	    precision[i], recall[i], _ = skm.precision_recall_curve(y_true[:,i], y_pred[:,i])
	    average_precision[i] = skm.average_precision_score(y_true[:,i], y_pred[:,i])
	    fpr[i], tpr[i], _ = skm.roc_curve(y_true[:,i], y_pred[:,i])    
	    roc_auc[i] = skm.auc(fpr[i], tpr[i])
	
	'''A micro-average - quantifing score on all classes jointly'''
	precision["micro"], recall["micro"], _ = skm.precision_recall_curve(y_true.ravel(), y_pred.ravel())
	average_precision["micro"] = skm.average_precision_score(y_true, y_pred, average="micro")
	print("Average precision score, micro-averaged over all classes: {0:0.2f}".format(average_precision["micro"]))
	
	'''Compute micro-average ROC curve and ROC area'''
	fpr["micro"], tpr["micro"],_ = skm.roc_curve(y_true.ravel(), y_pred.ravel())
	roc_auc["micro"] = skm.auc(fpr["micro"], tpr["micro"])

	'''Compute macro-average ROC curve and ROC area - First aggregate all false positive rates'''
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

	'''-Then interpolate all ROC curves'''
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(num_classes):
	    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	'''-Finally average it and compute AUC'''
	mean_tpr /= num_classes

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = skm.auc(fpr["macro"], tpr["macro"])
	
	return(accuracy, confusion_matrix, classification_report, precision, recall, fpr, tpr, roc_auc, average_precision)

train_full_nn=nn_wrap(**params)
print("-----------WRAPPER INSTANTIATED-------------")
#rsearch = sms.RandomizedSearchCV(estimator = train_full_nn, param_distributions=param_grid, n_iter=6) 
#rsearch.fit(merged_train_arrays, X_labels_train_full)
train_full_nn.fit(merged_train_arrays, X_labels_train_full)
print("----------FIT-----------")

filename = 'VISION_classification_trained_estimator_simplifiedClasses_noGrid'
joblib.dump(train_full_nn, filename)
print("trained estimator should be saved")

'''predict test data and get metrics'''
accuracy, confusion_matrix, classification_report, precision, recall, fpr, tpr, roc_auc, average_precision = train_full_nn.predict(merged_test_arrays, X_labels_test)
print("-----------PREDICTED--------------")

num_classes = int(confusion_matrix.shape[0])
print("num_classes = 5: ", num_classes == 5)

print("----------Begin Metrics----------")
print("-----------Accuracy on test data: ", accuracy)
print("Classification report:\n",classification_report)

'''plot the micro-averaged precision/recall curve'''
fig, ax = plt.subplots(figsize=(15,15))
plt.step(recall['micro'], precision['micro'], alpha=0.2, color='b', where='post')
plt.fill_between(recall['micro'], precision['micro'], alpha=0.2, color='b')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim([0.0,1.05])
plt.xlim([0.0,1.0])
fig.suptitle("Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision['micro']))
fig.savefig("micro-averaged_precision-recall_nogrid.png")
plt.close(fig)
print('micro-averaged precision/recall curve plotted')

'''plot the precision/recall curve for each class'''
colors = itertools.cycle(['navy', 'deeppink', 'darkslategray', 'darkorchid', 'saddlebrown','darkgreen', 'darkturquoise'])

fig,ax = plt.subplots(figsize=(20,15))
lines=[]
labels=[]
l,=plt.plot(recall['micro'], precision=['micro'], color='gold', lw=2)
lines.append(l)
labels.append('micro-average precision-recall (area={1:0.2f})'''.format(i,average_precision[i]))

for i, color in zip(range(num_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.appenc(l)
    labels.append('Precision-recall for class{0} (area={1:0.2f})'''.format(i,average_precision[i]))

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
fig.suptitle('Precision-Recall Curves')
plt.legend(lines, labels, loc='upper right', prop=dict(size=12))
plt.tight_layout()
fig.savefig("Precision_Recall_nogrid.png")
plt.close(fig)
print("precision/recall curve for each class plotted")

'''Plot the ROC-AUC - only micro and macro'''
fig, ax = plt.subplots(figsize=(18,15))
plt.plot(fpr['micro'],tpr["micro"], label = 'micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]), color='deeppink', linestyle=":", linewidth = 4)
plt.plot(fpr['macro'],tpr['macro'], label = 'macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]), color='navy', linestyle='--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
fig.suptitle('Receiver operating characteristic', fontsize=15)
plt.legend(loc='lower right')
fig.savefig("ROC_AUC_micro_macroOnly_nogrid.png")
plt.close(fig)
print("ROC-AUC for micro and macro only plotted")

'''plot the ROC-AUC all curves'''
fig,ax = plt.subplots(figsize=(10,10))
plt.plot(fpr['micro'], tpr['micro'], label='micro-average ROC curve (area={0:0.2f})'''.format(roc_auc['micro']), color = 'deeppink', linestyle = ':', linewidth =4)
plt.plot(fpr['macro'], tpr['macro'], label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc['macro']), color='navy', linestyle = '--', linewidth=2)

colors= itertools.cycle(['darkslategray', 'darkorange', 'darkorchid', 'saddlebrown', 'darkturquoise', 'darkgreen'])

for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=4, label='ROC curve for class {0} (area={1:0.2f})'''.format(i, roc_auc[i]))
plt.xlim([0.0, 1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
fig.suptitle("Receiver Operating Characteristic', fontsize=15)
plt.legend(loc='lower right', prop=dict(size=7))
fig.savefig("ROC_AUC_all_nogrid.png")
plt.close(fig)
print("ROC-AUC all classes plotted")

'''Confusion matrix with number labels'''
def plot_confusion_matrix(cm, labeling=True, normalize=False, title='Confusion Matrix', cmap='Greys'):
    if normalize:
	cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
	print("Normalized confusion matrix")
    
    m = np.max(np.abs(cm))
    vmin = 0 if normalize else -1*m
    fig, ax = plt.subplots(figsize=(16,14))
    ax.set_title(title, fontsize=20)
    im = ax.pcolor(cm, cmap=cmap, vmin = vmin, vmax=m)
    cbar = fig.colorbar(im, ax=ax)
    tick_marks = np.arange(int(cm.shape[0]))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    fmt = '.2f' if normalize else 'd'
    thresh = m/2.0

    for i,j in itertools.product(range(int(cm.shape[0])), range(int(cm.shape[1]))):
        if cmap == 'Greys':
	    color ='white' if cm[i,j] > thresh else 'black'
	elif cmap == 'viridis':
	    color = 'black' if cm[i,j] > thresh else 'white'
	if labeling:
	    plt.text(j+0.5,i+0.5, format(cm[i,j],fmt), horizontalalignment='center', verticalalignment = 'center', color=color)
    
    ax.set_ylabel('True label', fontsize=20)
    ax.set_xlabel('Predicted label', fontsize=20)
    plt.tight_layout()

    fig.savefig("confusion_matrix_cmap_{}_normalize_{}_sqlabeling_{}_numberLabels.png".format(cmap, normalize, labeling))
    plt.close(fig)

plot_confusion_matrix(confusion_matrix, normalize=True, cmap='viridis')
plot_confusion_matrix(confusion_matrix, normalize=True)
plot_confusion_matrix(confusion_matrix)
plot_confusion_matrix(confusion_matrix, labeling=False)
plot_confusion_matrix(confusion_matrix, cmap='viridis')
plot_confusion_matrix(confusion_matrix, cmap='viridis', labeling=False)
plot_confusion_matrix(confusion_matrix, normalize=True, cmap='viridis, labeling=False)
plot_confusion_matrix(confusion_matrix, normalize=True, labeling=False)
print("8 confusion matrices plotted")
print("-------End Metrics------")

#filename = 'VISION_classification_RSCV_best_simplifiedClasses_refinedGrid'
#joblib.dump(rsearch.best_estimator_, filename)
#print("best estimator should be saved")

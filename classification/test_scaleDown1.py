#!/usr/bin/env python3

import argparse as ap
import numpy as np
import math
import itertools
import sklearn.base as skb
import sklearn.metrics as skm
import sklearn.preprocessing as skp
from sklearn.externals import joblib
from scipy import interp
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


'''Usage ./test_scaleDown1.py --SavedModelFile saved_model_ext --TestDataFile test_data --LastCheckpoint last_checkpoint [--plotName_description]'''

parser = ap.ArgumentParser(description='Testing a PreTrained Model')
parser.add_argument('--SavedModelFile', action='store', nargs=1, type=str, required = False,  default='',help='Filename for saved model')
parser.add_argument('--TestDataFile', action='store', nargs=1, type=str, required = True, help="Filename for saved test data")
parser.add_argument('--LastCheckpoint', action='store', nargs=1, type=str, required= True, help ='Filename for last training checkpoint')
parser.add_argument('--plotName_description', action='store', nargs=1, type=str, required=False, default =[''], help='any description you would like added to the end of the plot files before the extension')
args=parser.parse_args()

model_ext = args.SavedModelFile[0] #Turns out I can't get the pickle-ing to work properly for this right now....
test_data_file = args.TestDataFile[0]
checkpoint = args.LastCheckpoint[0]
plotName = args.plotName_description[0]


'''This nn_wrap is different from the nn_wrap used for classification in ClassificationModel_*.py scripts
1) checkpoint (path with the saved training weights) is a necessary parameter for __init__
2) various self.Parameters are defined differently in __init__, specifically: self.inputNodes, self.hiddenNodes, self.inputShape, and of course self.checkpoint
3) computeNodes(), getInputArrayShape(), fit(), generator() will never be called...
4) model(self) -> create_model(self) with three commented syntax/line differences
5) predict(self,X,y) -> wrapper_predict(self,X,y)
'''
class nn_wrap(skb.BaseEstimator, skb.ClassifierMixin):
	def __init__(self, checkpoint, alpha, filters, kernel_size, activation, pool_size, strides, epochs, batchSize):
		self.padding='same'
		self.alpha = alpha
		self.filters = filters
		self.kernel_size = kernel_size
		self.activation = activation
		self.pool_size = pool_size
		self.strides = strides
		self.epochs = epochs
		self.needsCompiled = True
		self.inputNodes = 24 #num input features
		self.outputNodes = 5 #num_classes
		self.hiddenNodes = 4
		self.batchSize = batchSize #higher accuracy when smaller
		self.steps_per_epoch = 1 #redefined in getInputArrayShape method = rounded up number of batches
		self.inputShape = 24
		self.checkpoint = checkpoint

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

	def create_model(self):
		nn = tf.keras.Sequential()
		nn.add(keras.layers.Dense(self.inputNodes,
								  activation=self.activation,
								  input_shape=(self.inputShape,))) #difference between model(self) and create_model(self)
		nn.add(keras.layers.Dense(self.hiddenNodes,
								  activation=self.activation))
		nn.add(keras.layers.Dense(self.hiddenNodes,
								  activation=self.activation))
		nn.add(keras.layers.Dense(self.outputNodes,
								  activation=tf.nn.softmax))

		nn.compile(optimizer=tf.train.AdamOptimizer(),
				   loss='sparse_categorical_crossentropy',
				   metrics = ['accuracy'])

		nn.load_weights(self.checkpoint)#difference between model(self) and create_model(self)
		self.estimator = nn#difference between model(self) and create_model(self)
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

			nn = self.create_model()
			self.needsCompiled = False

		checkpoint_path = './classification/scaleDown1_simplifiedClasses_noGrid.ckpt'

		'''Create checkpoint callback so that weights are saved following training'''
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

	def wrapper_predict(self, X, y):
		self.steps_per_epoch = int(np.ceil(int(X.shape[0])/self.batchSize)) #difference between predict() and  wrapper_predict()
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


''' load test data'''
with np.load("{}".format(test_data_file)) as npzfile:
    cellTypeIndex = npzfile['Y_cellTypeIndex_test']
    labels = npzfile['X_labels_test']
    merged_test_arrays = npzfile['merged_test_arrays']

#default params for nn_wrap class
'''make sure that these parameters match whatever was used in training!!!'''
params = {'alpha':0.3,
		  'filters':20,
		  'kernel_size':6,
		  'pool_size': 4,
		  'strides':2,
		  'activation': 'elu',
		  'epochs': 5,
		  'batchSize': 32}


'loading trained_nn'
trained_nn = nn_wrap(checkpoint, **params)
trained_nn.create_model()

'''Get metrics'''
accuracy, confusion_matrix, classification_report, precision, recall, fpr, tpr, roc_auc, average_precision = trained_nn.wrapper_predict(merged_test_arrays, labels)

num_classes = int(confusion_matrix.shape[0])
print("num_classes = 5: ", num_classes == 5)

print("----BEGIN METRICS----")
print("Accuracy on test data: ", accuracy)

'''Classification report with number labels'''
print(classification_report)

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

	figname = "confusion_matrix_cmap_{}_normalize_{}_sqlabeling_{}_numberLabels_{}.png".format(cmap, normalize, labeling, plotName)
	fig.savefig(figname)
	print("confusion matrix is plotted and saved as {}".format(figname))
	plt.close(fig)

plot_confusion_matrix(confusion_matrix, normalize=True, cmap='viridis')
plot_confusion_matrix(confusion_matrix, normalize=True)
plot_confusion_matrix(confusion_matrix)
plot_confusion_matrix(confusion_matrix, labeling=False)
plot_confusion_matrix(confusion_matrix, cmap='viridis')
plot_confusion_matrix(confusion_matrix, cmap='viridis', labeling=False)
plot_confusion_matrix(confusion_matrix, normalize=True, cmap='viridis', labeling=False)
plot_confusion_matrix(confusion_matrix, normalize=True, labeling=False)
print("8 confusion matrices plotted")

print("Plotting micro-averaged precision/recall curve")
'''Plot the micro-averaged precision/recall curve'''
fig,ax=plt.subplots(figsize=(15,15))
plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
         where='post')
plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
fig.suptitle(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))
figname = "micro-averaged_precision_recall_{}.png".format(plotName)
fig.savefig(figname)
print("microaveraged precision recall curve is plotted and saved as {}".format(figname))
plt.close(fig)

print("plotting the precision/recall curve for each class")
'''Plot the precision/recall curve for each class'''
colors = itertools.cycle(['navy', 'deeppink', 'darkslategray', 'darkorchid', 'saddlebrown','darkgreen', 'darkturquoise'])

fig,ax = plt.subplots(figsize=(20,15))
lines = []
labels = []
l, = plt.plot(recall['micro'], precision['micro'], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall')

for i, color in zip(range(num_classes), colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class{0} (area={1:0.2f})'''.format(i, average_precision[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
fig.suptitle('Precision-Recall Curve')
plt.legend(lines, labels, loc='upper right', prop=dict(size=12))
plt.tight_layout()
figname = "Precision_Recall_testScaleDown1_{}.png".format(plotName)
fig.savefig(figname)
print("precision recall curve for each class is plotted and saved as {}".format(figname))
plt.close(fig)

print("plotting the ROC-AUC curves")
'''Plot the ROC-AUC'''
#Plot only micro and macro
fig,ax = plt.subplots(figsize=(18,15))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
fig.suptitle('Receiver operating characteristic', fontsize=15)
plt.legend(loc="lower right")
figname = "ROC_AUC_micro_macroOnly_{}.png".format(plotName)
fig.savefig(figname)
print("ROC AUC for micro and macro only is plotted and saved as {}".format(figname))
plt.close(fig)

# Plot all ROC curves
fig,ax = plt.subplots(figsize=(10,10))
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = itertools.cycle(['darkslategray', 'darkorange', 'darkorchid', 'saddlebrown', 'darkturquoise', 'darkgreen'])
for i, color in zip(range(num_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=4,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
fig.suptitle('Receiver operating characteristic', fontsize=15)
plt.legend(loc="lower right", prop=dict(size=7))
figname = "ROC_AUC_{}.png".format(plotName)
fig.savefig(figname)
print("ROC AUC for all classes is plotted and saved as {}".format(figname))
plt.close(fig)
print("-------End Metrics------")

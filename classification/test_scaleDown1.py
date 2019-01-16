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


'''Usage ./test_scaleDown1.py --SavedModelFile saved_model_ext --TestDataFile test_data'''

parser = ap.ArgumentParser(description='Testing a PreTrained Model')
parser.add_argument('--SavedModelFile', action='store', nargs=1, type=str, required = True, help='Filename for saved model')
parser.add_argument('--TestDataFile', action='store', nargs=1, type=str, required = True, help="Filename for saved test data")
args=parser.parse_args()

model_ext = args.SavedModelFile
test_data_file = args.TestDataFile

''' load saved estimator'''
estimator = joblib.load(model_ext)

''' load test data'''
with np.load("{}".format(test_data_file)) as npzfile:
    cellTypeIndex = npzfile['Y_cellTypeIndex_test']
    labels = npzfile['X_labels_test']
    merged_test_arrays = npzfile['merged_test_arrays']

#default params for nn_wrap class
params = {'alpha':0.3,
		  'filters':20,
		  'kernel_size':6,
		  'pool_size': 4,
		  'strides':2,
		  'activation': 'elu',
		  'epochs': 5,
		  'batchSize': 32}

'''Function to produce metrics of loaded estimator from test data'''
def score(X, y, estimator, alpha, filters, kernel_size, pool_size, strides, activation, epochs, batchSize):
    batchSize = batchSize
    filters = filters
    kernel_size = kernel_size
    activation = activation
    pool_size = pool_size
    strides = strides

    all_pred_y = []
    all_true_y = []
    instance_len = []
    total_instances = 0

    print("Score print 1: ", X.shape)

    sess = tf.InteractiveSession()

    placeholder = tf.placeholder(np.int32, shape=X.shape)
    dataset = tf.data.Dataset.from_tensor_slices((placeholder, y)).batch(batchSize)

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

            convInput = tf.layers.conv1d(seq_tensor,
                                         filters,
                                         padding = 'same',
                                         activation= activation)

            convInput = tf.layers.max_pooling1d(convInput,
                                                pool_size,
                                                strides,
                                                padding= 'same')

            convInput2 = tf.layers.conv1d(convInput,
                                          filters,
                                          padding='same',
                                          activation = activation)

            convInput2 = tf.layers.max_pooling1d(convInput2,
                                                 pool_size,
                                                 strides,
                                                 padding='same')

            flattenSeq = tf.layers.Flatten()(convInput2)

            tf.initialize_all_variables().run()
            flattenSeqNA = flattenSeq.eval()

            input_numpyArray2 = np.concatenate((flattenSeqNA,
                                               RNAseq_array,
                                               ATACseq_array),
                                              axis=1).astype(np.int32)

            print("Score print 3", input_numpyArray2.shape, "\n", labels_array.shape)
            pred_y = estimator.predict(input_numpyArray2)

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

    num_classes = int(np.max(y_true)+1)
    print("num_classes = 5: ", num_classes == 5)
    accuracy = correct_pred/total_instances
    print("score print 5: ", accuracy)

    '''confusion_matrix'''
    confusion_matrix = skm.confusion_matrix(y_true, y_pred)



    classification_report = skm.classification_report(y_true, y_pred)

    '''binarize the predictions'''
    y_true = skp.label_binarize(y_true, classes=np.arange(5))
    y_pred = skp.label_binarize(y_pred, classes=np.arange(5))


    '''precision, recall, average_precision, false_positive_rate, t_positive_rate, roc_auc'''
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = skm.precision_recall_curve(y_true[:, i],
                                                            y_pred[:, i])
        average_precision[i] = skm.average_precision_score(y_true[:, i], y_pred[:, i])
        fpr[i], tpr[i], _ = skm.roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = skm.auc(fpr[i], tpr[i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = skm.precision_recall_curve(y_true.ravel(),
        y_pred.ravel())
    average_precision["micro"] = skm.average_precision_score(y_true, y_pred,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = skm.roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = skm.auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area - First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # -Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # -Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = skm.auc(fpr["macro"], tpr["macro"])

    sess.close()

    return(accuracy,
           confusion_matrix,
           classification_report,
           precision,
           recall,
           fpr,
           tpr,
           roc_auc,
           average_precision)

'''Get metrics'''
accuracy, confusion_matrix, classification_report, precision, recall, fpr, tpr, roc_auc, average_precision = score(merged_test_arrays, labels, estimator, **params)

num_classes = int(confusion_matrix.shape[0])
print("num_classes = 5: ", num_classes == 5)

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

	fig.savefig("confusion_matrix_cmap_{}_normalize_{}_sqlabeling_{}_numberLabels.png".format(cmap, normalize, labeling))
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
print("-------End Metrics------")

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
fig.savefig("micro-averaged_precision_recall.png")
plt.close(fig)

'''Plot the precision/recall curve for each class'''
colors = itertools.cycle(['navy', 'deeppink', 'darkslategray', 'darkorchid', 'saddlebrown','darkgreen', 'darkturquoise'])

fig,ax = plt.subplots(figsize=(20,15))
lines = []
labels = []
l, = plt.plot(recall['micro'], precision['micro'], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area={1:0.2f})'''.format(i, average_precision[i]))

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
fig.savefig("Precision_Recall_testScaleDown1.png")
plt.close(fig)

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
fig.savefig("ROC_AUC_micro_macroOnly_testScaleDown1.png")
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
fig.savefig("ROC_AUC_testScaleDown1.png")
plt.close(fig)



# '''Confusion matrix with number labels'''
# def plot_confusion_matrix2(cm,
#                           labeling=True,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap='Greys'):
#     if normalize:
#         cm =cm.astype('float')/cm.sum(axis=1)[:,np.newaxis] #np.newaxis is an alias for none
#         print("Normalized confusion matrix")
#
#     m = np.max(np.abs(cm))
#     vmin = 0 if normalize else -1*m
#     fig, ax = plt.subplots(figsize=(16,14))
#     ax.set_title(title, fontsize=20)
#     im = ax.pcolor(cm, cmap=cmap, vmin = vmin, vmax=m)
#     cbar = fig.colorbar(im, ax=ax)
#     tick_marks = np.arange(int(cm.shape[0]))
#     #print(tick_marks)
#     plt.xticks(tick_marks, tick_marks)
#     plt.yticks(tick_marks, tick_marks)
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = m/2.0
#
#     for i,j in itertools.product(range(int(cm.shape[0])), range(int(cm.shape[1]))):
#         if cmap == 'Greys':
#             color='white' if cm[i,j] > thresh else 'black'
#         elif cmap =='viridis':
#             color = 'black'if cm[i,j] > thresh else 'white'
#         if labeling:
#             plt.text(j+0.5,i+0.5, format(cm[i,j], fmt),
#                     horizontalalignment='center',
#                     verticalalignment = 'center',
#                     color=color)
#     ax.set_ylabel('True label', fontsize=20)
#     ax.set_xlabel('Predicted label', fontsize=20)
#     plt.tight_layout()
#     fig.savefig("confusion_matrix_cmap_{}_normalize_{}_sqlabeling{}_numberLabels.png".format(cmap, normalize, labeling))
#     plt.close(fig)
#
# plot_confusion_matrix2(confusion_matrix, normalize=True, title='Confusion matrix', cmap='viridis')
# plot_confusion_matrix2(confusion_matrix, normalize=True, title='Confusion matrix')
# plot_confusion_matrix2(confusion_matrix, title='Confusion matrix')
# plot_confusion_matrix2(confusion_matrix, title='Confusion matrix', labeling=False)
# plot_confusion_matrix2(confusion_matrix, title='Confusion matrix', cmap='viridis')
# plot_confusion_matrix2(confusion_matrix, title='Confusion matrix', cmap='viridis', labeling=False)
# plot_confusion_matrix2(confusion_matrix, normalize=True, title='Confusion matrix', cmap='viridis', labeling=False)
# plot_confusion_matrix2(confusion_matrix, normalize=True, title='Confusion matrix', labeling=False)

# '''Confusion matrix with letter labels'''
# def plot_confusion_matrix(cm, labels,
#                           labeling=True,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap='Greys'):
#     legend_elements = ['A=Active', 'B=Bivalent','C=CTCF bound','E=Enhancer-like', 'H=Heterochromatin', 'N=Nuclease accessible','P=Promoter-like', 'Pc = Polycomb','Q=Quiescent', 'T=Transcribed']
#     if normalize:
#         cm =cm.astype('float')/cm.sum(axis=1)[:,np.newaxis] #np.newaxis is an alias for none
#         print("Normalized confusion matrix")
#
#     m = np.max(np.abs(cm))
#     vmin = 0 if normalize else -1*m
#     fig, ax = plt.subplots(figsize=(16,14))
#     ax.set_title(title, fontsize=20)
#     im = ax.pcolor(cm, cmap=cmap, vmin = vmin, vmax=m)
#     cbar = fig.colorbar(im, ax=ax)
#     tick_marks = np.arange(len(labels))
#     #print(tick_marks)
#     plt.xticks(tick_marks, labels, rotation=50)
#     plt.yticks(tick_marks, labels, rotation=50)
#     patch = mpatches.Patch(color=None,alpha=0)
#     plt.legend(handles=[patch,patch,patch,patch,patch,patch,patch,patch,patch,patch], labels=legend_elements, loc=(1.2,0.8))
#
#     fmt = '.2f' if normalize else 'd'
#     thresh = m/2.0
#
#     for i,j in itertools.product(range(int(cm.shape[0])), range(int(cm.shape[1]))):
#         if cmap == 'Greys':
#             color='white' if cm[i,j] > thresh else 'black'
#         elif cmap =='viridis':
#             color = 'black'if cm[i,j] > thresh else 'white'
#         if labeling:
#             plt.text(j+0.5,i+0.5, format(cm[i,j], fmt),
#                     horizontalalignment='center',
#                     verticalalignment = 'center',
#                     color=color)
#     ax.set_ylabel('True label', fontsize=20)
#     ax.set_xlabel('Predicted label', fontsize=20)
#     plt.tight_layout()
#     fig.savefig("confusion_matrix_cmap_{}_normalize_{}_sqlabeling{}.png".format(cmap, normalize, labeling))
#     plt.close(fig)
#
# plot_confusion_matrix(confusion_matrix, metric_labels, normalize=True, title='Confusion matrix', cmap='viridis') #labeling = True
# plot_confusion_matrix(confusion_matrix, metric_labels, normalize=True, labeling=False, title='Confusion matrix', cmap='viridis')
# plot_confusion_matrix(confusion_matrix, metric_labels, title='Confusion matrix', cmap='viridis') #normalize=False, labeling=True
# plot_confusion_matrix(confusion_matrix, metric_labels, labeling = False, title='Confusion matrix', cmap='viridis') #normalize=False
# plot_confusion_matrix(confusion_matrix, metric_labels, normalize=True, title='Confusion matrix') #labeling=True, cmap=Greys
# plot_confusion_matrix(confusion_matrix, metric_labels, normalize=True, labeling=False, title='Confusion matrix') #cmap=Greys
# plot_confusion_matrix(confusion_matrix, metric_labels, title='Confusion matrix') #normalize=False, labeling=True, cmap=Greys
# plot_confusion_matrix(confusion_matrix, metric_labels, labeling=False, title='Confusion matrix') #normalize=False, cmap=Greys

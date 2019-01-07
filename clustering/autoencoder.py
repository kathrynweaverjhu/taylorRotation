#!/usr/bin/env python3

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.model_selection as sms
import tensorflow as tf
from tensorflow import keras

print("Hello")
'''loading saved matrices of annotated data'''
npzfile = np.load("/home-3/kweave23@jhu.edu/work/users/kweave23/out/savedMatrices.npz")
print(npzfile.files)

cellTypeIndex = npzfile['cellTypeIndex']
labels = npzfile['labels']
sequences = npzfile['sequences']
RNA_seq = npzfile['RNA_seq']
ATAC_seq = npzfile['ATAC_seq']

print("Loaded")

'''Splitting Training and Testing Data'''
sss = sms.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=68)
for train_index, test_index in sss.split(labels, cellTypeIndex):
    X_labels_train_full, X_labels_test = labels[train_index], labels[test_index]
    X_sequences_train_full, X_sequences_test = sequences[train_index], sequences[test_index]
    X_RNAseq_train_full, X_RNAseq_test = RNA_seq[train_index], RNA_seq[test_index]
    X_ATACseq_train_full, X_ATACseq_test = ATAC_seq[train_index], ATAC_seq[test_index]
    Y_cellTypeIndex_train_full, Y_cellTypeIndex_test = cellTypeIndex[train_index], cellTypeIndex[test_index]

print("split done")
seq2d_train = np.reshape(X_sequences_train_full,(int(X_sequences_train_full.shape[0]), -1))
ATAC2d_train = np.reshape(X_ATACseq_train_full, (int(X_sequences_train_full.shape[0]),1))
merged_train_arrays = np.concatenate((seq2d_train,X_RNAseq_train_full,ATAC2d_train), axis=1).astype(np.int32)

print("Sequences_train_full_shape: ", X_sequences_train_full.shape)
print("Seq_2d_train_shape: ", seq2d_train.shape)
print("ATAC2d_train_shape: ", ATAC2d_train.shape)
print("Merged_train_array_shape:", merged_train_arrays.shape)

seq2d_test = np.reshape(X_sequences_test, (int(X_sequences_test.shape[0]),-1))
ATAC2d_test = np.reshape(X_ATACseq_test, (int(X_sequences_test.shape[0]),1))
merged_test_arrays = np.concatenate((seq2d_test,X_RNAseq_test,ATAC2d_test), axis=1).astype(np.int32)

print("Sequences_test_shape: ", X_sequences_test.shape)
print("Seq_2d_test_shape: ", seq2d_test.shape)
print("ATAC2d_test_shape: ", ATAC2d_test.shape)
print("merged_test_array_shape: ", merged_test_arrays.shape)

X_labels_train_full = X_labels_train_full.astype(np.int32)
X_labels_test = X_labels_test.astype(np.int32)

f = open('testArraysCluster.npz', 'wb')
np.savez(f, Y_cellTypeIndex_test = Y_cellTypeIndex_test, X_labels_test = X_labels_test, merged_test_arrays = merged_test_arrays)
f.close()
print("testArrays file should be saved")

'''input shape and autoencoder architecture_2: 169604 -> 5000 -> 1000 -> 500 -> 250 -> 3 -> 250 -> 500 -> 1000 -> 5000 -> 169604'''
m = tf.keras.Sequential()
m.add(keras.layers.Dense(5000, activation='elu', input_shape = (169604,)))
m.add(keras.layers.Dense(1000, activation='elu'))
m.add(keras.layers.Dense(500, activation='elu'))
m.add(keras.layers.Dense(250, activation='elu'))
m.add(keras.layers.Dense(3, activation='linear', name="code_layer"))
m.add(keras.layers.Dense(250, activation='elu'))
m.add(keras.layers.Dense(500, activation='elu'))
m.add(keras.layers.Dense(1000, activation='elu'))
m.add(keras.layers.Dense(5000, activation='elu'))
m.add(keras.layers.Dense(169604, activation='sigmoid'))

'''data_generator'''
def generator(X, epochs, batchSize):
    sess = tf.InteractiveSession()
        
    placeholder = tf.placeholder(tf.int32, shape=X.shape)
    dataset = tf.data.Dataset.from_tensor_slices((placeholder)).batch(batchSize)
        
    iterator = dataset.make_initializable_iterator()
    nextOutput = iterator.get_next()
    iterator.initializer.run(feed_dict = {placeholder:X})
    for epoch in range(epochs):
        try:
            while True:
                tf.initialize_all_variables().run()
                nextSet = tf.get_default_session().run(nextOutput)
                print(nextSet.shape)
                yield(nextSet, nextSet)
        except tf.errors.OutOfRangeError:
            iterator.initializer.run(feed_dict = {placeholder:X})
        
    sess.close()

'''hyperparameters'''
epochs = 2
batchSize = 600
steps_per_epoch = math.ceil(int(merged_train_arrays.shape[0])/batchSize)

'''compile'''
m.compile(loss='mean_squared_error', optimizer=tf.train.AdamOptimizer())
print("compiled")

'''fit'''
history = m.fit_generator(generator(merged_train_arrays, epochs, batchSize), epochs=epochs, steps_per_epoch=steps_per_epoch, workers=0)

'''3_dimensional visualization and reconstruction'''
encoder = keras.Model(m.input, m.get_layer("code_layer").output)
rep = encoder.predict(merged_train_arrays) #code_layer representation or 3_dimensional visualization
#rec = m.predict(merged_train_arrays) #reconstruction

print(rep.shape)

fig = plt.figure()
ax - Axes3D(fig)
im = ax.scatter(rep[:,0], rep[:,1], rep[:,2],c=X_labels_train_full, cmap='Spectral')
cbar = plt.colorbar(im, ticks=np.arange(max(X_labels_train_full)+1))
plt.show()



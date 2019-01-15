#!/usr/bin/env python3

import numpy as np
import umap
import hdbscan
import sklearn.model_selection as sms
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

''' Loading Annotated Data '''
npzfile = np.load("/home-3/kweave23@jhu.edu/work/users/kweave23/out/savedMatrices.npz")
#npzfile = np.load("/Users/cmdb/taylorRotation/preprocessing/savedMatrices250.npz")
print(npzfile.files)

cellTypeIndex = npzfile['cellTypeIndex']
labels = npzfile['labels']
sequences = npzfile['sequences']
RNA_seq = npzfile['RNA_seq']
ATAC_seq = npzfile['ATAC_seq']

preKnownPseudoClusters = int(max(labels) + 1)
print(preKnownPseudoClusters)

''' Splitting Training and Testing Data '''
sss = sms.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(labels, cellTypeIndex):
    X_labels_train_full, X_labels_test = labels[train_index], labels[test_index]
    X_sequences_train_full, X_sequences_test = sequences[train_index], sequences[test_index]
    X_RNAseq_train_full, X_RNAseq_test = RNA_seq[train_index].astype(np.int32), RNA_seq[test_index].astype(np.int32)
    X_ATACseq_train_full, X_ATACseq_test = ATAC_seq[train_index], ATAC_seq[test_index]
    Y_cellTypeIndex_train_full, Y_cellTypeIndex_test = cellTypeIndex[train_index], cellTypeIndex[test_index]

print("Data is split")

seq2d_train = np.reshape(X_sequences_train_full,(int(X_sequences_train_full.shape[0]), -1))
ATAC2d_train = np.reshape(X_ATACseq_train_full, (int(X_sequences_train_full.shape[0]),1))
merged_train_arrays = np.concatenate((seq2d_train,X_RNAseq_train_full,ATAC2d_train), axis=1).astype(np.int32)

print(X_sequences_train_full.shape)
print(seq2d_train.shape)
print(ATAC2d_train.shape)
print(merged_train_arrays.shape)

seq2d_test = np.reshape(X_sequences_test, (int(X_sequences_test.shape[0]),-1))
ATAC2d_test = np.reshape(X_ATACseq_test, (int(X_sequences_test.shape[0]),1))
merged_test_arrays = np.concatenate((seq2d_test,X_RNAseq_test,ATAC2d_test), axis=1).astype(np.int32)

print(X_sequences_test.shape)
print(seq2d_test.shape)
print(ATAC2d_test.shape)
print(merged_test_arrays.shape)

X_labels_train_full = X_labels_train_full.astype(np.int32)
X_labels_test = X_labels_test.astype(np.int32)

f = open('testArraysClusterAlgorithm.npz', 'wb')
np.savez(f, Y_cellTypeIndex_test = Y_cellTypeIndex_test, X_labels_test = X_labels_test, merged_test_arrays = merged_test_arrays)
f.close()
print("testArrays file should be saved")
    
parameters_a = [15,20,30] #n_neighbors
parameters_b = [10,15,20,25,30] #min_samples
parameters_c = [2,50,100,250,500] #min_cluster_size
print("--2D_Clustering--")
fig, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(45,20), sharey=True)
for parameter_a in parameters_a:
    clusterable_embedding = umap.UMAP(n_neighbors = parameter_a, min_dist = 0, n_components=2, metric = 'canberra', random_state=36).fit_transform(merged_train_arrays)

    image1 = ax1.scatter(clusterable_embedding[:,0], clusterable_embedding[:,1], alpha = 0.5, c=X_labels_train_full, cmap='viridis')
    cbar1 = plt.colorbar(image1, ticks=np.arange(27), ax=ax1)
    ax1.set_title("Following UMAP Embedding:\nColor-IDEAS label",fontsize=20)

    for parameter_b in parameters_b:
        for parameter_c in parameters_c:
            labels_cluster = hdbscan.HDBSCAN(min_samples = parameter_b, min_cluster_size=parameter_c, metric='canberra').fit_predict(clusterable_embedding)
            highestCluster = max(labels_cluster)
	    unique = np.unique(labels_cluster)
            numTicks = len(unique)

            image2 = ax2.scatter(clusterable_embedding[:,0], clusterable_embedding[:,1], alpha=0.5, c=labels_cluster, cmap='viridis')
            cbar2 = plt.colorbar(image2, ticks=np.arange(numTicks), ax=ax2)
            ax2.set_title("Following HDBSCAN:\nColor-cluster", fontsize=20)

            clustered = (labels_cluster >= 0)
            print(np.unique(clustered))
            image3 = ax3.scatter(clusterable_embedding[~clustered,0], clusterable_embedding[~clustered,1],c='gray', alpha=0.5)
            image3 = ax3.scatter(clusterable_embedding[clustered,0], clusterable_embedding[clustered,1], c=labels_cluster[clustered], alpha=0.5, cmap="viridis")
            cbar3 = plt.colorbar(image3, ticks=np.arange(highestCluster+1), ax=ax3)
            ax3.set_title("Following HDBSCAN:\nColor-cluster & noise", fontsize=20)

            fig.savefig("2d_{}_{}_{}_try3.png".format(parameter_a, parameter_b, parameter_c))
            plt.close(fig)
            ax2.cla()
            cbar2.remove()
            ax3.cla()
            cbar3.remove()

            noise = labels_cluster.count(-1)
            print("--Parameters:\nn_neighbors {}\nmin_samples {}\nmin_cluster_size{}".format(parameter_a, parameter_b, parameter_c))
            print("Noise: ", noise)
            print("Max cluster: ",highestCluster)
            print("Len of unique clusters: ",numTicks)
            print("Unique clusters: ",unique)

    ax1.cla()
    cbar1.remove()
#print("--3D_Clustering--")
#
#for parameter_a in parameters_a:
#    clusterable_embedding2 = umap.UMAP(n_neighbors = parameter_a, min_dist = 0, n_components=3, metric = 'canberra', random_state=36).fit_transform(merged_train_arrays)
#    
#    fig, ax = plt.subplots(figsize=(25,25))
#    ax = Axes3D(fig)
#    fig.suptitle("Following UMAP Embedding\nColor-IDEAS label", fontsize=20)
#    
#    def init_1():
#       im = ax.scatter(clusterable_embedding2[:,0], clusterable_embedding2[:,1], clusterable_embedding2[:,2], alpha=0.5, c=X_labels_train_full, cmap='viridis')
#       cbar1 = fig.colorbar(im, ticks=np.arange(27), shrink=0.5)
#       return fig,
#       
#    def animate_1(i):
#        ax.view_init(elev=10, azim=i)
#        return fig,
#        
#    anim = animation.FuncAnimation(fig, animate_1, init_func=init_1, frames=360, interval=20, blit=True)
#    anim.save('basic_animation_3d_{}.mp4'.format(parameter_a), fps=30, extra_args=['-vcodec','libx264'])
#
#    for parameter_b in parameters_b:
#        for parameter_c in parameters_c:
#            
#            labels_cluster2 = hdbscan.HDBSCAN(min_samples = parameter_b, min_cluster_size=parameter_c, metric='canberra').fit_predict(clusterable_embedding2)
#            highestCluster2 = max(labels_cluster2)
#            clustered2 = (labels_cluster2 >=0)
#            numTicks = len(np.unique(labels_cluster2))
#            
#            fig2, ax2 = plt.subplots(figsize=(25,25))
#            ax2 = Axes3D(fig2)
#            fig2.suptitle("Following HDBSCAN:\nColor-cluster", fontsize=20)
#            
#            def init_2():
#                im2 = ax2.scatter(clusterable_embedding2[:,0],clusterable_embedding2[:,1],clusterable_embedding2[:,2], alpha=0.5, c=labels_cluster2,cmap='viridis')
#                cbar2 = fig2.colorbar(im2, ticks=np.arange(numTicks), shrink=0.5)
#                return fig2,
#            
#            def animate_2(i):
#                ax2.view_init(elev=10, azim=i)
#                return fig2,
#                
#
#            anim2 = animation.FuncAnimation(fig2, animate_2, init_func=init_2, frames=360, interval=20, blit=True)
#            anim2.save('basic_animation_3d_{}_{}_{}_clusters.mp4'.format(parameter_a, parameter_b, parameter_c), fps=30, extra_args=['-vcodec','libx264'])
#            
#            fig3, ax3 = plt.subplots(figsize=(25,25))
#            ax3 = Axes3D(fig3)
#            fig3.suptitle("Following HDBSCAN:\nColor-cluster & noise", fontsize=20)
#            def init_3():
#                im3 = ax3.scatter(clusterable_embedding2[~clustered2,0], clusterable_embedding2[~clustered2,1], clusterable_embedding2[~clustered2,2],c='gray', 
#alpha=0.5)
#                im3 = ax3.scatter(clusterable_embedding2[clustered2,0], clusterable_embedding2[clustered2,1], 
#clusterable_embedding2[clustered2,2],c=labels_cluster2[clustered2], alpha=0.5, cmap='viridis')
#                cbar3 = fig3.colorbar(im3, ticks=np.arange(highestCluster2+1), shrink=0.5)
#                return fig3,
#            
#            def animate_3(i):
#                ax3.view_init(elev=10, azim=i)
#                return fig3,
#                
#            anim3 = animation.FuncAnimation(fig3, animate_3, init_func=init_3, frames=360, interval=20, blit=True)
#            anim3.save('basic_animation_3d_{}_{}_{}_noise.mp4'.format(parameter_a, parameter_b, parameter_c), fps=30, extra_args=['-vcodec', 'libx264'])
#            
#            noise = 0
#            
#            for i in range(len(labels_cluster2)):
#                if labels_cluster2[i] == -1:
#                    noise += 1
#            print("--Parameters:\nn_neighbors {}\nmin_samples {}\nmin_cluster_size{}".format(parameter_a, parameter_b, parameter_c))
#            print("Noise: ", noise)
#            print("Max cluster: ",highestCluster2)
#            print("Len of unique clusters: ",len(np.unique(labels_cluster2)))
#            print("Unique clusters: ",np.unique(labels_cluster2))
            


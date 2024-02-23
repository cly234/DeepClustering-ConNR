# coding='utf-8'
"""t-SNE 对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn import datasets
from sklearn.manifold import TSNE
import os
from mpl_toolkits.mplot3d import Axes3D
 
 
def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 4})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
    '''
    
    fig = plt.figure()
    ax = Axes3D(fig)
    for i in range(data.shape[0]):
        ax.scatter(data[i, 0], data[i, 1], data[i, 2],
               c=plt.cm.Set1(label[i] / 10.))
    return fig
    '''
 
 
def get_tsne(mem_features, mem_labels, epoch):
    print('Computing t-SNE embedding')
    result = TSNE(n_components=2, init='pca').fit_transform(mem_features)
    
    t0 = time()
   
    fig = plot_embedding(result, mem_labels,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    save_path = './ckpt/byol_rerank2/save_images/tsne'
    os.makedirs(save_path, exist_ok=True) 
    fig.savefig(save_path+"tsne-{}.jpg".format(epoch))
 
 

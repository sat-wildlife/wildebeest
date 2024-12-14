# -*- coding: utf-8 -*-
"""visualization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hmwvgr-PQDy1i-uExGrfMrrrixAEQx6P
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import pylab
# %matplotlib inline
# %pylab inline
pylab.rcParams['figure.figsize'] = (15, 10)

def visualize_data(data,data2):
    f, axarr = plt.subplots(1,2)
    data = data[:, :, 0:3]
    axarr[0].imshow(data)

    # visualize only RGB bands by percentile
    #data = data[:, :, 0:-1]
#    _ = data[:, :, 0].copy()
#    data[:, :, 0] = data[:, :, 2]
#    data[:, :, 2] = _
#    data = data.astype(np.float)
#    # perform stretching for better visualization
#    for i in range(data.shape[2]):
#        p2, p98 = np.percentile(data[:, :, i], (2, 98))
#        data[:, :, i] = exposure.rescale_intensity(data[:, :, i],
#                                                      in_range=(p2, p98))
#    axarr[0].imshow(data)
    
    axarr[0].set_title("Satellite image")
    plt.xticks([])
    plt.yticks([])
    

    values = np.unique(data2.ravel())
    im = axarr[1].imshow(data2[:,:,0], cmap=plt.cm.gray)
    axarr[1].set_title("Labeled image")
    #colors = [im.cmap(im.norm(value)) for value in values] 
    colors = ['black', 'white']
    #colors = [im.cmap(value) for value in values]
    data2 = ["Non-wildebeest", "wildebeest"]
    patches = [mpatches.Patch(color=colors[i], label=j, edgecolor='black') for i, j in zip(range(len(values)), data2)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks([])
    plt.yticks([])

def visualize_predict(data,data2, data3):
    f, axarr = plt.subplots(1,3)
    data = data[:, :, 0:3]
    axarr[0].imshow(data)

    # visualize only RGB bands by percentile
    #data = data[:, :, 0:-1]
#    _ = data[:, :, 0].copy()
#    data[:, :, 0] = data[:, :, 2]
#    data[:, :, 2] = _
#    data = data.astype(np.float)
#    # perform stretching for better visualization
#    for i in range(data.shape[2]):
#        p2, p98 = np.percentile(data[:, :, i], (2, 98))
#        data[:, :, i] = exposure.rescale_intensity(data[:, :, i],
#                                                      in_range=(p2, p98))
#    axarr[0].imshow(data)
    
    axarr[0].set_title("Satellite image")
    plt.xticks([])
    plt.yticks([])
    

    values = np.unique(data2.ravel())
    im = axarr[1].imshow(data2[:,:,0], cmap=plt.cm.gray)
    axarr[1].set_title("Labeled image")
    #colors = [im.cmap(im.norm(value)) for value in values] 
    # colors = ['black', 'white']
    #colors = [im.cmap(value) for value in values]
    # data2 = ["Non-wildebeest", "wildebeest"]
    # patches = [mpatches.Patch(color=colors[i], label=j, edgecolor='black') for i, j in zip(range(len(values)), data2)]
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks([])
    plt.yticks([])

    values = np.unique(data3.ravel())
    im = axarr[2].imshow(data3[:,:,0], cmap=plt.cm.gray)
    axarr[2].set_title("Predicted image")
    #colors = [im.cmap(im.norm(value)) for value in values] 
    colors = ['black', 'white']
    #colors = [im.cmap(value) for value in values]
    data3 = ["Non-wildebeest", "wildebeest"]
    patches = [mpatches.Patch(color=colors[i], label=j, edgecolor='black') for i, j in zip(range(len(values)), data3)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks([])
    plt.yticks([])


def visualize_prediction(data):
    values = np.unique(data.ravel())
    im = plt.imshow(data[:,:], cmap=plt.cm.gray)
    plt.title("Predicted image")
    #colors = [im.cmap(im.norm(value)) for value in values]
    colors = ['black', 'white']
    #colors = [im.cmap(value) for value in values]
    data = ["Non-wildebeest", "wildebeest"]
    patches = [mpatches.Patch(color=colors[i], label=j, edgecolor='black') for i, j in zip(range(len(values)), data)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks([])
    plt.yticks([])
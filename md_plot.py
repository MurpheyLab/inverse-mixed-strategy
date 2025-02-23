"""
Utility package for visualizing multi-dimensional distributions
Author: Max Muchen Sun
"""

import numpy as np 
import matplotlib.pyplot as plt 


def md_plot(data_list, labels):
    for data in data_list:
        if len(data.shape) != 2:
            raise ValueError("Data must be a 2D numpy array.")
    
    N, D = data_list[0].shape
    fig, axs = plt.subplots(D-1, D-1, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    
    # Hide all plots initially
    for ax in axs.flat:
        ax.set_visible(False)

    # Loop through each unique pair of dimensions
    for i in range(D-1):
        for j in range(i+1, D):
            ax = axs[j-1, i]
            for k, data in enumerate(data_list):
                ax.scatter(data[:, i], data[:, j], color='C'+str(k), alpha=0.2, label=labels[k])
            ax.set_xlabel(f'Dimension {i+1}')
            ax.set_ylabel(f'Dimension {j+1}')
            ax.set_visible(True)
            ax.set_aspect('equal')
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.legend()

    # Hide the upper triangle of the plot matrix
    for i in range(D-1):
        for j in range(i+1, D-1):
            axs[i, j].set_visible(False)
            
    plt.show()

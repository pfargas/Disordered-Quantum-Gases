import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def plot_dispersors(length,dispersors, savefig=False, fig_name="" ,**kwargs):
    """Plots a set of dispersors in the plane.
    Inputs:
        dispersors: numpy array of shape (n, 2) where n is the number of dispersors. Each row contains the x and y coordinates of a dispersor.
        savefig: bool, whether to save the figure or not.
        fig_name: str, the name of the file where the figure will be saved.
        kwargs: additional arguments to be passed to the plt.scatter function.
    Outputs:
        None
    """
    
    if kwargs == {}:
        kwargs = {'c': 'r', 'marker': 'x', 's': 2}
    
    plt.scatter(dispersors[:, 0], dispersors[:, 1], **kwargs)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Dispersors')
    plt.xlim(-length//2-length//10, length//2+length//10)
    plt.ylim(-length//2-length//10, length//2+length//10)
    plt.hlines((-length//2, length//2), -length//2, length//2, colors='k', linestyles='dashed', linewidth=0.5)
    plt.vlines((-length//2, length//2), -length//2, length//2, colors='k', linestyles='dashed', linewidth=0.5)
    if savefig:
        if fig_name == "":
            fig_name = len(dispersors) + 'dispersors' + datetime.now().strftime("%Y%m%d%H%M%S") + '.png'
        if not os.path.exists("./plots"):
            os.makedirs("./plots")    
        fig_name = "./plots/" + fig_name
        plt.savefig(fig_name)
    else:
        plt.show()
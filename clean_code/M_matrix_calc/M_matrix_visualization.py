import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

def plot_real_imag_M_matrix(M_matrix, savefig=False, fig_name="", **kwargs):
    """Plots the real and imaginary parts of the M matrix.
    Inputs:
        M_matrix: numpy array of shape (n, n) where n is the number of dispersors. Contains the M matrix.
        savefig: bool, whether to save the figure or not.
        fig_name: str, the name of the file where the figure will be saved.
        kwargs: additional arguments to be passed to the plt.imshow function.
    Outputs:
        None
    """
    
    if kwargs == {}:
        kwargs = {'cmap': 'viridis'}
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(np.real(M_matrix), **kwargs)
    ax[0].set_title('Real part of M matrix')
    ax[1].imshow(np.imag(M_matrix), **kwargs)
    ax[1].set_title('Imaginary part of M matrix')
    if savefig:
        if fig_name == "":
            fig_name = 'M_matrix' + datetime.now().strftime("%Y%m%d%H%M%S") + '.png'
        if not os.path.exists("./plots"):
            os.makedirs("./plots")
        fig_name = "./plots/" + fig_name
        plt.savefig(fig_name)
    else:
        plt.show()
        
def plot_abs_M_matrix(M_matrix, savefig=False, fig_name="", **kwargs):
    """Plots the absolute value of the M matrix.
    Inputs:
        M_matrix: numpy array of shape (n, n) where n is the number of dispersors. Contains the M matrix.
        savefig: bool, whether to save the figure or not.
        fig_name: str, the name of the file where the figure will be saved.
        kwargs: additional arguments to be passed to the plt.imshow function.
    Outputs:
        None
    """
    
    if kwargs == {}:
        kwargs = {'cmap': 'viridis'}
    
    plt.imshow(np.abs(M_matrix), **kwargs)
    plt.title('Absolute value of M matrix')
    if savefig:
        if fig_name == "":
            fig_name = 'M_matrix' + datetime.now().strftime("%Y%m%d%H%M%S") + '.png'
        if not os.path.exists("./plots"):
            os.makedirs("./plots")
        fig_name = "./plots/" + fig_name
        plt.savefig(fig_name)
    else:
        plt.show()
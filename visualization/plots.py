""" Bundled plotting functionalities.

    @author: j-huthmacher
"""
import io
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from PIL import Image

import scipy.ndimage as ndimage
import seaborn as sns
import numpy as np
import torch
from torch import nn
from sklearn.manifold import TSNE

plt.style.use('seaborn')

# Needed to collect figures per frame and merge them afterwards to a gif.
images = {

}


def create_gif(fig: plt.Figure, path: str, fill: bool = True):
    """ Creates a gif from given figures.

        To create a gif just handover the figures you want to merge into a gif 
        step by step in separeted (consecutive) function calls. The gif is recreated
        after each new figure. This helps to have intermediate results, even when 
        an excpetion occurred somewhere.

        When you want to create two gifs, make sure that you set fill == False for the
        las figure of the first gif.

        Paramters:
            fig: matplotlib.Figure
                Figure that represents one frame/image in the gif.
            path: str
                Path where the final gif is stored.
            fill: bool
                Determines if there are images left. If fill == True this means
                we continue providing figures for a gif. In case fill == False
                we clean the global image arrays.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches="tight")
    buf.seek(0)

    im = Image.open(buf)

    if not path in images:
        images[path] = [im]
    else:
        images[path].append(im)

    im.save(fp=path, format='GIF', append_images=images[path],
            save_all=True, duration=200, loop=0)
    fig.savefig(path.replace(".gif", ".final.png"), dpi=150)

    if not fill:
        images[path].clear()


def eval_plot(x: np.array, y: np.array, enc: nn.Module, clf: nn.Module, loss: dict = {},
              metric: dict = {}, prec: float = 0.005, n_epochs: int = None, model_name: str = "", 
              clf_plot_args: dict = {}, loss_plot_args: dict = {}, metric_plot_args: dict = {}):
    """ Creates an evaluation plot consisting of predicted class boundaries, loss and accuracy.

        Paramters:
            x: np.array
                Feature matrix.
            y: np.array
                Labels for the data in x.
            enc: nn.Module
                Encoder that is trained to encode the high dimensional input for the
                classification with the downstream classifier.
            clf: nn.Module
                Downstream classifier, e.g. simple MLP.
            loss: dict
                Dictionary that contains the losse/s that are plotted. The key in the dictionary
                represents the label in the plot and the value is an array of the loss values.
            metric: dict
                Dictionary that contains the metric/s that are plotted. The key in the dictionary
                represents the label in the plot and the value is an array of the metric values.
            prec: float
                Precision for the class contour if a mesh is generated (Not used at the moment!).
            model_name: str
                Defines the title of the plot.
            clf_plot_args: dict
                Matplotlib arguments that are passed to the constructor of the axis for the
                classification boundaries.
            loss_plot_args: dict
                Matplotllib arguments that are passed to the constructor of the axis for the
                loss curve.
            metric_plot_args: dict
                Matplotllib arguments that are passed to the constructor of the axis for the
                metric curve.
        Return:
            matpliotlib.Figure: Figure of the plot.            
    """
    #### Figure Set Up ####
    width_ratio = 1.5
    height=5
    figsize=(height + (height * width_ratio), height)
    grid = (2, 2)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(grid[0], grid[1], width_ratios =[1, width_ratio])

    fig.suptitle(model_name, fontsize=12)

    #### Class Contour ####
    ax = fig.add_subplot(gs[0:2, 0], **clf_plot_args)
    class_contour(enc(x), y, clf, prec, ax = ax)

    #### Loss Curve ####
    ax = fig.add_subplot(gs[0, 1:], **loss_plot_args)

    for _, name in enumerate(loss):
        l = loss[name]
        if isinstance(l, list):
            l = np.array(l)

        ax.axhline(l.min(), 0, l.shape[0], lw=1, ls=":", c="grey")
        ax.plot(l, label=name)

        if n_epochs is not None:
            ax.set_xlim(0, n_epochs)

        ax.text(x=ax.get_xlim()[1] + 0.5, y= l.min(), s='%.3f (%s)' % (l.min(), name), va="center")
        # ax.text(x=ax.get_xlim()[1] + 0.5, y= ax.get_ylim()[0], s='Value: %s' % (l[-1]), va="center")

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=True, shadow=True, ncol=5)


    #### Accuracy Curve ####
    if metric is not None:
        ax = fig.add_subplot(gs[1, 1:], **metric_plot_args)

        for _, name in enumerate(metric):
            m = metric[name]
            if isinstance(m, list):
                m = np.array(m)        

            ax.axhline(m.max(), 0, m.shape[0], lw=1, ls=":", c="grey")
            
            ax.plot(m, label=name)
            
            if n_epochs is not None:
                ax.set_xlim(0, n_epochs)

            ax.text(x=ax.get_xlim()[1] + 0.5, y= m.max(), s='%.3f (%s)' % (m.max(), name), va="center")
            # ax.text(x=ax.get_xlim()[1] + 0.5, y=ax.get_ylim()[0], s='Value: %s' % (m[-1]), va="center")

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
                  fancybox=True, shadow=True, ncol=5)
    
    fig.tight_layout()

    return fig


def class_contour(x: np.array, y: np.array, clf: nn.Module, prec: float = 0.05,
                  title: str = "Class Contour", ax: plt.Axes = None):
    """ Plots the class contour IF the dimnesions is 2 or lower.
        Otherwise the embeddings are visualized using TSNE.

        Paramters:
            x: np.array
                Feature matrix.
            y: np.array
                Labels for the x.
            clf: nn.Module
                Downstream classifier.
            prec: float
                Precision of the mesh grid (if it is created, i.e. when input dim is <=2)
            title: str
                Title of the plot.
            ax: matplotlib.Axes
                Axes for the plot. If None a new axis is created.
        Return:
            matplotlib.Figure: Figure of the plot IF the axis paramter is None.
            Otherwise nothing is returned.

    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    
    if len(x.shape) != 2:
        x = np.array(list(x))
    if len(y.shape) != 2:
        y = np.array(list(y))

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))  
    
    clf_device = clf.device
    
    mode = ""
    try:
        h = prec  # step size in the mesh
        x_min, x_max = x[:, 0].min() - 0, x[:, 0].max() + min(h, 0)
        y_min, y_max = x[:, 1].min() - 0, x[:, 1].max() + min(h, 0)

        # np.linspace(2.0, 3.0, num=5)
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, num=int((x_max - x_min) // h)),
                            np.linspace(y_min, y_max, num=int((y_max - y_min) // h)))
        
        clf = clf.cpu()
        Z = clf(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).detach().cpu()).detach().cpu().numpy()
        
        Z = np.argmax(Z, axis=1)
        Z = Z.reshape(xx.shape)

        Z = ndimage.gaussian_filter(Z, sigma=1.0, order=0)

        ax.contourf(xx, yy, Z, alpha=0.8, cmap=sns.color_palette("Spectral", as_cmap=True))
    except:
        mode = " - TSNE"
        x = TSNE(n_components=2).fit_transform(x)


    ax.scatter(x[:, 0], x[:, 1], c=y.astype(int), cmap=sns.color_palette("Spectral", as_cmap=True), edgecolors='w')

    ax.set_title(f"{title} (Sample Shape: {x.shape}) {mode}")

    clf = clf.to(clf_device)

    if fig is not None:
        return fig
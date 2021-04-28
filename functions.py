import itertools
import os
from collections import deque

import matplotlib.path
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix)
from statsmodels.graphics.mosaicplot import mosaic


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300, im_path="images/"):
    '''Saves figure to given image path'''
    path = os.path.join(im_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def histograms_plot(df, features, rows, cols):
    '''Creates histogram plots from dataframe'''
    fig = plt.figure(figsize=(20, 20))
    for i, feature in enumerate(features):
        ax = fig.add_subplot(rows, cols, i + 1)
        df[feature].hist(bins=50 if i != 1 else 100, ax=ax)
        ax.set_title("Distribution de '" + feature + "'")

    fig.suptitle("Histogrammes des données", y=1.02)


def rainbowarrow(ax, start, end, cmap="viridis", n=50, lw=3):
    '''Gives arrow a gradient color'''
    cmap = plt.get_cmap(cmap, n)
    # Arrow shaft: LineCollection
    x = np.linspace(start[0], end[0], n)
    y = np.linspace(start[1], end[1], n)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, linewidth=lw)
    lc.set_array(np.linspace(0, 1, n))
    ax.add_collection(lc)
    # Arrow head: Triangle
    tricoords = [(0, -0.4), (0.5, 0), (0, 0.4), (0, -0.4)]
    angle = np.arctan2(end[1] - start[1], end[0] - start[0])
    rot = matplotlib.transforms.Affine2D().rotate(angle)
    tricoords2 = rot.transform(tricoords)
    tri = matplotlib.path.Path(tricoords2, closed=True)
    ax.scatter(end[0], end[1], c=1, s=(2 * lw)**2,
               marker=tri, cmap=cmap, vmin=0)
    ax.autoscale_view()


def plot_corr_circle(data, pca, comp1, comp2):
    '''Plots correlation circle from results of PCA'''
    coord1 = pca.components_[comp1 - 1] * \
        np.sqrt(pca.explained_variance_[comp1 - 1])
    coord2 = pca.components_[comp2 - 1] * \
        np.sqrt(pca.explained_variance_[comp2 - 1])

    cmap = sns.color_palette("flare", as_cmap=True)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    for i, j, nom in zip(coord1, coord2, data.columns):
        plt.text(i, j, nom)
        rainbowarrow(ax, (0, 0), (i, j), cmap=cmap, lw=2)
    plt.axis((-1.2, 1.2, -1.2, 1.2))

    # cercle
    c = plt.Circle((0, 0), radius=1, color='gray', fill=False)
    ax.add_patch(c)
    test = np.array((np.linspace(0, 1), np.linspace(0, 1)))
    cax = ax.imshow(test, interpolation='nearest', cmap=cmap)
    cax.set_visible(False)
    cbar = fig.colorbar(cax, shrink=.3)
    cbar.set_label('cos2')
    xlab = 'Dim ' + str(comp1) + ': ' + \
        str(pca.explained_variance_ratio_[comp1 - 1] * 100)[:4] + '%'
    ylab = 'Dim ' + str(comp2) + ': ' + \
        str(pca.explained_variance_ratio_[comp2 - 1] * 100)[:4] + '%'
    ax.set_xlabel(xlab, fontsize=12)
    ax.set_ylabel(ylab, fontsize=12)
    # ax.set_title(
    #     "Cercle de corrélation sur les dimensions {} et {}".format(
    #         comp1, comp2)
    # )


def nclass_classification_mosaic_plot(n_classes, results):
    """
    build a mosaic plot from the results of a classification

    parameters:
    n_classes: number of classes
    results: results of the prediction in form of an array of arrays

    In case of 3 classes the prediction could look like
    [[10, 2, 4],
     [1, 12, 3],
     [2, 2, 9]]
    where there is one array for each class and each array holds the
    predictions for each class [class 1, class 2, class 3].

    This is just a prototype including colors for 4 classes.
    """
    class_lists = [range(n_classes)] * 2
    mosaic_tuples = tuple(itertools.product(*class_lists))

    res_list = results[0]
    for i, l in enumerate(results):
        if i == 0:
            pass
        else:
            tmp = deque(l)
            tmp.rotate(-i)
            res_list.extend(tmp)
    data = {t: res_list[i] for i, t in enumerate(mosaic_tuples)}

    _, ax = plt.subplots(figsize=(11, 10))

    font_color = '#2c3e50'
    pallet = [
        '#1F77B4',
        '#E41A1C',
        '#4DAF4A',
        '#984EA3'
    ]
    colors = deque(pallet[:n_classes])
    all_colors = []
    for i in range(n_classes):
        if i > 0:
            colors.rotate(-1)
        all_colors.extend(colors)

    props = {(str(a), str(b)): {
        'color': all_colors[i]} for i, (a, b) in enumerate(mosaic_tuples)}

    def labelizer(k): return ''

    mosaic(data, labelizer=labelizer, properties=props, ax=ax)

    # title_font_dict = {'fontsize': 16, 'color': font_color}
    axis_label_font_dict = {'fontsize': 12, 'color': font_color}

    ax.tick_params(axis="x", which="both", bottom=False, top=False)
    ax.axes.yaxis.set_ticks([])
    ax.tick_params(axis='x', which='major', labelsize=14)

    ax.set_xlabel('Observed Class', fontdict=axis_label_font_dict, labelpad=10)
    ax.set_ylabel('Predicted Class',
                  fontdict=axis_label_font_dict, labelpad=35)

    legend_elements = [
        Patch(facecolor=all_colors[i], label='Class {}'.format(chr(65+i)))
        for i in range(n_classes)
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1.018), fontsize=16)


def plot_cf_matrix(y_true, y_pred, draw_mosaic=True, **kwargs):
    '''PLots confusion matrix and mosaic plot from classification results'''
    classes = ['A', 'B', 'C', 'D']
    cf_mx = confusion_matrix(y_true, y_pred, normalize=None)
    plt.figure(figsize=(10, 10))
    cm = ConfusionMatrixDisplay(cf_mx, display_labels=classes)
    cm.plot(**kwargs)
    if draw_mosaic:
        n_classes = len(classes)
        cf_mx = cf_mx.tolist()
        nclass_classification_mosaic_plot(n_classes, cf_mx)


def reg_to_class(y_pred):
    '''Creates an array with classes from regression results'''
    n = len(y_pred)
    y_reg_to_class = np.zeros(n)
    for i in range(n):
        if 0 <= y_pred[i] < 20:
            y_reg_to_class[i] = 3
        elif 20 <= y_pred[i] < 40:
            y_reg_to_class[i] = 2
        elif 40 <= y_pred[i] < 60:
            y_reg_to_class[i] = 1
        else:
            y_reg_to_class[i] = 0

    return y_reg_to_class


def plot_results(metrics, y_true_reg, y_true_class, y_pred, scores_list):
    '''Plots results of classification and regression'''
    for metric in metrics:
        print(metric.__name__.replace('_', ' ').title(), ":",
              round(metric(y_true_reg, y_pred), 5))
    print("\nConverting regression to classification...")
    y_reg_to_class = reg_to_class(y_pred)
    acc_score = accuracy_score(y_true_class, y_reg_to_class)
    print("Accuracy score:", acc_score, "\n")
    scores_list.append(acc_score)

    plt.figure()
    plt.scatter(y_true_reg, y_pred, edgecolors=(0, 0, 0))
    plt.plot([y_true_reg.min(), y_true_reg.max()], [y_true_reg.min(), y_true_reg.max()],
             'r--', lw=3)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.xlim(y_true_reg.min() - 3, 101)
    plt.ylim(y_pred.min() - 3, 101)


if __name__ == '__main__':
    pass

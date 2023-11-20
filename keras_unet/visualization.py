import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

from tqdm import tqdm 
import random
import colorsys

import numpy as np

def plot_overlay_segmentation(x, y, spacing=(1,1), step=1):
    asp = spacing[0]/spacing[1]
    
    for i in range(0, x.shape[0], step):
    # ground truth and prediction
        true = np.array(np.argmax(y[i],axis=-1), dtype='uint8')

        true_color = np.zeros((true.shape[0], true.shape[1], 4), dtype='float32')
        true_color[true == 1] = (1,0,0, 0.3)
        true_color[true == 2] = (0,1,0, 0.3)
        true_color[true == 3] = (0,0,1, 0.3)
        true_color[true == 4] = (0,1,1, 0.3)
        true_color[true == 5] = (1,0,1, 0.3)
        true_color[true == 6] = (1,1,0, 0.3)
            
        f, axarr = plt.subplots(1, 2, figsize=(10,5))
        axarr[0].imshow(x[i, :, :, 0], 'gray', interpolation='none', aspect=asp)
        axarr[0].set_title("MRI")
        axarr[0].axis('off')

        axarr[1].imshow(x[i, :, :, 0], 'gray', interpolation='none', aspect=asp)
        axarr[1].imshow(true_color, interpolation='none', aspect=asp)
        axarr[1].set_title("Ground Truth")
        axarr[1].axis('off')


def save_segmentation(x, y, save_filename):
    true = np.array(np.argmax(y,axis=-1), dtype='uint8')
    alpha = true.copy()
    alpha[true == 0] = 0
    alpha[true != 0] = 0.3

    true_color = np.zeros((true.shape[0], true.shape[1], 4), dtype='float32')
    true_color[true == 1] = (1,0,0, 0.3)
    true_color[true == 2] = (0,1,0, 0.3)
    true_color[true == 3] = (0,0,1, 0.3)
    true_color[true == 4] = (0,1,1, 0.3)
    true_color[true == 5] = (1,0,1, 0.3)
    true_color[true == 6] = (1,1,0, 0.3)
    
    plt.imshow(x[ :, :, 0], 'gray', interpolation='none')
    plt.imshow(true_color, interpolation='none')
    plt.axis('off')
              
    plt.savefig(save_filename, dpi=100,  bbox_inches='tight')
    plt.close()        
        
def save_overlay_segmentation(x, y, save_path=None, spacing=(1,1), step=1):
    asp = spacing[0]/spacing[1]
    
    for i in tqdm(range(0, x.shape[0], step)):
    # ground truth and prediction
        true = np.array(np.argmax(y[i],axis=-1), dtype='uint8')

        true_color = np.zeros((true.shape[0], true.shape[1], 4), dtype='float32')
        true_color[true == 1] = (1,0,0, 0.3)
        true_color[true == 2] = (0,1,0, 0.3)
        true_color[true == 3] = (0,0,1, 0.3)
        true_color[true == 4] = (0,1,1, 0.3)
        true_color[true == 5] = (1,0,1, 0.3)
        true_color[true == 6] = (1,1,0, 0.3)
      
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(x[i, :, :, 0], 'gray', interpolation='none', aspect=asp)
        axarr[0].set_title("MRI")
        axarr[0].axis('off')

        axarr[1].imshow(x[i, :, :, 0], 'gray', interpolation='none', aspect=asp)
        axarr[1].imshow(true_color, interpolation='none', aspect=asp)
        axarr[1].set_title("Ground Truth")
        axarr[1].axis('off')
              
        f.savefig(save_path + '_' + str(i) + '.png', dpi=300, bbox_inches='tight')

        
        
def plot_compare_segmentation(x, y_true, y_pred, save_path=None, spacing=(1,1), step=1,
                              img_titles=None, pred_titles=None):  
    asp = spacing[0]/spacing[1]

    if img_titles is None:
        img_titles = ['MRI'] * x.shape[0]
    if pred_titles is None:
        pred_titles = ['Prediction'] * x.shape[0]
    
    for i in range(0, x.shape[0], step):
    # ground truth and prediction
        true = np.array(np.argmax(y_true[i],axis=-1), dtype='uint8')
        true_color = np.zeros((true.shape[0], true.shape[1], 4), dtype='float32')
        true_color[true == 1] = (1,0,0, 0.3)
        true_color[true == 2] = (0,1,0, 0.3)
        true_color[true == 3] = (0,0,1, 0.3)
        true_color[true == 4] = (0,1,1, 0.3)
        true_color[true == 5] = (1,0,1, 0.3)
        true_color[true == 6] = (1,1,0, 0.3)
        
        pred = np.array(np.argmax(y_pred[i],axis=-1), dtype='uint8')
        pred_color = np.zeros((pred.shape[0], pred.shape[1], 4), dtype='float32')
        pred_color[pred == 1] = (1,0,0, 0.3)
        pred_color[pred == 2] = (0,1,0, 0.3)
        pred_color[pred == 3] = (0,0,1, 0.3)
        pred_color[pred == 4] = (0,1,1, 0.3)
        pred_color[pred == 5] = (1,0,1, 0.3)
        pred_color[pred == 6] = (1,1,0, 0.3)
        
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(x[i, :, :, 0], 'gray', interpolation='none', aspect=asp)
        axarr[0].set_title(img_titles[i])
        axarr[0].axis('off')
        
        axarr[1].imshow(x[i, :, :, 0], 'gray', interpolation='none', aspect=asp)
        axarr[1].imshow(true_color, interpolation='none', aspect=asp)
        axarr[1].set_title("Ground Truth")
        axarr[1].axis('off')

        axarr[2].imshow(x[i, :, :, 0], 'gray', interpolation='none', aspect = asp)
        axarr[2].imshow(pred_color, interpolation='none', aspect=asp)
        axarr[2].set_title(pred_titles[i])
        axarr[2].axis('off')
    
    
def plot_segm_history(history, metrics=['iou', 'val_iou'], losses=['loss', 'val_loss']):
    # summarize history for iou
    plt.figure(figsize=(12,6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle('metrics over epochs', fontsize=20)
    plt.ylabel('metric', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    #plt.yticks(np.arange(0.3, 1, step=0.02), fontsize=35)
    #plt.xticks(fontsize=35)
    plt.legend(metrics, loc='center right', fontsize=15)
    plt.show()
    # summarize history for loss
    plt.figure(figsize=(12,6))    
    for loss in losses:
        plt.plot(history.history[loss], linewidth=3)
    plt.suptitle('loss over epochs', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    #plt.yticks(np.arange(0, 0.2, step=0.005), fontsize=35)
    #plt.xticks(fontsize=35)
    plt.legend(losses, loc='center right', fontsize=15)
    plt.show()


def plot_superposition(image, mask, ax, alpha=0.5):

    # Display the image
    ax.imshow(image, cmap='gray')

    # Overlay the mask
    ax.imshow(mask, cmap='jet', alpha=alpha)

def visualize_BestWorstOnes(X_test, y_test, y_pred,df, metric, label, nb_show=2, show_best=True, show_worst=False, ):
    ixs_show = []
    ixs_ascend_sort = df.loc[:, (metric, label)].argsort().values  # ascending order
    
    if show_worst:
        ixs_worst = ixs_ascend_sort[:nb_show]
        ixs_show.append(ixs_worst)
    
    if show_best:
        ixs_best = ixs_ascend_sort[-nb_show:]
        ixs_show.append(ixs_best)
    
    ixs_show = np.concatenate(ixs_show)
    print(ixs_show)
    
    # Prepare titles
    metric_values = df.loc[:, (metric, label)][ixs_show].values
    
    titles_for_imgs = [f'MRI index_{ix}' for ix in ixs_show]
    
    titles_for_preds = [f"Pred ({metric}_{label} : {value:.3f})" for value in metric_values]
    
    # Call your plotting function here (plot_compare_segmentation or equivalent)
    # Replace the function call below with your actual plotting function
    plot_compare_segmentation(X_test[ixs_show, ], y_test[ixs_show, ], y_pred[ixs_show, ],
                              " ", spacing=(1, 1), step=1, img_titles=titles_for_imgs,
                              pred_titles=titles_for_preds)

# Example usage:
# visualize_metrics(df, 'Dice', 'all', nb_show=2, show_best=True, show_worst=False)
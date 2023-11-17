import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

from tqdm import tqdm 

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

        
        
def plot_compare_segmentation(x, y_true, y_pred, save_path=None, spacing=(1,1), step=1):  
    asp = spacing[0]/spacing[1]
    
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
        axarr[0].set_title("MRI")
        axarr[0].axis('off')
        
        axarr[1].imshow(x[i, :, :, 0], 'gray', interpolation='none', aspect=asp)
        axarr[1].imshow(true_color, interpolation='none', aspect=asp)
        axarr[1].set_title("Ground Truth")
        axarr[1].axis('off')

        axarr[2].imshow(x[i, :, :, 0], 'gray', interpolation='none', aspect = asp)
        axarr[2].imshow(pred_color, interpolation='none', aspect=asp)
        axarr[2].set_title('Prediction')
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

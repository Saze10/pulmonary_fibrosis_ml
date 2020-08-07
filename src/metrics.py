# Custom metrics i.e. specificity, sensitivity and confusion matrix

import keras.backend as K
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics

# Most of this is actually irrelevant for the OSIC competition
# We will be using laplace_log_likelihood the most, maybe plot_training

# Training plot
def plot_training(model_history, num_epochs, model_num):
    
    plt.style.use('ggplot')
    plt.switch_backend('agg')    

    fig, ax = plt.subplots()
    ax.plot(np.arange(1, num_epochs + 1), model_history.history["loss"], label = "Cross-entropy Loss")
    ax.plot(np.arange(1, num_epochs + 1), model_history.history["acc"], label = "Accuracy")

    axes = plt.gca()
    axes.set_ylim([0, None])

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss / Accuracy")
    
    ax.legend()
    plt.savefig('../../../data/models/model_{}/training_plot_{}.png'.format(str(model_num).zfill(3), \
									str(model_num).zfill(3)))

# Threshold probability vector to binary

def threshold(y_pred, thresh):
    return (y_pred >= thresh) * 1

# Ouput the ROC

def ROC(y_pred, y_true, model_num, plot = True):
    pred_sens = []
    pred_spec = []
    for thresh in range(0, 11):
        thresh /= 10
        sens = sens_value(y_pred, y_true, thresh)
        spec = spec_value(y_pred, y_true, thresh)
        pred_sens.append(sens)
        pred_spec.append(spec)

        print("Threshold: {:.1f}, Sensitivity: {:.3f}, Specificity: {:.3f}".format(thresh, sens, spec))
   
    # Calculate AUROC
    auroc = sklearn.metrics.roc_auc_score(y_true[:, 1], y_pred[:, 1])
    print("\nAUROC of {:.3f}".format(auroc))
    
    if plot:
                
            ## Plot figure

        x = np.linspace(0, 1, 50)
        
        plt.switch_backend('agg')    
        plt.style.use('ggplot')
        fig, ax = plt.subplots()

        # Change the x to (1-spec) for the ROC curve
        roc, = ax.step(x = 1 - np.array(pred_spec), y = np.array(pred_sens), 
		    where = 'post', label = 'ROC curve with AUROC = {:.3f}'.format(auroc))
        equality, = ax.plot(x, x, dashes = [6, 2], label = 'Classification due to chance with AUROC = 0.5')
        
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Charcteristics (ROC) Curve')
	    
        ax.legend()
        plt.savefig('../../../data/models/model_{}/ROC_{}.png'.format(str(model_num).zfill(3), \
									str(model_num).zfill(3)))


# Ouput binary confusion matrix

def confusion_matrix(y_pred, y_true, thresh = 0.5):
    # Threshold to turn probability vector to binary
    y_pred = threshold(y_pred, thresh)
    
    # Taking the True node from the binary output rather than both
    y_pred = y_pred[:, 1]
    y_true = y_true[:, 1]

    not_true = 1 - y_true
    not_pred = 1 - y_pred

    FP = K.sum(not_true * y_pred).eval(session = tf.Session())
    TN = K.sum(not_true * not_pred).eval(session = tf.Session())

    FN = K.sum(y_true * not_pred).eval(session = tf.Session())
    TP = K.sum(y_true * y_pred).eval(session = tf.Session())
    
    print('''
       BINARY CONFUSION MATRIX
            THRESHOLD: {}
	      +-----------------+
    	      |    predicted    |
    +---------+-----------------+
    |actually |True	|False	|
    |---------|---------|-------|
    |True     |{}	|{}	|
    |False    |{}	|{}	|
    +---------+---------+-------+
    '''.format(thresh, TP, FN, FP, TN))

def spec_value(y_pred, y_true, thresh = 0.5):

    """
    specificity:
    y_pred = matrix of labels that are predicted to be true
    y_true = labels that are true by the GT
    not_true = labels that are not true by the GT
    not_pred = labels that are predicted to be not true
    return spec
    """

    # Threshold to turn probability vector to binary
    y_pred = threshold(y_pred, thresh)

    # Taking the True node from the binary output rather than both
    y_pred = y_pred[:, 1]
    y_true = y_true[:, 1]
  
    not_true = 1 - y_true
    not_pred = 1 - y_pred

    FP = K.sum(not_true * y_pred).eval(session = tf.Session())
    TN = K.sum(not_true * not_pred).eval(session = tf.Session())

    spec = TN / (TN + FP)

    return spec


def sens_value(y_pred, y_true, thresh = 0.5):

    """
    sensitivity:
    y_pred = matrix of labels that are predicted to be true
    y_true = labels that are true by the GT
    not_true = labels that are not true by the GT
    not_pred = labels that are predicted to be not true
    return sens
    """

    # Threshold to turn probability vector to binary
    y_pred = threshold(y_pred, thresh)
    
    # Taking the True node from the binary output rather than both
    y_pred = y_pred[:, 1]
    y_true = y_true[:, 1]
    
    # Subtracting the y_true and y_pred from 1 gives the complement 
    not_true = 1 - y_true
    not_pred = 1 - y_pred

    FN = K.sum(y_true * not_pred).eval(session = tf.Session())
    TP = K.sum(y_true * y_pred).eval(session = tf.Session())

    sens = TP / (TP + FN)

    return sens


    ## Tensor metrics



# evaluation metric function
def laplace_log_likelihood(actual_fvc, predicted_fvc, confidence, return_values = False):
    """
    Calculates the modified Laplace Log Likelihood score for this competition.
    """
    sd_clipped = np.maximum(confidence, 70)
    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)
    metric = - ((np.sqrt(2) * delta) / sd_clipped) - np.log(np.sqrt(2) * sd_clipped)

    if return_values:
        return metric
    else:
        return np.mean(metric)



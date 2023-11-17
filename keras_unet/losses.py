from tensorflow.keras import backend as K
import tensorflow as tf


def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.

    Also known as the intersection-over-union loss.

    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.

    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?

    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.

    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.

    # Returns
        The Jaccard distance between the two tensors.

    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)

    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


# Class Dice coefficient averaged over batch
def dice_coef( y_true, y_pred, axis=1, eps=1e-6):
    intersection = tf.reduce_sum(input_tensor=y_pred * y_true, axis=axis)
    union = tf.reduce_sum(input_tensor=y_pred * y_pred + y_true * y_true, axis=axis)
    dice = (2. * intersection + eps) / (union + eps)
    return tf.reduce_mean(input_tensor=dice, axis=0)  # average over batch


def dice_loss(y_true, y_pred):
    n_classes = y_pred.shape[-1]

    flat_logits = tf.reshape(tf.cast(y_pred, tf.float32),
                             [tf.shape(input=y_pred)[0], -1, n_classes])
    flat_labels = tf.reshape(y_true,
                             [tf.shape(input=y_pred)[0], -1, n_classes])

    dice_loss = tf.reduce_mean(input_tensor = 1 - dice_coef( tf.keras.activations.softmax(flat_logits, axis=-1), flat_labels), name='dice_loss_ref')
    return dice_loss



def adaptive_loss(y_true, y_pred, switch_at_threshold=0.3):
    dice_l = dice_loss(y_true, y_pred)

    return tf.cond(
        dice_l < switch_at_threshold,
        true_fn=lambda: dice_l,
        false_fn=lambda: tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
        )

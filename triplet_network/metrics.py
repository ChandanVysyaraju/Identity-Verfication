# import the necessary packages
import tensorflow as tf



@tf.function
def triplet_loss(y_true, y_pred, alpha=0.2):
    # calculate the triplet loss between the true labels and
    # the predicted labels
    y_true = tf.cast(y_true, y_pred.dtype)  # Not used in the loss function
    positive_dist, negative_dist, positive_negative_dist = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    loss = tf.maximum(positive_dist - negative_dist + alpha, 0.)
    loss = tf.reduce_mean(loss)
    return loss



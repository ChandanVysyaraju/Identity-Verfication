# import the necessary packages
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def make_triplets(images, labels, batch_size=256, img_shape=(784,)):
    x_anchors = np.zeros((batch_size, *img_shape))
    x_positives = np.zeros((batch_size, *img_shape))
    x_negatives = np.zeros((batch_size, *img_shape))

    for i in range(0, batch_size):
        # We need to find an anchor, a positive example and a negative example
        random_index = random.randint(0, images.shape[0] - 1)
        x_anchor = images[random_index]
        y = labels[random_index]

        indices_for_pos = np.squeeze(np.where(labels == y), axis=0)
        indices_for_neg = np.squeeze(np.where(labels != y), axis=0)

        x_positive = images[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
        x_negative = images[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]

        x_anchors[i] = x_anchor
        x_positives[i] = x_positive
        x_negatives[i] = x_negative

    return [x_anchors, x_positives, x_negatives]





def euclidean_distance_triplet(embeddings):
    # unpack the vectors into separate lists
    (featsA, featsB, featsC) = tf.split(embeddings, 3, axis=1, name='split')
    # compute the sum of squared distances between the vectors
    distAB = euclidean_distance((featsA, featsB))
    distAC = euclidean_distance((featsA, featsC))
    distBC = euclidean_distance((featsB, featsC))
    distances = K.concatenate([distAB, distAC, distBC], axis=1)
    return distances


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def plot_training(H, plotPath):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    if "loss" in H.history:
        plt.plot(H.history["loss"], label="train_loss")
    if "val_loss" in H.history:
        plt.plot(H.history["val_loss"], label="val_loss")
    if "accuracy" in H.history:
        plt.plot(H.history["accuracy"], label="train_acc")
    if "val_accuracy" in H.history:
        plt.plot(H.history["val_accuracy"], label="val_acc")
    # plt.title("Training Loss and Accuracy")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)

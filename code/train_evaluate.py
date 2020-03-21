# import tensorflow as tf
import os

import tensorflow as tf
from sklearn import preprocessing

from libs import *

global dropOut
global layer1Nodes
global layer2Nodes

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
############Path to the image and the labels file##################
filename = '../data/list_attr_celeba.txt'
imagepath = '../data/img_align_celeba/'
(celebData, labels, imageNames) = dataloader(imagepath, filename)  # load the 70000 dataset
train_num = 50400
val_num = 5600
test_num = 14000

trains_images = celebData[0:train_num, :, :]  # getting the training sets
train_images_labels = labels[0:train_num, :]

val_images = celebData[train_num:train_num + val_num, :, :]  # getting the validation sets
val_images_labels = labels[train_num:train_num + val_num, :]

test_images = celebData[train_num + val_num:train_num + val_num + test_num, :, :]  # getting the training sets
test_images_labels = labels[train_num + val_num:train_num + val_num + test_num, :]

# flattening the input array and reshaping the labels as per requirement of the tnesorflow algo
trains_images = trains_images.reshape([train_num, 784])
val_images = val_images.reshape([val_num, 784])
test_images = test_images.reshape([test_num, 784])
train_images_labels = train_images_labels.reshape([train_num, ])
val_images_labels = val_images_labels.reshape([val_num, ])
test_images_labels = test_images_labels.reshape([test_num, ])

# standardizing the image data set with zero mean and unit standard deviation
trains_images = preprocessing.scale(trains_images)
val_images = preprocessing.scale(val_images)
test_images = preprocessing.scale(test_images)


###########################################################
# function for building the model of the CNN
# function cnn_model_fn(features, labels, mode)
# input : fearures are the 784 input features of each image, labels nd mode in which the CNN model is run
# output : Estimator for the model
def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""

    global dropOut  # globally declaring it be used as shared vairable
    global layer1Nodes
    global layer2Nodes

    # Input Layer
    # Celeb images are 28x28 pixels, and have one color channel
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=layer1Nodes,  # 32
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=layer2Nodes,  # 64
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * layer2Nodes])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    # Add dropout operation; 0.4 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=dropOut, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 2]
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    # onehot_labels = labels
    # loss = tf.losses.softmax_cross_entropy(
    #     onehot_labels=onehot_labels, logits=logits)
    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    number = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='number')
    receiver_tensors = features = {'x': number}
    # features = tf.tile(number, multiples=[1, 2])
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


###########################################################
# function for running CNN on the Celeb data
# function main(dropout, layer1nodes, layer2nodes)
# input : dropout value (default value: 0.4), layer1nodes is the number of units in first hidden layer (default: 32), layer2nodes: nodes in 2nd layer(default:64)
def train_evaluate(dropout=0.4, layer1nodes=32, layer2nodes=64):
    global dropOut
    global layer1Nodes
    global layer2Nodes
    dropOut = dropout
    layer1Nodes = layer1nodes
    layer2Nodes = layer2nodes

    # saving the trained model on this path
    modelName = "../models/celeb_glasses_model"

    # Load training and eval data
    train_data = np.asarray(trains_images, dtype=np.float32)  # Returns np.array
    train_labels = train_images_labels
    eval_data = np.asarray(val_images, dtype=np.float32)  # Returns np.array
    eval_labels = val_images_labels
    test_data = np.asarray(test_images, dtype=np.float32)  # Returns np.array
    test_labels = test_images_labels

    # Create the Estimator
    celeb_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=modelName)

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=1000)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    celeb_classifier.train(
        input_fn=train_input_fn,
        steps=1000)  # epoch count

    celeb_classifier.export_saved_model('../models/saved_model', serving_input_receiver_fn)

    # Evaluate the training set and print results
    Train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        num_epochs=1,
        shuffle=False)
    train_results = celeb_classifier.evaluate(input_fn=Train_input_fn)
    print("Training set accuracy", train_results)

    # Evaluate the validation set and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = celeb_classifier.evaluate(input_fn=eval_input_fn)
    print("validation set accuracy", eval_results)

    # Evaluate the Test set and print results
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_data},
        y=test_labels,
        num_epochs=1,
        shuffle=False)
    test_results = celeb_classifier.evaluate(input_fn=test_input_fn)
    print("Test accuracy", test_results)


train_evaluate()

import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_utils_v2 import *
from common.lr_utils import load_dataset
from BDNN import *

'''
Steps for build a model:
1. Initialize parameters / Define hyperparameters
2. Loop for num_iterations:
    a. Forward propagation
    b. Compute cost function
    c. Backward propagation
    d. Update parameters (using paramters, and grads from backprop)
4. Use trained parameters to predict labels
'''

def two_layer_model(X, Y, layer_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    '''
    Implement a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layer_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    '''
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1]
    (n_x, n_h, n_y) = layer_dims
    parameters = initialaize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        cost = compute_cost(A2, Y)
        dA2 = - (np.divide(Y, A2) - np.divide(1-Y, 1-A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2
        parameters = update_parameters(parameters, grads, learning_rate)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        if print_cost and i%100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)
    
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per tens)")
    plt.title("Learning rate = " + str(learning_rate))
    # plt.show()
    return parameters


def L_layer_model(X, Y, layer_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    '''
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    '''
    np.random.seed(1)
    costs = []
    parameters = initialaize_parameters_deep(layer_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i%100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel("cost")
    plt.xlabel("iterations (per tens)")
    plt.title("Learning rate = " + str(learning_rate))
    # plt.show()
    return parameters

def predict_accuracy_for_2L(X, Y, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
    A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
    accuracy = float((np.dot(Y,A2.T) + np.dot(1-Y,1-A2.T))/float(Y.size)*100)
    return accuracy

def predict_accuracy_for_NL(X, Y, parameters):
    AL, caches = L_model_forward(X, parameters)
    accuracy = float((np.dot(Y,AL.T) + np.dot(1-Y,1-AL.T))/float(Y.size)*100)
    return accuracy

def test_own_image(file_path, num_px, parameters):
    image = np.array(ndimage.imread(file_path, flatten=False))
    my_image = scipy.misc.imresize(image, size = (num_px, num_px)).reshape((num_px*num_px*3, 1))
    AL, cache = L_model_forward(my_image, parameters)
    my_predict = np.squeeze(AL) >= 0.5

if __name__ == "__main__":

    plt.rcParams["figure.figsize"] = (5.0, 4.0) # set default size of plots
    plt.rcParams["image.interpolation"] = "nearest"
    plt.rcParams["image.cmap"] = "gray"

    np.random.seed(1)
    train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

    # Example of a picture
    # index = 11
    # plt.imshow(train_x_orig[index])
    # plt.show()
    # print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

    # Explore your dataset
    # m_train = train_x_orig.shape[0]
    # num_px = train_x_orig[1]
    # m_test = test_x_orig.shape[0]
    # print ("Number of training examples: " + str(m_train))
    # print ("Number of testing examples: " + str(m_test))
    # print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    # print ("train_x_orig shape: " + str(train_x_orig.shape))
    # print ("train_y shape: " + str(train_y.shape))
    # print ("test_x_orig shape: " + str(test_x_orig.shape))
    # print ("test_y shape: " + str(test_y.shape))
    
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1
    train_x = train_x_flatten/255
    test_x = test_x_flatten/255
    print("train_x's shape:" + str(train_x.shape))
    print("test_x's shape:" + str(test_x.shape))

    # # two-layer neural network
    # n_x = 12288
    # n_h = 7
    # n_y = 1
    # # layer_dims = (n_x, n_h, n_y)

    # # parameters = two_layer_model(train_x, train_y, layer_dims, num_iterations = 2500, print_cost = True)

    # # predictions_train = predict_accuracy_for_2L(train_x, train_y, parameters)
    # # print(predictions_train)

    # # predictions_test = predict_accuracy_for_2L(test_x, test_y, parameters)
    # # print(predictions_test)

    # layer_dims = [12288, 20, 7, 5, 1]
    # parameters = L_layer_model(train_x, train_y, layer_dims, num_iterations = 2500, print_cost = True)
    # predictions_train = predict_accuracy_for_NL(train_x, train_y, parameters)
    # print(predictions_train)
    
    # predictions_test = predict_accuracy_for_NL(test_x, test_y, parameters)
    # print(predictions_test)



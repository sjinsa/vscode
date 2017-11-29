import numpy as np
import scipy
import h5py
import matplotlib.pyplot as plt
import math
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

def sigmoid(z):
    s = 1.0/(1+np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[0]
    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    assert(A.shape == (1,X.shape[1]))
    cost = (-1.0/m)*(np.dot(Y, np.log(A).T) + np.dot(1-Y, np.log(1-A).T))
    dw = np.dot(X, (A-Y).T)/m
    db = np.sum(A-Y)/m
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw":dw, "db":db}
    return grads, cost



def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0 :
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        
    
    params = {"w":w, "b":b}
    grads = {"dw":dw, "db":db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    assert(A.shape == (1, m))
    Y_prediction = np.floor(A/0.5)
    assert(Y_prediction.shape == (1, m))
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations = 1900, learning_rate = 0.5, print_cost = False):
    assert(X_train.shape[0] == X_test.shape[0])
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {0} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {0} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs" : costs, 
         "Y_prediction_test" : Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b, 
         "learning_rate" : learning_rate, 
         "num_iterations" : num_iterations}
    return d




if __name__ == "__main__":
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = 0.5, print_cost = False)
    # index = 10
    # plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
    # print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") +  "\" picture.")

    # costs = np.squeeze(d["costs"])
    # print(costs)
    # plt.plot(costs)
    # plt.ylabel("cost")
    # plt.xlabel("iterations (per hundreds)")
    # plt.title("Learning rate =" + str(d["learning_rate"]))
    # plt.show()


    # learning_rates = [0.01, 0.001, 0.0001]
    # models = {}
    # for i in learning_rates:
    #     print('learning rate is: ' + str(i))
    #     models[str(i)] = model(train_set_x, train_set_y, test_set_x,  test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    #     print("\n" + "-----------------------------------" + "\n")
    # for i in learning_rates:
    #     plt.plot(np.squeeze(models[str(i)]["costs"]), label = str(models[str(i)]["learning_rate"]))
    
    # plt.ylabel("cost")
    # plt.xlabel("iterations")

    # legend = plt.legend(loc = "upper right", shadow=True)
    # frame = legend.get_frame()
    # frame.set_facecolor('0.90')

    # plt.show()
    #C:\Users\Public\Pictures\Sample Pictures\Chry.jpg
    accuracys = []
    for index in range(10):
        my_image = "C:\\Users\Public\Pictures\Sample Pictures\\cat" + str(index) + ".jpg"
        image = np.array(ndimage.imread(my_image, flatten=False))
        my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px*num_px*3)).T
        my_image = my_image/255
        my_predicted_image = predict(d["w"], d["b"], my_image)
        accuracys.append(my_predicted_image)
        print("index = " + str(index) + " y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

    sum = np.sum(accuracys)
    print("real accuracy: {} %".format(sum/len(accuracys) * 100))
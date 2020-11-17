from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from data import load_mnist
import matplotlib.pyplot as plt
np.random.seed(654)

(xtrain, ytrain), (xtest, ytest), size = load_mnist(final=True)
xtrain = xtrain/255
xest = xtest/255

ytrain_cat = np.zeros((ytrain.size, ytrain.max()+1))
ytrain_cat[np.arange(ytrain.size),ytrain] = 1
ytest_cat = np.zeros((ytest.size, ytest.max()+1))
ytest_cat[np.arange(ytest.size),ytest] = 1

m = xtrain.shape[0]

X_train, X_test = xtrain.T, xtest.T
Y_train, Y_test = ytrain_cat.T, ytest_cat.T
shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]

def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s

def compute_loss(Y, Y_hat):

    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum

    return L

def forward_pass_tensor(X, params):

    cache = {}

    cache["Z1"] = np.matmul(params["W1"], X) + params["b1"]
    cache["A1"] = sigmoid(cache["Z1"])
    cache["Z2"] = np.matmul(params["W2"], cache["A1"]) + params["b2"]
    cache["A2"] = np.exp(cache["Z2"]) / np.sum(np.exp(cache["Z2"]), axis=0)

    return cache

def back_propagate(X, Y, params, cache):

    dZ2 = cache["A2"] - Y
    dW2 = (1./m_batch) * np.matmul(dZ2, cache["A1"].T)
    db2 = (1./m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params["W2"].T, dZ2)
    dZ1 = dA1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))
    dW1 = (1./m_batch) * np.matmul(dZ1, X.T)
    db1 = (1./m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads

# hyperparameters
learning_rate = 0.05
batch_size = 32
batches = -(-m // batch_size)
epochs = 150
# initialization
params = { "W1": np.random.uniform(-1, 1, size=(300,784))* np.sqrt(1. / 784),
           "b1": np.zeros(shape=(300,1)) * np.sqrt(1. / 784),
           "W2": np.random.uniform(-1, 1, size=(10,300)) * np.sqrt(1. / 300),
           "b2": np.zeros(shape=(10,1)) * np.sqrt(1. / 300)}
train_cost_list = []
test_cost_list = []
# train
for i in range(epochs):

    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train[:, permutation]

    for j in range(batches):

        begin = j * batch_size
        end = min(begin + batch_size, X_train.shape[1] - 1)
        X = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        cache = forward_pass_tensor(X, params)
        grads = back_propagate(X, Y, params, cache)

        params["W1"] = params["W1"] - learning_rate * grads["dW1"]
        params["b1"] = params["b1"] - learning_rate * grads["db1"]
        params["W2"] = params["W2"] - learning_rate * grads["dW2"]
        params["b2"] = params["b2"] - learning_rate * grads["db2"]

    cache = forward_pass_tensor(X_train, params)
    train_cost = compute_loss(Y_train, cache["A2"])
    train_cost_list.append(train_cost)
    cache = forward_pass_tensor(X_test, params)
    test_cost = compute_loss(Y_test, cache["A2"])
    test_cost_list.append(test_cost)
    print("Epoch {}: training cost = {}, test cost = {}".format(i+1 ,train_cost, test_cost))

cache = forward_pass_tensor(X_test, params)
predictions = np.argmax(cache["A2"], axis=0)
labels = np.argmax(Y_test, axis=0)
print(classification_report(predictions, labels))


plt.plot(train_cost_list, label='Training')
plt.plot(test_cost_list, label='Validation')
plt.legend()
plt.show()

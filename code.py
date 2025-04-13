import numpy as np
import time
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns

def initialize_parameters(layer_dims):
    """
    Initializes parameters for a multi-layer neural network using He initialization.
    Parameters:
        layer_dims: Dimensions of each layer in the network.
    Returns:
        parameters: Dictionary containing initialized weights and biases for each layer.
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2. / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_forward(A, W, b):
    """
    Performs the linear part of a layer's forward propagation.
    Parameters:
        A: Activations from previous layer.
        W: Weights matrix.
        b: Bias vector.
    Returns:
        Z: Pre-activation parameter.
        linear_cache: Cache containing A, W, b for backpropagation.
    """
    Z = np.dot(W, A) + b
    linear_cache = {"A": A, "W": W, "b": b}

    return Z, linear_cache

def softmax(Z):
    """
    Applies the softmax function to the input array.
    Parameters:
        Z: Input array.
    Returns:
        A: Softmax output.
        activation_cache: Input array for use in backpropagation.
    """
    exp_Z = np.exp(Z - np.max(Z, axis=0))
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    activation_cache = Z

    return A, activation_cache

def relu(Z):
    """
    Applies the ReLU function to the input array.
    Parameters:
        Z: Input array.
    Returns:
        A: ReLU output.
        activation_cache: Input array for use in backpropagation.
    """
    A = np.maximum(0, Z)
    activation_cache = Z

    return A, activation_cache

def linear_activation_forward(A_prev, W, B, activation):
    """
    Performs forward propagation for the LINEAR->ACTIVATION layer.
    Parameters:
        A_prev: Activations from previous layer.
        W, B : Weights and biases.
        activation: The activation to use.

    Returns:
        A: Output of the activation function.
        cache: Cache containing linear_cache and activation_cache.
    """
    Z, linear_cache = linear_forward(A_prev, W, B)

    if activation == "relu":
        A, activation_cache = relu(Z)
    elif activation == "softmax":
        A, activation_cache = softmax(Z)

    cache = {"linear_cache": linear_cache, "activation_cache": activation_cache}

    return A, cache

def apply_batchnorm(A):
    """
    Applies batch normalization to the activations of a layer before activation function.
    Parameters:
        A: Activations from the current layer.
    Returns:
        NA: Normalized activations.
    """
    mean = np.mean(A, axis=1, keepdims=True)
    var = np.var(A, axis=1, keepdims=True)
    NA = (A - mean) / np.sqrt(var + 0.000001)

    return NA

def L_model_forward(X, parameters, use_batchnorm):
    """
    Implements forward propagation for the [LINEAR->BATCHNORM?->RELU]*(L-1)->LINEAR->SOFTMAX computation.
    Parameters:
        X: Data input.
        parameters: Parameters of the model.
        use_batchnorm: If True, applies batch normalization.
    Returns:
        AL: Last post-activation value.
        caches: List of caches from each linear_activation_forward().
    """
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A, linear_cache = linear_activation_forward(A_prev, W, b, activation='relu')
        if use_batchnorm:
            A = apply_batchnorm(A)
        caches.append(linear_cache )

    W = parameters['W' + str(L)]
    b = parameters['b' + str(L)]
    AL, linear_cache  = linear_activation_forward(A, W, b, activation='softmax')
    caches.append(linear_cache )

    return AL, caches

def compute_cost(AL, Y, parameters=None, lambd=0.00001):
    """
    Compute the cost with or without L2 regularization.
    Parameters:
        AL: Probabilities vector, output of the forward propagation.
        Y: True label vector.
        parameters: Parameters of the model, required for L2 regularization.
        lambd: Regularization hyperparameter.
    Returns:
        cost: Cross-entropy cost.
    """
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(AL + 0.000001))
    if parameters is not None:
        L2_regularization_cost = 0.0
        L = len(parameters) // 2
        for l in range(1, L + 1):
            L2_regularization_cost += np.sum(np.square(parameters['W' + str(l)]))
        L2_regularization_cost = (lambd / (2 * m)) * L2_regularization_cost
        cost += L2_regularization_cost

    return cost

def Linear_backward(dZ, cache):
    """
    Performs linear portion of backward propagation for a single layer (layer l).
    Parameters:
        dZ: Gradient of the cost with respect to the linear output of current layer l.
        cache: Tuple of values (A_prev, W, b) coming from the forward propagation in the current layer.
    Returns:
        dA_prev: Gradient of the cost with respect to the activation.
        dW: Gradient of the cost with respect to W.
        db: Gradient of the cost with respect to b.
    """
    A_prev, W, b = cache['A'], cache['W'], cache['b']
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation, y=None):
    """
    Performs the backward propagation for the LINEAR->ACTIVATION layer.
    Parameters:
        dA: Post-activation gradient for current layer l.
        cache: Tuple of values (linear_cache, activation_cache) from forward propagation in current layer.
        activation: The activation to use.
        y: True labels vector.
    Returns:
        dA_prev: Gradient of cost with respect to activation of previous layer.
        dW: Gradient of cost with respect to weights of current layer.
        db: Gradient of cost with respect to biases of current layer.
    """
    linear_cache, activation_cache = cache['linear_cache'], cache['activation_cache']
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, y)

    dA_prev, dW, db = Linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

def relu_backward(dA, activation_cache):
    """
    Implements the backward propagation for a single RELU unit.
    Parameters:
        dA: Post-activation gradient.
        activation_cache: 'Z' value from forward propagation.
    Returns:
        dZ: Gradient of the cost with respect to Z.
    """
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0

    return dZ

def softmax_backward(dA, activation_cache):
    """
    Implements the backward propagation for softmax function.
    Parameters:
        dA: Post-activation gradient.
        activation_cache: 'Z' value from forward propagation.
    Returns:
        dZ: Gradient of the cost with respect to Z.
    """
    dZ = dA - activation_cache

    return dZ

def L_model_backward(AL, Y, caches, parameters=None, lambd=0.00001):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SOFTMAX group.
    Parameters:
        AL: Probabilities vector, output of the forward propagation.
        Y: True label vector.
        caches: List of caches containing every cache of linear_activation_forward().
        parameters: Parameters of the model, required for adding L2 regularization.
        lambd: Regularization hyperparameter.
    Returns:
        grads: A dictionary with the gradients.
    """
    grads = {}
    L = len(caches)
    m = AL.shape[1]

    current_cache = caches[L-1]
    dA, dW, db = linear_activation_backward(AL, current_cache, activation="softmax",y=Y)

    grads["dW" + str(L)] = dW

    if parameters is not None:
      grads["dW" + str(L)] += lambd * parameters["W" + str(L)]


    grads["dA" + str(L-1)] = dA
    grads["db" + str(L)] = db

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        if parameters is not None:
          grads["dW" + str(l + 1)] += lambd * parameters["W" + str(l + 1)]
        grads["db" + str(l + 1)] = db_temp

    return grads

def Update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent.
    Parameters:
        parameters: Parameters of the model.
        grads: Gradients of the parameters.
        learning_rate: Learning rate.
    Returns:
        parameters: Updated parameters.
    """
    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]

    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False, use_L2 = False, lambd=0.00001):
    """
    Implements a multi-layer neural network model with optional batch normalization and L2 regularization.
    Parameters:
        X: Input data, shape (number of examples, num_px * num_px * 3).
        Y: True "label" vector (e.g., containing 0 through 9 if 10 classes), shape (1, number of examples).
        layers_dims: List containing the input size and each layer size.
        learning_rate: The learning rate, scalar.
        num_iterations: Number of iterations of the optimization loop.
        batch_size: Size of a mini batch.
        use_batchnorm: If True, apply batch normalization after the activation of each layer except the output layer.
        use_L2: If True, add L2 regularization to the cost.
        lambd: Regularization hyperparameter.

    Returns:
        best_parameters: Parameters learnt by the model (weights and biases).
        costs: List of costs computed during the training, this will be used to plot the learning curve.
        val_costs: List of costs computed on the validation set during training.
        val_costs_every_100: List of costs computed on the validation set every 100 steps.
        epoch_count: Number of epochs model was trained for.
        val_X: Validation set input data.
        val_y: Validation set true labels.
        train_X: Training set input data.
        train_y: Training set true labels.
    """
    np.random.seed(42)
    costs = []
    val_costs = []
    val_costs_every_100 = []
    times_100=[]
    parameters = initialize_parameters(layers_dims)

    m = X.shape[0]
    validation_size = int(m * 0.2)
    indices = np.random.permutation(m)

    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    X_train = X[training_indices]
    Y_train = Y[training_indices]
    X_val = X[validation_indices]
    Y_val = Y[validation_indices]

    train_X = (X_train.reshape(X_train.shape[0], -1) / 255).T
    val_X = (X_val.reshape(X_val.shape[0], -1) / 255).T
    train_y = Y_train.T
    val_y = Y_val.T

    no_improvement_count = 0
    best_val_cost = float('inf')
    val_cost = float('inf')
    best_parameters = None
    treshold = 0.0005
    epoch_count = 0
    start_time = time.time()

    m = train_X.shape[1]

    iterations_per_epoch = m // batch_size

    for i in range(num_iterations):
        permutation = np.random.permutation(m)
        X_shuffled = train_X[:, permutation]
        Y_shuffled = train_y[:, permutation]

        for k in range(0, m, batch_size):
            X_batch = X_shuffled[:, k:k+batch_size]
            Y_batch = Y_shuffled[:, k:k+batch_size]

            AL, caches = L_model_forward(X_batch, parameters, use_batchnorm)
            if use_L2:
              cost = compute_cost(AL, Y_batch, parameters)
              grads = L_model_backward(AL, Y_batch, caches, parameters)
            else:
              cost = compute_cost(AL, Y_batch)
              grads = L_model_backward(AL, Y_batch, caches)

            parameters = Update_parameters(parameters, grads, learning_rate)


            current_step = ((i) * iterations_per_epoch) + ( k // batch_size)

            val_cost = 0.0
            AL_val, _ = L_model_forward(val_X, parameters, use_batchnorm)
            if use_L2:
              val_cost = compute_cost(AL_val, val_y, parameters)
            else:
              val_cost = compute_cost(AL_val, val_y)
            val_costs.append(val_cost)

            if current_step % 100 == 0:
              costs.append(cost)
              val_costs_every_100.append(val_cost)
              current_time = time.time()
              calc_time = current_time - start_time
              times_100.append(calc_time)
              print(f"Epoch number: {i}, Iteration number: {current_step}, Train Cost: {cost:.6f}, Val Cost: {val_cost:.6f}")


            if val_cost <= 0.25:
                treshold = 0.005

            if time.time() - start_time > 60 * 60:
                    treshold = 0.01


            if len(val_costs) > 1 and val_costs[-2] - val_costs[-1] < treshold:
                no_improvement_count += 1
            else:
                no_improvement_count = 0

            if no_improvement_count >= 100:
                print(f"Stopping early at epoch {i} due to no improvement in validation cost.")
                print(f"Training completed in {current_step} training steps.")
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Training time: {elapsed_time:.2f} seconds.")
                iteration_count = i
                return parameters, costs, val_costs, val_costs_every_100,times_100, epoch_count , val_X, val_y , train_X, train_y

        epoch_count = i + 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Training completed in {current_step} training steps.")
    print(f"Training completed in {epoch_count} epochs.")
    print(f"Training time: {elapsed_time:.2f} seconds.")

    return best_parameters, costs, val_costs, val_costs_every_100,times_100, epoch_count , val_X, val_y, train_X, train_y

def Predict(X, Y, parameters, use_batchnorm):
  probas, caches = L_model_forward(X, parameters, use_batchnorm=use_batchnorm)

  predictions = np.argmax(probas, axis=0)
  labels = np.argmax(Y, axis=0)

  accuracy = np.mean(predictions == labels)

  return accuracy



def plot_results(train_costs, val_costs):
    """
    Plots the training and validation costs over training steps.
    Parameters:
        train_costs: List of costs during the training process.
        val_costs: List of costs on the validation set during the training.
    """
    assert len(train_costs) == len(val_costs), "The lists must be of the same length"

    batches = list(range(len(train_costs)))

    plt.figure(figsize=(10, 6))

    plt.plot(batches, train_costs, label='Training Cost')

    plt.plot(batches, val_costs, color='red', label='Validation Cost')

    plt.title("Costs per Training Step (in hundreds)")
    plt.xlabel("Training Step (in hundreds)")
    plt.ylabel("Cost")

    plt.legend()

    plt.grid(True)

    plt.show()


def plot_batch(train_costs, val_costs,time,name):
    """
    Plots the training and validation loss over time with specific batch configuration.

    Parameters:
        train_costs: List of training loss values recorded at each time step.
        val_costs: List of validation loss values recorded at each time step.
        time: List of time steps corresponding to each recorded cost.
        name: Descriptor for the batch configuration used during the training.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(time, train_costs, label='Training Loss', marker='o')
    plt.plot(time, val_costs, label='Validation Loss', marker='x')
    plt.xlabel('Time Steps')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss Progress over Time , {name}  batch')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_weights(parameters):
    """
     Plots histograms of model parameters to visualize their distribution.

    Parameters:
        parameters: A dictionary where keys are parameter names (strings) and values are the parameter values (arrays).
    """
    fig, axes = plt.subplots(nrows=len(parameters), ncols=2, figsize=(15, 5 * len(parameters)))
    for i, (key, value) in enumerate(parameters.items()):
        sns.histplot(value.flatten(), ax=axes[i, 0], kde=True)
        axes[i, 0].set_title(f'Histogram of {key}')

    plt.tight_layout()
    plt.show()

def main_Q4_16():
    use_batch_norm=False
    use_L2=False
    num_iteration= 100000
    batch_size=16

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_y = np.eye(10)[train_y]
    test_y = np.eye(10)[test_y]

    test_X = (test_X.reshape(test_X.shape[0], -1) / 255).T
    test_y = test_y.T

    layers = [784, 20, 7, 5, 10]
    learning_rate = 0.009

    best_parameters, costs, val_costs, val_costs_every_100, times_100,epoch_count , val_X, val_y, train_X, train_y = L_layer_model(train_X, train_y, layers, learning_rate, num_iteration, batch_size)

    plot_results(costs, val_costs_every_100)
    plot_weights(best_parameters)

    test_accuracy = Predict(test_X, test_y, best_parameters, use_batchnorm=use_batch_norm)
    print("test accuracy: ", test_accuracy)
    train_accuracy = Predict(train_X, train_y, best_parameters, use_batchnorm=use_batch_norm)
    print("train accuracy: ", train_accuracy)
    val_accuracy = Predict(val_X, val_y, best_parameters, use_batchnorm=use_batch_norm)
    print("val accuracy: ", val_accuracy)

    print(best_parameters)
    plot_batch(costs, val_costs_every_100,times_100,"16")


def main_Q4_4():
    use_batch_norm=False
    use_L2=False
    num_iteration= 100000
    batch_size=4

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_y = np.eye(10)[train_y]
    test_y = np.eye(10)[test_y]

    test_X = (test_X.reshape(test_X.shape[0], -1) / 255).T
    test_y = test_y.T

    layers = [784, 20, 7, 5, 10]
    learning_rate = 0.009

    best_parameters, costs_4, val_costs, val_costs_every_100_4, times_100_4,epoch_count , val_X, val_y, train_X, train_y = L_layer_model(train_X, train_y, layers, learning_rate, num_iteration, batch_size)

    plot_batch(costs_4, val_costs_every_100_4,times_100_4,"4")
    plot_weights(best_parameters)

    test_accuracy = Predict(test_X, test_y, best_parameters, use_batchnorm=use_batch_norm)
    print("test accuracy: ", test_accuracy)
    train_accuracy = Predict(train_X, train_y, best_parameters, use_batchnorm=use_batch_norm)
    print("train accuracy: ", train_accuracy)
    val_accuracy = Predict(val_X, val_y, best_parameters, use_batchnorm=use_batch_norm)
    print("val accuracy: ", val_accuracy)


def main_Q4_32():
    use_batch_norm=False
    use_L2=False
    num_iteration= 100000
    batch_size=32

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_y = np.eye(10)[train_y]
    test_y = np.eye(10)[test_y]

    test_X = (test_X.reshape(test_X.shape[0], -1) / 255).T
    test_y = test_y.T

    layers = [784, 20, 7, 5, 10]
    learning_rate = 0.009

    best_parameters, costs_32, val_costs, val_costs_every_100_32, times_100_32,epoch_count , val_X, val_y, train_X, train_y = L_layer_model(train_X, train_y, layers, learning_rate, num_iteration, batch_size)

    plot_batch(costs_32, val_costs_every_100_32,times_100_32,"32")
    plot_weights(best_parameters)

    test_accuracy = Predict(test_X, test_y, best_parameters, use_batchnorm=use_batch_norm)
    print("test accuracy: ", test_accuracy)
    train_accuracy = Predict(train_X, train_y, best_parameters, use_batchnorm=use_batch_norm)
    print("train accuracy: ", train_accuracy)
    val_accuracy = Predict(val_X, val_y, best_parameters, use_batchnorm=use_batch_norm)
    print("val accuracy: ", val_accuracy)

def main_Q5():
    use_batch_norm=True
    use_L2=False
    num_iteration= 100000
    batch_size=16

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_y = np.eye(10)[train_y]
    test_y = np.eye(10)[test_y]

    test_X = (test_X.reshape(test_X.shape[0], -1) / 255).T
    test_y = test_y.T

    layers = [784, 20, 7, 5, 10]
    learning_rate = 0.009

    best_parameters, costs, val_costs, val_costs_every_100, times_100,epoch_count , val_X, val_y, train_X, train_y = L_layer_model(train_X, train_y, layers, learning_rate, num_iteration, batch_size, use_batch_norm, use_L2)

    plot_results(costs, val_costs_every_100)

    test_accuracy = Predict(test_X, test_y, best_parameters, use_batchnorm=use_batch_norm)
    print("test accuracy: ", test_accuracy)
    train_accuracy = Predict(train_X, train_y, best_parameters, use_batchnorm=use_batch_norm)
    print("train accuracy: ", train_accuracy)
    val_accuracy = Predict(val_X, val_y, best_parameters, use_batchnorm=use_batch_norm)
    print("val accuracy: ", val_accuracy)

    print(best_parameters)


def main_Q6():
    use_batch_norm=False
    use_L2=True
    num_iteration= 100000
    batch_size=16

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_y = np.eye(10)[train_y]
    test_y = np.eye(10)[test_y]

    test_X = (test_X.reshape(test_X.shape[0], -1) / 255).T
    test_y = test_y.T

    layers = [784, 20, 7, 5, 10]
    learning_rate = 0.009

    best_parameters, costs, val_costs, val_costs_every_100, times_100,epoch_count , val_X, val_y, train_X, train_y = L_layer_model(train_X, train_y, layers, learning_rate, num_iteration, batch_size, use_batch_norm, use_L2)

    plot_results(costs, val_costs_every_100)
    plot_weights(best_parameters)

    test_accuracy = Predict(test_X, test_y, best_parameters, use_batchnorm=use_batch_norm)
    print("test accuracy: ", test_accuracy)
    train_accuracy = Predict(train_X, train_y, best_parameters, use_batchnorm=use_batch_norm)
    print("train accuracy: ", train_accuracy)
    val_accuracy = Predict(val_X, val_y, best_parameters, use_batchnorm=use_batch_norm)
    print("val accuracy: ", val_accuracy)

    print(best_parameters)

if __name__ == '__main__':
    main_Q4_16()
    main_Q4_4()
    main_Q4_32()
    main_Q5()
    main_Q6()

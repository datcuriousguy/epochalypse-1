import matplotlib.pyplot as plt
import numpy
import numpy as np
from keras.layers import Dense
from keras.models import Sequential

X = numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
Y = numpy.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

predictionList = []
errorList = []
iterations_list = []

print('lol:')
INPUT = 40

correct_answer = INPUT * 2
bestEpochs = []


def runNetwork():
    """
    Trains a neural network to predict twice the value of a given input.

    The function runs the training process for a predefined number of iterations,
    adjusting the number of epochs for each iteration. It records the prediction
    errors and visualizes the change in error with the increase in the number of epochs.
    """
    iterations = 100
    epochs = 166
    units = 1
    input_dim = 1
    for i in range(iterations):
        INPUT = 40
        model1 = Sequential()
        model1.add(Dense(units=units, input_dim=input_dim))
        model1.compile(loss='mean_squared_error', optimizer='sgd')
        model1.fit(X, Y, epochs=epochs, verbose=0)
        p1 = model1.predict(np.array([INPUT]), verbose=0)
        p1 = float(p1[0][0])

        # THIS IS WHERE THE MAGIC HAPPENS

        epochs = epochs

        # error of the output

        iterations_list.append(i)
        error = correct_answer - p1
        errorList.append(error)
        print(f'{INPUT} x 2 = {p1}               epochs = {epochs}               {(i / iterations) * 100}% done')

    if p1 >= 79:
        bestEpochs.append(p1)

    fig = plt.figure()
    plt.title('change in error with increase in no of epochs over 10 trials')
    ax = plt.axes(projection='3d')
    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.clabel('epochs')
    ax.scatter3D(iterations_list, errorList, input_dim)
    print(f"Achieved {(len(bestEpochs) / iterations) * 100}% Success rate")
    print(f'Best Epochs were: {bestEpochs}')
    plt.show()


runNetwork()

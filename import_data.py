import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tools

label = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprised', 'Neutral']
SIZE = 48


def load_data():
    print("load_data")
    data = pd.read_csv("data/fer2013.csv")
    X_data = data['pixels'].as_matrix()
    D = SIZE ** 2
    N = data.shape[0]

    X = np.zeros((N, D))
    Y = data['emotion'].as_matrix()

    for i in range(0, N, 1):
        tools.print_progress(i, N, prefix='Progress:', suffix='Complete', bar_length=50)
        X[i] += np.array(X_data[i].split(), dtype=float)

    return X, Y


def show_image(image_data):
    print("show_image")
    image = image_data.reshape((SIZE, SIZE))

    plt.imshow(image, cmap='gray')
    plt.show()


def show_image_with_title(image_data, title):
    plt.title(title)
    show_image(image_data)


def prepare_data():
    print("prepare_data")
    X, Y = load_data()
    X = X / X.mean()
    tools.save_numpy_array_to_file(X, "fer_preprocessed_x")
    tools.save_numpy_array_to_file(Y, "fer_preprocessed_y")


def read_prepared_data():
    X = tools.read_numpy_array_from_file("fer_preprocessed_x")
    Y = tools.read_numpy_array_from_file("fer_preprocessed_y")
    return X, Y

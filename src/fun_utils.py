from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split

def load_mnist(csv_filename):
    """Load the MNIST dataset.
    
    input:
        csv_filename: string with the path to the dataset to load
    
    output:    
        X: set of images (numpy array)
        y: labels (numpy array)

    """
    
    data = read_csv(csv_filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """

    Xtr, Xts, ytr, yts = train_test_split(x, y, train_size=0.5, random_state=10, shuffle=True)

    return Xtr, ytr, Xts, yts
    

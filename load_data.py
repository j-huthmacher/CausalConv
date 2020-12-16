""" Data loading, processing and transformation script.

    @author: j-huthmacher
"""
import pandas as pd
import numpy as np


def load_data(name: str = "eeg-eye-state", lim: int = None):
    """ Function to load the desired data set.

        Parameters:
            name: str
                Name of the data set. Currently just one data set is available.
                However, this may change in the future.
            lim: int
                Limit to just get a subset of the data set.
        Return:
            pd.DataFrame, pd.DataFrame: Returns x, y (i.e. features, labels)
    """
    # Resolution 117 seconds
    # Class - 1: eye closed, 0: eye open 
    eeg = pd.read_csv("./data/eeg-eye-state_csv.csv")
    eeg["Class"] = eeg["Class"] - 1  # Normalize labels 
    x = eeg[eeg.columns[:-1]]
    y = eeg[eeg.columns[-1]]

    if lim is None:
        lim = y.shape[0]

    return x[:lim], y[:lim]


def split_data(x: np.array, y: np.array, train_ratio: float = 0.8,
               val_ratio: float = 0.1, mode: int = 2, lim: int = None):
    """ Function to split the data in train and/or validation and/or test set.

        Hint: In general, you can create also a test set evene when you can't set
        a ratio for it. The splitting takes place as follows: E.g. the trian ratio is
        0.8 and the validation ratio is 0.1. This means we use the first 80% of the
        data for the trainings set, the next 10% (i.e. 80% - 90%) for validation, and
        the remaining data points are used for the test set, i.e. here the last 10%.

        Paramters:
            x: np.array
                Feature matrix.
            y: np.array
                Labels for the features.
            train_ratio: float
                Ratio that defines how much of the complete data set is used
                for the trainings set. E.g. 0.8 means we use 80% fore the trainings
                set.
            val_ratio: float
                Ratio that defines how much of the complete data set is used
                for the validation set. E.g. 0.1 means we use 80% fore the
                validation set.
            mode: int
                Determines which sets are returned.
                1 (or lower) --> Only trainings set
                2 --> Train + Val
                3 --> Train + Val + Test
            lim: int
                Limit to just consider a subset for the split.
        Return:
            list: Depending on the mode
            --> train_set or [train_set, val_set] or [train_set, val_set, test_set]
    """
    train_threshold = int(y.shape[0] * train_ratio)
    val_threshold = int(y.shape[0] * (train_ratio + val_ratio))

    if lim is not None:
        train_threshold = int((lim) * train_ratio)
        val_threshold = int((lim) * (train_ratio + val_ratio))
    else:
        lim = y.shape[0]

    sets = []
    sets.append(list(zip(x[:train_threshold], y[:train_threshold])))

    if mode > 1:
        if mode <= 2:
            sets.append(list(zip(x[train_threshold:lim], y[train_threshold:lim])))
        else:
            sets.append(list(zip(x[train_threshold:val_threshold], y[train_threshold:val_threshold])))

    if mode > 2:
        sets.append(list(zip(x[train_threshold:val_threshold], y[train_threshold:val_threshold])))

    return sets[0] if mode <= 1 else sets
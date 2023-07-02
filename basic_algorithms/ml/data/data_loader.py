from pathlib import Path
import numpy as np
import pandas as pd


DATASET_PATH = f'{Path(__file__).parent}/diabetes.csv'


def load_diabetes() -> (np.ndarray, np.ndarray):
    """
    Load diabetes dataset.
    :return: X, Y (numpy arrays)
    """
    df = pd.read_csv(DATASET_PATH)
    data = df.values
    X = data[:, 0:8]
    Y = data[:, 8]
    return X, Y



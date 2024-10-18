import pickle
import os


def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pickle.dump(data, file)

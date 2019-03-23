import urllib.request
import os.path
import gzip
import pickle
import os
from typing import Optional, Dict, Any, Tuple
import numpy as np

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784


def _download(file_name: str):
    file_path = dataset_dir + "/" + file_name

    if os.path.exists(file_path):
        return

    print("Downloading  {0}  ... ".format(file_name))
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")


def download_mnist():
    for v in key_file.values():
        _download(v)


def _load_label(file_name: str):
    file_path: str = dataset_dir + "/" + file_name

    print("Converting + {0} + to NumPy Array ...".format(file_name))
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _load_img(file_name: str):
    file_path: str = dataset_dir + "/" + file_name

    print("Converting + {0} + to NumPy Array ... ".format(file_name))
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data


def _convert_numpy() -> Dict[str, Optional[Any]]:
    data_set: Dict[str, Optional[Any]] = {'train_img': _load_img(key_file['train_img']),
                                          'train_label': _load_label(key_file['train_label']),
                                          'test_img': _load_img(key_file['test_img']),
                                          'test_label': _load_label(key_file['test_label'])}
    return data_set


def init_mnist() -> None:
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done")


def _change_one_hot_label(x):
    t = np.zeros((x.size, 10))
    for idx, row in enumerate(t):
        row[x[idx]] = 1
    return t


def load_mnist(normalize=True, flatten=True, one_hot_label=False) -> \
        Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load MNIST Dataset
    :param normalize:
    :param flatten:
    :param one_hot_label:
    :return:
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        data_set = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            data_set[key] = data_set[key].astype(np.float32)
            data_set[key] /= 255.0

    if one_hot_label:
        data_set['train_label'] = _change_one_hot_label(data_set['train_label'])
        data_set['test_label'] = _change_one_hot_label(data_set['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            data_set[key] = data_set[key].reshapce(-1, 1, 28, 28)

    return (data_set['train_img'], data_set['train_label']), (data_set['test_img'], data_set['test_label'])


if __name__ == '__main__':
    init_mnist()

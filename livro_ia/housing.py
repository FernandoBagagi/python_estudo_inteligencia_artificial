import os
import tarfile
import urllib.request
from zlib import crc32

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

DOWNLOAD_ROOT = 'https://github.com/ageron/handson-ml2/raw/refs/heads/master/'
HOUSING_PATH = os.path.join('livro_ia', 'datasets', 'housing')
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'


def fecth_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, 'housing.tgz')

    urllib.request.urlretrieve(housing_url, tgz_path)

    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    rng = np.random.default_rng(seed=42)
    shuffled_indices = rng.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xFFFFFFFF < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


def main() -> None:
    # fecth_housing_data()

    housing = load_housing_data()

    print('###### HEAD ######')
    print(housing.head())

    print('###### INFO ######')
    housing.info()

    print('###### DESCRIBE ######')
    print(housing.describe())

    print('###### PLOT ######')
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()

    train_set, test_set = split_train_test(housing, 0.2)

    housing_with_id = housing.reset_index()
    housing_with_id['id'] = housing['longitude'] * 1000 + housing['latitude']
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'id')

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


if __name__ == '__main__':
    main()

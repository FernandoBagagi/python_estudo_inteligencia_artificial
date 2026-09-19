import os
import tarfile
import urllib.request
from zlib import crc32

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

DOWNLOAD_ROOT = 'https://github.com/ageron/handson-ml2/raw/refs/heads/master/'
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'
HOUSING_PATH = os.path.join('livro_ia', 'datasets', 'housing')


def fecth_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path, exist_ok=True)

    tgz_path = os.path.join(housing_path, 'housing.tgz')

    urllib.request.urlretrieve(housing_url, tgz_path)

    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)


def load_housing_data(csv_path):

    if not os.path.isfile(csv_path):
        fecth_housing_data()

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


def income_cat_proportions(data):
    return data['income_cat'].value_counts() / len(data)


def main() -> None:

    csv_path = os.path.join(HOUSING_PATH, 'housing.csv')

    housing = load_housing_data(csv_path)

    print('###### HEAD ######')
    print(housing.head())

    print('###### INFO ######')
    housing.info()

    print('###### value counts ocean_proximity ######')
    print(housing['ocean_proximity'].value_counts())

    print('###### DESCRIBE ######')
    print(housing.describe())

    print('###### PLOT ######')
    housing.hist(bins=50, figsize=(20, 15))
    plt.show()

    print('###### split train test numpy ######')
    train_set, test_set = split_train_test(housing, 0.2)
    print('Tamanho conjunto: ', len(housing))
    print('Tamanho conjunto treinamento: ', len(train_set))
    print('Tamanho conjunto teste: ', len(test_set))

    print('###### split train test index ######')
    housing_with_id = housing.reset_index()
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')
    print('Tamanho conjunto treinamento: ', len(train_set))
    print('Tamanho conjunto teste: ', len(test_set))

    print('###### split train test id(longitude, latitude)  ######')
    housing_with_id['id'] = housing['longitude'] * 1000 + housing['latitude']
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'id')
    test_set.head()

    print('###### split train test sklearn  ######')
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    test_set.head()

    housing['median_income'].hist()
    plt.show()

    housing['income_cat'] = pd.cut(
        housing['median_income'],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    print('###### value counts income_cat ######')
    print(housing['income_cat'].value_counts())

    housing['income_cat'].hist()
    plt.show()

    print('###### StratifiedShuffleSplit ######')

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    split_iterator = split.split(housing, housing['income_cat'])
    train_index, test_index = next(split_iterator)
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

    print('value counts income_cat conjunto treinamento')
    print(income_cat_proportions(strat_train_set))

    print('value counts income_cat conjunto teste')
    print(income_cat_proportions(strat_test_set))

    print('value counts income_cat conjunto')
    print(income_cat_proportions(housing))

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame(
        {
            'Overall': income_cat_proportions(housing),
            'Stratified': income_cat_proportions(strat_test_set),
            'Random': income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props['Rand. %error'] = (
        100 * compare_props['Random'] / compare_props['Overall'] - 100
    )
    compare_props['Strat. %error'] = (
        100 * compare_props['Stratified'] / compare_props['Overall'] - 100
    )

    print('compare_props')
    print(compare_props)

    for set_ in (strat_train_set, strat_test_set):
        set_.drop('income_cat', axis=1, inplace=True)


if __name__ == '__main__':
    main()

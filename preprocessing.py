import re
import pandas as pd
import numpy as np
import seaborn as sns


def search(sample):
     error_list = re.findall(r'���\.\d+', sample)
     if len(error_list) > 0:
          unpack = error_list[0]
     else:
          unpack = 'no_error'
     return unpack


path = 'Dataset_1.csv'
ds = pd.read_csv(path, sep=';',low_memory=False, index_col='ENTITY_STABLE_ID')

error_array = np.array([])
for column in ds.columns:
    ds[column] = ds[column].astype(str)
    local_errors = ds[column].apply(search)
    error_array = np.append(error_array, local_errors.unique())

error_idx = np.array([])
for column in ds.columns:
    local_idx = ds[np.isin(ds[column].to_numpy(), error_array)].index.to_numpy()
    error_idx = np.append(error_idx, local_idx)

ds_copy = ds.copy()
ds_copy.drop(error_idx, axis=0, inplace=True)
ds_copy.to_csv('cleaned_ds.csv')

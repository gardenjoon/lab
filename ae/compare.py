import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm


def compare(now_val, compare_val):
    now_val = np.array(now_val.values, dtype=np.float64)
    compare_val = np.array(compare_val.values, dtype=np.float64)
    compared = np.square(now_val - compare_val)
    return np.nansum(compared)

def extract(target_df, compare_df, num):
    extracted = {}
    for target_t, target_v in target_df.iterrows():
        compared_array = pd.DataFrame(index=compare_df.index)

        target_v = np.array(target_v)
        compare_list = np.array(compare_df.values)

        compared_array[target_t] = (np.square(target_v - compare_list)).sum(axis=1)
        compared_array.sort_values(target_t, inplace=True)

        extracted[target_t] = list(compared_array.index)[:num]
    return extracted
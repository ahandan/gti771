from skimage.feature import local_binary_pattern
from skimage import filters
import pandas as pd
import numpy as np
def run():

    fer2013 = pd.read_csv('fer2013.csv')
    df = fer2013
    df['lbp_pixels'] = 0
    df['sobel_pixels'] = 0
    df['lbp_pca'] = 0
    df['sobel_pca'] = 0
    df = df.apply(string_toarray, axis=1)
    df = df.apply(lbp, axis=1)
    df = df.apply(sobel, axis=1)
    df.to_csv('Big_df.csv',sep='\t')

def string_toarray(x):
    string_array = np.array(x['pixels'].split(' '))
    x['pixels'] = np.array([int(s) for s in string_array])
    return x


def lbp(x):
    radius = 1
    n_points = 8 * radius
    float_array = local_binary_pattern(x['pixels'].reshape(48,48), n_points, radius).reshape(-1)
    x['lbp_pixels'] = np.array([int(i) for i in float_array])
    return x

def sobel(x):
    x['sobel_pixels'] = filters.sobel(x['pixels'].reshape(48,48)).reshape(-1)
    return x

if __name__ == "__main__":
    run()
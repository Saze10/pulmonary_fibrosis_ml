import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Image as show_gif
import scipy.misc
import pydicom 
import seaborn as sns


train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

print('Shape of Training data: ', train_df.shape)
print('Shape of Test data: ', test_df.shape)

print(f"The total patient ids are {train_df['Patient'].count()}")
print(f"Number of unique ids are {train_df['Patient'].value_counts().shape[0]} ")


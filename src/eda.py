import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Image as show_gif
import scipy.misc
import pydicom 
import seaborn as sns
from fastai2.basics import *
from fastai2.callback.all import *
from fastai2.vision.all import *
from fastai2.medical.imaging import *

#train_df = pd.read_csv('../data/train.csv')
#test_df = pd.read_csv('../data/test.csv')

#print('Shape of Training data: ', train_df.shape)
#print('Shape of Test data: ', test_df.shape)

#print(f"The total patient ids are {train_df['Patient'].count()}")
#print(f"Number of unique ids are {train_df['Patient'].value_counts().shape[0]} ")

source = Path('../data')
train = source/'train'
train_files = get_dicom_files(train)
info_view = train_files[2133]
dimg = dcmread(info_view)
print(dimg.PixelData[:200])
plt.imshow(dimg.pixel_array)
plt.show()

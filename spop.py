import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datetime import datetime
# from torchvision import datasets, transforms, utils
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset
# import torchvision.utils as vutils

'''
    https://hazy.com/blog/2020/01/31/synthpop-for-python/
    pip install py-synthpop
'''

from synthpop import Synthpop

df = pd.read_csv("/Users/javier/Desktop/SynData/SynthData/Dataset/kag_risk_factors_cervical_cancer.csv")


'''
	Raw data contains missing values ('?'). We convert them into -1.
'''
for column in df.columns:
	df[column].loc[df[column] == '?'] = -1

for column in df.columns:
	df[column] = df[column].astype(float)


my_data_types = {}
for column in df.columns:
	my_data_types[column] = 'float'


spop = Synthpop()

spop.fit(df, dtypes=my_data_types)

synth_df = spop.generate(len(df))

synth_df.head()

string = 'Hydrogen'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
# from datetime import datetime
# from torchvision import datasets, transforms, utils
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision.transforms as transforms
# from torch.utils.data import Dataset
# import torchvision.utils as vutils
# import json
# import PIL
# import PGGANs.py
!ls


os.getcwd()
dataKag = os.listdir(os.path.join("/scratch/st-mgorges-1/jtoledom", "nobackup/SynData/"))[0]
df = pd.read_csv(os.path.join("/scratch/st-mgorges-1/jtoledom", "nobackup/SynData", dataKag))
'''
	Raw data contains missing values ('?'). We convert them into -1.
'''
df.head()
df.info()
df.plot.hist()


for column in df.columns:
	df[column].loc[df[column] == '?'] = -1

ar = df.to_numpy()
ims = ar.astype(float).reshape(858,6,6)
np.concatenate([np.concatenate([ims[i + 5*j,:,:] for i in range(5)], axis=0) for j in range(5)] ,axis=1).shape

hor, ver = 10,10
plt.imshow(np.concatenate([np.concatenate([ims[i + ver*j,:,:] for i in range(ver)], axis=0) for j in range(hor)] ,axis=1))

import random

Te_idx = random.sample(range(858),100)

Tr_idx = list(set(range(858))-set(Te_idx))

os.path.isdir(os.path.join("/scratch/st-mgorges-1/jtoledom", "nobackup/SynData/KagCerCanRisk", "train")) or os.mkdir(os.path.join("/scratch/st-mgorges-1/jtoledom", "nobackup/SynData/KagCerCanRisk", "train"))

tr_df = pd.DataFrame()
tr_df["Indices"] = Tr_idx
tr_df["Files"] = 0

for (i,idx) in enumerate(Tr_idx):
	tr_df["Files"].iloc[i] = f'{idx}.dat'
	np.savetxt(os.path.join("/scratch/st-mgorges-1/jtoledom", "nobackup/SynData/KagCerCanRisk", "train", tr_df["Files"].iloc[i]), ims[idx,:,:], fmt='%.2f')

tr_df.head()
tr_df.to_csv(os.path.join("/scratch/st-mgorges-1/jtoledom", "nobackup/SynData/KagCerCanRisk/train.csv"), index=False)


os.path.isdir(os.path.join("/scratch/st-mgorges-1/jtoledom", "nobackup/SynData/KagCerCanRisk", "test")) or os.mkdir(os.path.join("/scratch/st-mgorges-1/jtoledom", "nobackup/SynData/KagCerCanRisk", "test"))

te_df = pd.DataFrame()
te_df["Indices"] = Te_idx
te_df["Files"] = 0

for (i,idx) in enumerate(Te_idx):
	te_df["Files"].iloc[i] = f'{idx}.dat'
	np.savetxt(os.path.join("/scratch/st-mgorges-1/jtoledom", "nobackup/SynData/KagCerCanRisk", "test", te_df["Files"].iloc[i]), ims[idx,:,:], fmt='%.2f')

te_df.head()
te_df.to_csv(os.path.join("/scratch/st-mgorges-1/jtoledom", "nobackup/SynData/KagCerCanRisk/test.csv"), index=False)

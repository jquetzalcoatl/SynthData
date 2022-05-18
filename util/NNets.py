# string = 'Hydrogen'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datetime import datetime
# from torchvision import datasets, transforms, utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.utils as vutils
import json
import PIL
# import PGGANs.py
# !ls

def select_nn(arg):
	if arg == "Class":
		class nn(SimpleClas):
			pass       
	elif arg == "Disc":
		class nn(SimpleDisc):
			pass
	elif arg == "Gen":
		class nn(SimpleGen):
			pass
	return nn

class SimpleClas(nn.Module):
	def __init__(self, linearSize=6, nclasses=10):
		super(SimpleClas, self).__init__()
		self.nclasses = nclasses
		self.linearSize = linearSize
		self.getLSize()
		self.seqIn = nn.Sequential(*self.genSeq())

	def block(self, chnIn = 1, chnOut = 64, dpout = 0.1, c = 0):
		if c == 1:
			return nn.Sequential(nn.Conv2d(chnIn, chnOut, 3, 1, 1),
									nn.LeakyReLU(negative_slope=0.02),
									nn.BatchNorm2d(chnOut),
									nn.Dropout2d(dpout),
									nn.AvgPool2d(2),
									)
		else:
			return nn.Sequential(nn.Conv2d(chnIn, chnOut, 3, 1, 1),
									nn.LeakyReLU(negative_slope=0.02),
									nn.AvgPool2d(2),
									)

	def genSeq(self, chnIn = 1, chnOut = 64, dpout = 0.1):
		loop = len(self.l) - 3
		l = [self.block(chnIn,chnOut,dpout,1)]
		for i in range(loop):
			l.append(self.block(chnOut*2**i,chnOut*2**(i+1),dpout,0))
		l.append(nn.Sequential(nn.Flatten(), nn.Linear(chnOut*2**(i+1), self.nclasses), nn.LogSoftmax(dim=1),))
		return l

	def getLSize(self):
		self.l=[self.linearSize]
		while self.l[-1] > 0:
			self.l.append(int(np.floor(self.l[-1]/2)))

	def forward(self, x):
		x = self.seqIn(x)
		return x


class SimpleDisc(nn.Module):
	def __init__(self, linearSize = 6):
		super(SimpleDisc, self).__init__()
		self.linearSize = linearSize
		self.getLSize()
		self.seqIn = nn.Sequential(*self.genSeq())

	def block(self, chnIn = 1, chnOut = 64, dpout = 0.1, c = 0):
		if c == 1:
			return nn.Sequential(nn.Conv2d(chnIn, chnOut, 3, 1, 1),
									nn.LeakyReLU(negative_slope=0.02),
									nn.BatchNorm2d(chnOut),
									nn.Dropout2d(dpout),
									nn.AvgPool2d(2),
									)
		else:
			return nn.Sequential(nn.Conv2d(chnIn, chnOut, 3, 1, 1),
									nn.LeakyReLU(negative_slope=0.02),
									nn.AvgPool2d(2),
									)

	def genSeq(self, chnIn = 1, chnOut = 4, dpout = 0.1):
		loop = len(self.l) - 3
		l = [self.block(chnIn,chnOut,dpout,1)]
		for i in range(loop):
			l.append(self.block(chnOut*2**i,chnOut*2**(i+1),dpout,0))
		l.append(nn.Sequential(nn.Flatten(), nn.Linear(chnOut*2**(i+1), 1), nn.LeakyReLU(negative_slope=0.02),))
# 		l.append(nn.Sequential(nn.Flatten(), nn.Linear(chnOut*2**(i+1), 1), nn.Sigmoid(),))
		return l

	def getLSize(self):
		self.l=[self.linearSize]
		while self.l[-1] > 0:
			self.l.append(int(np.floor(self.l[-1]/2)))

	def forward(self, x):
		x = self.seqIn(x)
		return x

class SimpleGen(nn.Module):
	def __init__(self):
		super(SimpleGen, self).__init__()
		self.seqIn = nn.Sequential(nn.Linear(100,1024),
						nn.BatchNorm1d(1024),
# 						nn.ReLU(),
						nn.LeakyReLU(negative_slope=0.02),
						Reshape(256,2,2),

						nn.ConvTranspose2d(256,128,3,1,1),
						nn.Upsample(scale_factor=2),
						nn.BatchNorm2d(128),
						nn.LeakyReLU(negative_slope=0.02),

						nn.ConvTranspose2d(128,1,3,1,0),
# 						nn.BatchNorm2d(1),
						# nn.Hardtanh(),
# 						nn.LeakyReLU(negative_slope=0.001),
						nn.ReLU(),
						)

	def forward(self, x):
		x = self.seqIn(x)
		return x

# class SimpleGen(nn.Module):
# 	def __init__(self):
# 		super(SimpleGen, self).__init__()
# 		self.seqIn = nn.Sequential(nn.Linear(10,32),
# 						nn.BatchNorm1d(32),
# 						nn.LeakyReLU(negative_slope=0.02),
# # 						Reshape(256,2,2),
                                   
# 						nn.Linear(32,64),
# 						nn.BatchNorm1d(64),
# 						nn.LeakyReLU(negative_slope=0.02),
                                   
# 						nn.Linear(64,128),
# 						nn.BatchNorm1d(128),
# 						nn.LeakyReLU(negative_slope=0.02),
                                   
# 						nn.Linear(128,36),
# 						nn.BatchNorm1d(36),
# 						nn.LeakyReLU(negative_slope=0.001),
                                   
# 						Reshape(1,6,6),

# # 						nn.ConvTranspose2d(256,128,3,1,1),
# # 						nn.Upsample(scale_factor=2),
# # 						nn.BatchNorm2d(128),
# # 						nn.LeakyReLU(negative_slope=0.02),

# # 						nn.ConvTranspose2d(128,1,3,1,0),
# 						# nn.BatchNorm2d(1),
# 						# nn.Hardtanh(),
# # 						nn.LeakyReLU(negative_slope=0.001),
# 						)

# 	def forward(self, x):
# 		x = self.seqIn(x)
# 		return x


class Reshape(nn.Module):
	def __init__(self, *args):
		super(Reshape, self).__init__()
		self.shape = args
	def forward(self, x):
		sh = (x.shape[0], ) + self.shape
		return x.view(sh)


class accuracy(object):
	def __init__(self, nLabels=10):
		self.nLabels = nLabels

	def perLabel(self, x,y):
		totalPerLabel = [sum(y == torch.zeros_like(y).to(device))]
		correctPerLabel = [sum((y == x)*(y == torch.zeros_like(y).to(device)))]
		for i in range(1, self.nLabels):
			totalPerLabel.append(sum(y == i*torch.ones_like(y).to(device)))
			correctPerLabel.append(sum((y == x)*(y == i*torch.ones_like(y).to(device))))
		l = torch.Tensor(totalPerLabel)
		correct = torch.Tensor(correctPerLabel)
		r = torch.zeros(self.nLabels)
		for idx in range(len(l)):
			if l[idx] != 0:
				# print(idx, l[idx], correct[idx])
				r[idx] = correct[idx]/l[idx]
			else:
				r[idx] = 0.0
	#     print(r)
		return r, l

	def overAllLabel(self, x,y):
		totalPerLabel = [sum(y == torch.zeros_like(y).to(device))]
		correctPerLabel = [sum((y == x)*(y == torch.zeros_like(y).to(device)))]
		for i in range(1, self.nLabels):
			totalPerLabel.append(sum(y == i*torch.ones_like(y).to(device)))
			correctPerLabel.append(sum((y == x)*(y == i*torch.ones_like(y).to(device))))
		l = torch.Tensor(totalPerLabel)
		correct = torch.Tensor(correctPerLabel)
	#     print(r)
		return sum(correct)/sum(l), l

	def validationPerLabel(self, dataL, model):
		acc = torch.zeros(self.nLabels)
		ds = torch.zeros(self.nLabels)
		for i, data in enumerate(dataL, 0):
	#         r1.zero_grad()
			# Format batch
			x = data[0].to(device)
			y = data[1].to(device)
			output = model(x).max(1)[1]
			r, l = self.perLabel(output,y)
			acc = acc + r
			ds = ds + l
	#     print(i)
		return acc/(i+1)

	def validation(self, dataL, model):
		acc = 0
		ds = 0
		for i, data in enumerate(dataL, 0):
	#         r1.zero_grad()
			# Format batch
			x = data[0].to(device)
			y = data[1].to(device)
			output = model(x).max(1)[1]
			r, l = self.overAllLabel(output,y)
			acc = acc + r
			ds = ds + l
	#     print(i)
		return acc/(i+1)



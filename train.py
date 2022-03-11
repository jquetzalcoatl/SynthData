import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.utils as vutils
import json
import PIL
import logging
import sys
import argparse


sys.path.insert(1, '/scratch/st-mgorges-1/jtoledom/SynthData/util')

from loaders import generateDatasets, inOut#, saveJSON, loadJSON#, MyData
from NNets import SimpleClas, SimpleDisc, SimpleGen
# from tools import accuracy, tools, per_image_error, predVsTarget
from plotter import myPlots

# class DiffSur(SimpleCNN):
# 	pass
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



class train(object):
	def __init__(self, dict, latentSpace=100, std_tr=0.25, s=128, lam=0.0001):
		self.dict=dict
		self.real_label = 1.0
		self.fake_label = 0.0
		self.latentSpace = latentSpace
		self.std_tr = std_tr
		self.s = s
		self.lam = lam
		self.dict["lambda"] = self.lam

	def covLoss(self, xReal, xSyn, alpha=2):
		yReal = xReal - xReal.mean(axis=0)
		ySyn = xSyn - xSyn.mean(axis=0)
		l = ySyn.shape[2] * ySyn.shape[3]
		bs = xReal.shape[0]
		# print(l, yReal.view((bs,l)).transpose(0,1).shape, yReal.view((bs,l)).shape)
		covReal = torch.mm(yReal.view((bs,l)).transpose(0,1), yReal.view((bs,l)))/bs
		covSyn = torch.mm(ySyn.view((bs,l)).transpose(0,1), ySyn.view((bs,l)))/bs
		loss = torch.mean( (covReal - covSyn)**alpha)
		return loss

	def trainClass(self, device, epochs=100, snap=25):
		clas = select_nn('Class')
		clas = clas().to(device)
# 		clas = MLP().to(device)
		criterion = nn.NLLLoss()
		opt = optim.Adam(clas.parameters(), lr=self.dict['lr'])
		trainloader, testloader = generateDatasets(self.dict['PathRoot'][-1], batch_size=self.dict['BatchSize'], num_workers=self.dict['NumWorkers'], std_tr=self.std_tr, s=self.s).getDataLoaders()
		error_list = []
		acc, accTrain = [], []
		for epoch in range(epochs):
			error = 0.0
			for (i, data) in enumerate(trainloader):
				clas.zero_grad()
				x = data[0].to(device)
				y = data[1].to(device)
				yhat = clas(x)
				err = criterion(yhat,y)
				err.backward()
				opt.step()
				error += err.item()
			error_list.append(error/(i+1))
			acc.append(accuracy().validation(testloader, clas).item())
			accTrain.append(accuracy().validation(trainloader, clas).item())
			myPlots().clasPlots(device, error_list, acc, accTrain, epoch)

			if epoch % snap == snap-1 :
				inOut().save_model(clas, 'Class', opt, error_list, epoch, self.dict)
		print("Done!!")
		return error_list, acc, accTrain

	def trainGANs(self, device, epochs=100, snap=25):
		myLog = inOut()
		myLog.logFunc(self.dict)
		disc = select_nn('Disc')
		disc = disc().to(device)
# 		disc = Disc(ngpu, flag, alpha, nc).to(device)
		gen = select_nn('Gen')
		gen = gen().to(device)
# 		gen = Gen(ngpu, flag, alpha, nc).to(device)
		criterion = nn.BCELoss()

		#OPT
		optD = optim.Adam(disc.parameters(), lr=self.dict['lr'], betas=(0.5, 0.99))
		optG = optim.Adam(gen.parameters(), lr=self.dict['lr'], betas=(0.5, 0.99))
		trainloader, testloader = generateDatasets(self.dict['PathRoot'][-1], batch_size=self.dict['BatchSize'], num_workers=self.dict['NumWorkers'], std_tr=self.std_tr, s=self.s).getDataLoaders()
		error_list_D = []
		error_list_G = []

		for epoch in range(epochs):
			errorD = 0.0
			errorG = 0.0
			for (i, data) in enumerate(trainloader):
				disc.zero_grad()
				x = data[0].to(device)
				# y = data[1].to(device)
				yreal = disc(x)
				yR = torch.full((yreal.shape[0],1), self.real_label, device=device)
				z = torch.randn(x.shape[0], self.latentSpace).to(device)
				xfake = gen(z)
				yfake = disc(xfake.detach())
				yF = torch.full((yfake.shape[0],1), self.fake_label, device=device)
				err = criterion(yreal,yR) + criterion(yfake,yF)
				err.backward()
				optD.step()
				errorD += err.item()

				gen.zero_grad()
				yfake = disc(xfake)
				errG = criterion(yfake, yR)
				errG.backward()
				optG.step()
				errorG += errG.item()
				# if i > 2:
				# 	break
			error_list_D.append(errorD/(i+1))
			error_list_G.append(errorG/(i+1))
			myPlots().plotGANs(device, error_list_D, error_list_G, testloader, gen, epoch, self.dict)

			if epoch % snap == snap-1 :
				inOut().save_model(gen, 'Gen', optG, error_list_G, epoch, self.dict)
				inOut().save_model(disc, 'Disc', optD, error_list_D, epoch, self.dict)
		print("Done!!")
		return error_list_D, error_list_G

	def trainWGANs(self, device, epochs=100, snap=25):
		myLog = inOut()
		myLog.logFunc(self.dict)
		disc = select_nn('Disc')
		disc = disc().to(device)
# 		disc = Disc(ngpu, flag, alpha, nc).to(device)
		gen = select_nn('Gen')
		gen = gen().to(device)
# 		gen = Gen(ngpu, flag, alpha, nc).to(device)
		wass = lambda x : x.mean()
		criterion = wass
		reg = self.covLoss
		#OPT
		optD = optim.Adam(disc.parameters(), lr=self.dict['lr'], betas=(0.5, 0.99))
		optG = optim.Adam(gen.parameters(), lr=self.dict['lr'], betas=(0.5, 0.99))
		trainloader, testloader = generateDatasets(self.dict['PathRoot'][-1], batch_size=self.dict['BatchSize'], num_workers=self.dict['NumWorkers'], std_tr=self.std_tr, s=self.s).getDataLoaders()
		error_list_D = []
		error_list_G = []
		error_list_Reg = []

		for epoch in range(epochs):
			errorD = 0.0
			errorG = 0.0
			errorReg = 0.0
			for (i, data) in enumerate(trainloader):
				disc.zero_grad()
				x = data[0].to(device)
				# y = data[1].to(device)
				yreal = disc(x)
				yR = torch.full((yreal.shape[0],1), self.real_label, device=device)
				z = torch.randn(x.shape[0],self.latentSpace).to(device)
				xfake = gen(z)
				yfake = disc(xfake.detach())
				yF = torch.full((yfake.shape[0],1), self.fake_label, device=device)
				err = - criterion(yreal) + criterion(yfake)
				err.backward()
				optD.step()
				errorD += err.item()

				gen.zero_grad()
				yfake = disc(xfake)
				l1 = - criterion(yfake)
				l2 = self.lam * reg(x, xfake)
				# errG = - criterion(yfake) + self.lam * reg(x, xfake)
				errG = l1 + l2
				errG.backward()
				optG.step()
				errorG += errG.item()
				errorReg += l2.item()
				# if i > 2:
				# 	break
			error_list_D.append(errorD/(i+1))
			error_list_G.append(errorG/(i+1))
			error_list_Reg.append(errorReg/(i+1))
			myPlots().plotGANs(device, error_list_D, error_list_G, testloader, gen, epoch, self.dict)
			print(f'ErrG: {error_list_G[-1]}, ErrReg: {error_list_Reg[-1]}')

			if epoch % snap == snap-1 :
				inOut().save_model(gen, 'Gen', optG, error_list_G, epoch, self.dict)
				inOut().save_model(disc, 'Disc', optD, error_list_D, epoch, self.dict)
		print("Done!!")
		return error_list_D, error_list_G

	def trainWGANs_GP(self, device, epochs=100, snap=25, lam = 1.0, gamma = 750):
		myLog = inOut()
		myLog.logFunc(self.dict)
		disc = select_nn('Disc')
		disc = disc().to(device)
# 		disc = Disc(ngpu, flag, alpha, nc).to(device)
		gen = select_nn('Gen')
		gen = gen().to(device)
		wass = lambda x : x.mean()
		criterion = wass
		#OPT
		optD = optim.Adam(disc.parameters(), lr=self.dict['lr'], betas=(0.5, 0.99))
		optG = optim.Adam(gen.parameters(), lr=self.dict['lr'], betas=(0.5, 0.99))
		trainloader, testloader = generateDatasets(self.dict['PathRoot'][-1], batch_size=self.dict['BatchSize'], num_workers=self.dict['NumWorkers'], std_tr=self.std_tr, s=self.s).getDataLoaders()
		error_list_D = []
		error_list_G = []

		for epoch in range(epochs):
			errorD = 0.0
			errorG = 0.0
			for (i, data) in enumerate(trainloader):
				disc.zero_grad()
				x = data[0].to(device)
				# y = data[1].to(device)
				yreal = disc(x)
				yR = torch.full((yreal.shape[0],1), self.real_label, device=device)
				z = torch.randn(x.shape[0], self.latentSpace).to(device)
				xfake = gen(z)
				yfake = disc(xfake.detach())
				yF = torch.full((yfake.shape[0],1), self.fake_label, device=device)
				err = - criterion(yreal) + criterion(yfake) + lam * self.get_gp(x, xfake, disc, gamma )
				err.backward()
				optD.step()
				errorD += err.item()

				gen.zero_grad()
				yfake = disc(xfake)
				errG = - criterion(yfake)
				errG.backward()
				optG.step()
				errorG += errG.item()
				# if i > 2:
				# 	break
			error_list_D.append(errorD/(i+1))
			error_list_G.append(errorG/(i+1))
			myPlots().plotGANs(device, error_list_D, error_list_G, testloader, gen, epoch, self.dict)

			if epoch % snap == snap-1 :
				inOut().save_model(gen, 'Gen', optG, error_list_G, epoch, self.dict)
				inOut().save_model(disc, 'Disc', optD, error_list_D, epoch, self.dict)
		print("Done!!")
		return error_list_D, error_list_G

	def trainWGANs_GPandUnroll(self, device, epochs=100, snap=25, lam = 1.0, gamma = 750, K=1):
# 		disc = Disc().to(device)
# 		discK = Disc().to(device)
# 		gen = Gen().to(device)
		disc = select_nn('Disc')
		disc = disc().to(device)
		discK = select_nn('Disc')
		discK = discK().to(device)
		gen = select_nn('Gen')
		gen = gen().to(device)
		wass = lambda x : x.mean()
		self.criterion = wass
		#OPT
		optDK = optim.Adam(disc.parameters(), lr=self.dict['lr'], betas=(0.5, 0.99))
		optD = optim.Adam(disc.parameters(), lr=self.dict['lr'], betas=(0.5, 0.99))
		optG = optim.Adam(gen.parameters(), lr=self.dict['lr'], betas=(0.5, 0.99))
		trainloader, testloader = generateDatasets(self.dict['PathRoot'][-1], batch_size=self.dict['BatchSize'], num_workers=self.dict['NumWorkers'], std_tr=self.std_tr, s=self.s).getDataLoaders()
		error_list_D = []
		error_list_G = []

		for epoch in range(epochs):
			errorD = 0.0
			errorG = 0.0
			for (i, data) in enumerate(trainloader):
				disc.zero_grad()
				x = data[0].to(device)
				# y = data[1].to(device)
				yreal = disc(x)
				yR = torch.full((yreal.shape[0],1), self.real_label, device=device)
				z = torch.randn(x.shape[0], self.latentSpace).to(device)
				xfake = gen(z)
				yfake = disc(xfake.detach())
				yF = torch.full((yfake.shape[0],1), self.fake_label, device=device)
				err = - self.criterion(yreal) + self.criterion(yfake) + lam * self.get_gp(x, xfake, disc, gamma )
				err.backward()
				optD.step()
				errorD += err.item()

				self.copy_unroll(data, gen, discK, optDK, disc, optD, K, gamma, lam)

				gen.zero_grad()
				yfake = discK(xfake)
				errG = - self.criterion(yfake)
				errG.backward()
				optG.step()
				errorG += errG.item()
				# if i > 2:
				# 	break
			error_list_D.append(errorD/(i+1))
			error_list_G.append(errorG/(i+1))
			myPlots().plotGANs(device, error_list_D, error_list_G, testloader, gen, epoch, self.dict)

			if epoch % snap == snap-1 :
				inOut().save_model(gen, 'Gen', optG, error_list_G, epoch, self.dict)
				inOut().save_model(disc, 'Disc', optD, error_list_D, epoch, self.dict)
		print("Done!!")
		return error_list_D, error_list_G

	def continuetrainWGANs(self, device, epochs=100, snap=25):
		# disc = Disc().to(device)
		# gen = Gen().to(device)
		lastEpoch, _, gen = inOut().load_model( "Gen", self.dict, device)
		_, _, disc = inOut().load_model( "Disc", self.dict, device)
		print(lastEpoch)
		wass = lambda x : x.mean()
		criterion = wass
		reg = self.covLoss
		#OPT
		optD = optim.Adam(disc.parameters(), lr=self.dict['lr'], betas=(0.5, 0.99))
		optG = optim.Adam(gen.parameters(), lr=self.dict['lr'], betas=(0.5, 0.99))
		trainloader, testloader = generateDatasets(self.dict['PathRoot'][-1], batch_size=self.dict['BatchSize'], num_workers=self.dict['NumWorkers'], std_tr=self.std_tr, s=self.s).getDataLoaders()
		error_list_D = self.dict["LossDisc"]
		error_list_G = self.dict["LossGen"]

		for epoch in range(epochs):
			errorD = 0.0
			errorG = 0.0
			for (i, data) in enumerate(trainloader):
				disc.zero_grad()
				x = data[0].to(device)
				# y = data[1].to(device)
				yreal = disc(x)
				yR = torch.full((yreal.shape[0],1), self.real_label, device=device)
				z = torch.randn(x.shape[0],self.latentSpace).to(device)
				xfake = gen(z)
				yfake = disc(xfake.detach())
				yF = torch.full((yfake.shape[0],1), self.fake_label, device=device)
				err = - criterion(yreal) + criterion(yfake)
				err.backward()
				optD.step()
				errorD += err.item()

				gen.zero_grad()
				yfake = disc(xfake)
				errG = - criterion(yfake) + self.lam * reg(x, xfake)
				errG.backward()
				optG.step()
				errorG += errG.item()
				# if i > 2:
				# 	break
			error_list_D.append(errorD/(i+1))
			error_list_G.append(errorG/(i+1))
			myPlots().plotGANs(device, error_list_D, error_list_G, testloader, gen, epoch, self.dict)

			if epoch % snap == snap-1 :
				inOut().save_model(gen, 'Gen', optG, error_list_G, epoch, self.dict)
				inOut().save_model(disc, 'Disc', optD, error_list_D, epoch, self.dict)
		print("Done!!")
		return error_list_D, error_list_G

	def unroll(self, device, data, gen, netDK, optimizerDK, gamma, lam):
		# gp_lambda = 10
		netDK.zero_grad()
		x = data[0].to(device)
		# y = data[1].to(device)
		yreal = netDK(x)
		yR = torch.full((yreal.shape[0],1), self.real_label, device=device)
		z = torch.randn(x.shape[0],self.latentSpace).to(device)
		xfake = gen(z)
		yfake = netDK(xfake.detach())
		yF = torch.full((yfake.shape[0],1), self.fake_label, device=device)
		err = - self.criterion(yreal) + self.criterion(yfake) + lam * self.get_gp(device, x, xfake, netDK, gamma)
		err.backward()
		optimizerDK.step()
		# errorD += err.item()

	def copy_unroll(self, data, gen, netDK, optimizerDK, netD, optimizerD, K=1, gamma = 750, lam = 10):
		optimizerDK.load_state_dict(optimizerD.state_dict())
		#     optimizerDK = copy.deepcopy(optimizerD)
		netDK.load_state_dict(netD.state_dict())
		for i in range(K):
			self.unroll(device, data, gen, netDK, optimizerDK, gamma, lam)

	def get_gp(self, device, x, fake_x, nn, gamma = 750):
		batch = x.shape[0]
		alpha = torch.rand(batch, 1, 1, 1).to(device)

		x_hat = alpha * x.detach() + (1 - alpha) * fake_x.detach()
		x_hat.requires_grad_(True)

		pred_hat = nn(x_hat)
		gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat,
					grad_outputs=torch.ones(pred_hat.size()).to(device),
					create_graph=True, retain_graph=True, only_inputs=True)[0]

		grad_norm = gradients.view(batch, -1).norm(2, dim=1)
		return grad_norm.sub(gamma).pow(2).mean()/(gamma**2)
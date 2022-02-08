string = 'Hydrogen'
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
!ls

class MyData(Dataset):
	def __init__(self, df, datasetName="KagCerCanRisk", dset = "test", std=0.25, s=6):
		self.df = df
		self.path = os.path.join(PATH, datasetName, dset)
		self.t = transforms.Compose([
							   transforms.ToTensor(),
							   transforms.Resize(s, interpolation=PIL.Image.NEAREST),
							   AddGaussianNoise(0., std),
							   ])
		self.fileNames = os.listdir(self.path)
		self.fileNames.sort()

	def __len__(self):
		return self.df.shape[0]

	def __getitem__(self, index):
		file = self.path + "/" + self.df.Files[index]
		image = self.load_image(file)
		label = self.df.Indices[index].astype(np.compat.long) - 1
		return image, label

	def load_image(self, file_name):
		x = np.loadtxt(file_name).astype(np.float32)
		image = self.t(x)
		return image

class AddGaussianNoise(object):
	def __init__(self, mean=0., std=1.):
		self.std = std
		self.mean = mean

	def __call__(self, tensor):
		return tensor + torch.randn(tensor.size()) * self.std + self.mean

	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class generateDatasets(object):
	def __init__(self, datasetName = "KagCerCanRisk", batch_size=40, num_workers=8, std_tr=0.25, s=6):
		self.bsize = batch_size
		self.nworkers = num_workers
		self.df_test = pd.read_csv(os.path.join(PATH, datasetName) + "/test.csv")
		self.df_train = pd.read_csv(os.path.join(PATH, datasetName) + "/train.csv")
		self.test = MyData(self.df_test, datasetName=datasetName, dset="test", std=0.0, s=s)
		self.train = MyData(self.df_train, datasetName=datasetName, dset="train",std= std_tr, s=s)

	def outputDatasets(self, typeSet = "test"):
		if typeSet == "test":
			return self.test, self.df_test
		elif typeSet == "train":
			return self.train, self.df_train

	def getWeights(self):
		wTest = np.zeros(self.df_test.Label.unique().size)
		for i in range(self.df_test.Label.size):
			wTest[int(self.df_test.Label[i])-1] += 1
		wTrain = np.zeros(self.df_train.Label.unique().size)
		for i in range(self.df_train.Label.size):
			wTrain[int(self.df_train.Label[i])-1] += 1
		if np.prod(wTest == self.df_test.Label.size/len(wTest)):
			print("Labels are balanced in test set")
		if np.prod(wTrain == self.df_train.Label.size/len(wTrain)):
			print("Labels are balanced in train set")
		return wTest, wTrain

	def getDataLoaders(self):
		trainloader = torch.utils.data.DataLoader(self.train, batch_size=self.bsize,
					shuffle=True, num_workers=self.nworkers)
		testloader = torch.utils.data.DataLoader(self.test, batch_size=self.bsize,
					shuffle=True, num_workers=self.nworkers)
		return trainloader, testloader


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

	def genSeq(self, chnIn = 1, chnOut = 64, dpout = 0.1):
		loop = len(self.l) - 3
		l = [self.block(chnIn,chnOut,dpout,1)]
		for i in range(loop):
			l.append(self.block(chnOut*2**i,chnOut*2**(i+1),dpout,0))
		l.append(nn.Sequential(nn.Flatten(), nn.Linear(chnOut*2**(i+1), 1), nn.Sigmoid(),))
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
						nn.ReLU(),
						Reshape(256,2,2),

						nn.ConvTranspose2d(256,128,3,1,1),
						nn.Upsample(scale_factor=2),
						nn.BatchNorm2d(128),
						nn.LeakyReLU(negative_slope=0.02),

						nn.ConvTranspose2d(128,1,3,1,0),
						# nn.BatchNorm2d(1),
						# nn.Hardtanh(),
						nn.LeakyReLU(negative_slope=0.02),
						)

	def forward(self, x):
		x = self.seqIn(x)
		return x

class MLP(SimpleClas):
	pass
class Disc(SimpleDisc):
	pass
class Gen(SimpleGen):
	pass


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

class inOut(object):
	def __init__(self, datasetName = "KagCerCanRisk", tag = 'backup'):
		self.PATH = os.path.join(PATH, datasetName) + "/"
	def save_model(self, model, module, optimizer, loss, epoch, dir):
		# global PATH
		os.path.isdir(self.PATH + "Models/") or os.mkdir(self.PATH + "Models/")
		# date = datetime.now().__str__()
		# date = date[:16].replace(':', '-').replace(' ', '-')
		path = self.PATH + "Models/" + dir + "/"
		os.path.isdir(path) or os.mkdir(path)
		filename = os.path.join(path, f'{module}-backup.pt')
		torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': loss,
				}, filename)
		if len(dict[module]) == 0 or tag != 'backup':
			dict[module].append(filename)
		else:
			dict[module][-1] = filename
		# dict[module].append(filename)
		self.saveDict(dict)

	def load_model(self, module, dict):
		if module == "Class":
			model = MLP().to(device)
			checkpoint = torch.load(dict[module][-1])
			model.load_state_dict(checkpoint['model_state_dict'])
			# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			last_epoch = checkpoint['epoch']
			loss = checkpoint['loss']
			return last_epoch, loss, model
		elif module == "Gen":
			model = Gen().to(device)
			checkpoint = torch.load(dict[module][-1])
			model.load_state_dict(checkpoint['model_state_dict'])
			# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			last_epoch = checkpoint['epoch']
			loss = checkpoint['loss']
			return last_epoch, loss, model
		elif module == "Disc":
			model = Disc().to(device)
			checkpoint = torch.load(dict[module][-1])
			model.load_state_dict(checkpoint['model_state_dict'])
			# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			last_epoch = checkpoint['epoch']
			loss = checkpoint['loss']
			return last_epoch, loss, model
		elif module == "Enc":
			model = Enc().to(device)
			checkpoint = torch.load(dict[module][-1])
			model.load_state_dict(checkpoint['model_state_dict'])
			# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			last_epoch = checkpoint['epoch']
			loss = checkpoint['loss']
			return last_epoch, loss, model

	def newDict(self, dir = "0"):
		# os.path.isdir(PATH) or os.mkdir(PATH)
		os.path.isdir(self.PATH + "Dict/") or os.mkdir(self.PATH + "Dict/")
		path = self.PATH + "Dict/" + dir + "/"
		os.path.isdir(path) or os.mkdir(path)
		date = datetime.now().__str__()
		date = date[:16].replace(':', '-').replace(' ', '-')
		DICT_NAME = f'Dict-{date}.json'
		dict = {
			"Class" : [],
			"Gen" :	[],
			"Disc" : [],
			# "Enc" : [],
			"Path" : [path + DICT_NAME]
			}
		dictJSON = json.dumps(dict)
		f = open(dict["Path"][-1],"w")
		f.write(dictJSON)
		f.close()
		return dict

	def loadDict(self, dir):
		f = open(os.path.join(self.PATH, "Dict", dir),"r")
		dict = json.load(f)
		f.close()
		return dict

	def saveDict(self, dict):
		dictJSON = json.dumps(dict)
		f = open(dict["Path"][-1],"w")
		f.write(dictJSON)
		f.close()

class myPlots():
	def clasPlots(self, error_list, acc, accTrain, epoch):
		fig, (ax1, ax2) = plt.subplots(2, sharex=True)
		ax1.plot(error_list, 'b*-', lw=3, ms=12)
		ax1.set(ylabel='loss', title='Epoch {}'.format(epoch+1))
		ax2.plot(acc, 'r*-', lw=3, ms=12)
		ax2.plot(accTrain, 'g*-', lw=3, ms=12)
		ax2.set(xlabel='epochs', ylabel='%', title="Accuracy")
		plt.show()

	def plotGANs(self, error_list_D, error_list_G, testloader, gen, epoch):
		real_batch = next(iter(testloader))
		fake_batch = gen(torch.randn(25,100).to(device))

		fig, axs = plt.subplots(2, 2)
		axs[0, 0].plot(error_list_D, 'b*-', lw=3, ms=12)
		axs[0,0].set(ylabel='Disc Loss', title='Epoch {}'.format(epoch+1))
		im = axs[0, 1].imshow(vutils.make_grid(real_batch[0].to(device)[:25], padding=2, normalize=False, range=(-1,1),  nrow=5).cpu()[1,:,:],cmap='cividis', interpolation='nearest')
		axs[0, 1].set_title('Real')
		fig.colorbar(im, ax=axs[0, 1])

		frame1 = fig.gca()
		frame1.axes.get_xaxis().set_visible(False)
		frame1.axes.get_yaxis().set_visible(False)

		axs[1, 0].plot(error_list_G, 'r*-', lw=3, ms=12)
		axs[1,0].set(ylabel='Gen Loss', title='')

		im = axs[1, 1].imshow(vutils.make_grid(fake_batch.to(device), padding=2, normalize=False, range=(-1,1),  nrow=5).detach().cpu()[1,:,:],cmap='cividis', interpolation='nearest')
		fig.colorbar(im, ax=axs[1, 1])
		plt.show()

class train(object):
	def __init__(self, latentSpace=100, std_tr=0.25, s=128):
		self.real_label = 1.0
		self.fake_label = 0.0
		self.latentSpace = latentSpace
		self.std_tr = std_tr
		self.s = s
		self.lam = 0.5

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

	def trainClass(self, epochs=100, snap=25):
		clas = MLP().to(device)
		criterion = nn.NLLLoss()
		opt = optim.Adam(clas.parameters(), lr=lr)
		trainloader, testloader = generateDatasets(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s).getDataLoaders()
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
				# if i > 2:
				# 	break
			error_list.append(error/(i+1))
			acc.append(accuracy().validation(testloader, clas).item())
			accTrain.append(accuracy().validation(trainloader, clas).item())
			# acc.append(i)
			# fig, (ax1, ax2) = plt.subplots(2, sharex=True)
			# ax1.plot(error_list, 'b*-', lw=3, ms=12)
			# ax1.set(ylabel='loss', title='Epoch {}'.format(epoch+1))
			# ax2.plot(acc, 'r*-', lw=3, ms=12)
			# ax2.plot(accTrain, 'g*-', lw=3, ms=12)
			# ax2.set(xlabel='epochs', ylabel='%', title="Accuracy")
			# plt.show()
			myPlots().clasPlots(error_list, acc, accTrain, epoch)

			if epoch % snap == snap-1 :
				inOut().save_model(clas, 'Class', opt, error_list, epoch, dir)
		print("Done!!")
		return error_list, acc, accTrain

	def trainGANs(self, epochs=100, snap=25):
		disc = Disc(ngpu, flag, alpha, nc).to(device)
		gen = Gen(ngpu, flag, alpha, nc).to(device)
		criterion = nn.BCELoss()

		#OPT
		optD = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.99))
		optG = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.99))
		trainloader, testloader = generateDatasets(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s).getDataLoaders()
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
			myPlots().plotGANs(error_list_D, error_list_G, testloader, gen, epoch)

			if epoch % snap == snap-1 :
				inOut().save_model(gen, 'Gen', optG, error_list_G, epoch, dir)
				inOut().save_model(disc, 'Disc', optD, error_list_D, epoch, dir)
		print("Done!!")
		return error_list_D, error_list_G

	def trainWGANs(self, epochs=100, snap=25):
		disc = Disc().to(device)
		gen = Gen().to(device)
		wass = lambda x : x.mean()
		criterion = wass
		reg = self.covLoss
		#OPT
		optD = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.99))
		optG = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.99))
		trainloader, testloader = generateDatasets(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s).getDataLoaders()
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
			myPlots().plotGANs(error_list_D, error_list_G, testloader, gen, epoch)

			if epoch % snap == snap-1 :
				inOut().save_model(gen, 'Gen', optG, error_list_G, epoch, dir)
				inOut().save_model(disc, 'Disc', optD, error_list_D, epoch, dir)
		print("Done!!")
		return error_list_D, error_list_G

	def trainWGANs_GP(self, epochs=100, snap=25, lam = 1.0, gamma = 750):
		disc = Disc().to(device)
		gen = Gen().to(device)
		wass = lambda x : x.mean()
		criterion = wass
		#OPT
		optD = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.99))
		optG = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.99))
		trainloader, testloader = generateDatasets(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s).getDataLoaders()
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
			myPlots().plotGANs(error_list_D, error_list_G, testloader, gen, epoch)

			if epoch % snap == snap-1 :
				inOut().save_model(gen, 'Gen', optG, error_list_G, epoch, dir)
				inOut().save_model(disc, 'Disc', optD, error_list_D, epoch, dir)
		print("Done!!")
		return error_list_D, error_list_G

	def trainWGANs_GPandUnroll(self, epochs=100, snap=25, lam = 1.0, gamma = 750, K=1):
		disc = Disc().to(device)
		discK = Disc().to(device)
		gen = Gen().to(device)
		wass = lambda x : x.mean()
		self.criterion = wass
		#OPT
		optD = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.99))
		optDK = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.99))
		optG = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.99))
		trainloader, testloader = generateDatasets(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s).getDataLoaders()
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
			myPlots().plotGANs(error_list_D, error_list_G, testloader, gen, epoch)

			if epoch % snap == snap-1 :
				inOut().save_model(gen, 'Gen', optG, error_list_G, epoch, dir)
				inOut().save_model(disc, 'Disc', optD, error_list_D, epoch, dir)
		print("Done!!")
		return error_list_D, error_list_G

	def continuetrainWGANs(self, epochs=100, snap=25):
		# disc = Disc().to(device)
		# gen = Gen().to(device)
		lastEpoch, _, gen = inOut().load_model( "Gen", dict)
		_, _, disc = inOut().load_model( "Disc", dict)
		print(lastEpoch)
		wass = lambda x : x.mean()
		criterion = wass
		#OPT
		optD = optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.99))
		optG = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.99))
		trainloader, testloader = generateDatasets(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, std_tr=self.std_tr, s=self.s).getDataLoaders()
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
				errG = - criterion(yfake)
				errG.backward()
				optG.step()
				errorG += errG.item()
				# if i > 2:
				# 	break
			error_list_D.append(errorD/(i+1))
			error_list_G.append(errorG/(i+1))
			myPlots().plotGANs(error_list_D, error_list_G, testloader, gen, epoch)

			if epoch % snap == snap-1 :
				inOut().save_model(gen, 'Gen', optG, error_list_G, epoch, dir)
				inOut().save_model(disc, 'Disc', optD, error_list_D, epoch, dir)
		print("Done!!")
		return error_list_D, error_list_G

	def unroll(self, data, gen, netDK, optimizerDK, gamma, lam):
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
		err = - self.criterion(yreal) + self.criterion(yfake) + lam * self.get_gp(x, xfake, netDK, gamma)
		err.backward()
		optimizerDK.step()
		# errorD += err.item()

	def copy_unroll(self, data, gen, netDK, optimizerDK, netD, optimizerD, K=1, gamma = 750, lam = 10):
		optimizerDK.load_state_dict(optimizerD.state_dict())
		#     optimizerDK = copy.deepcopy(optimizerD)
		netDK.load_state_dict(netD.state_dict())
		for i in range(K):
			self.unroll(data, gen, netDK, optimizerDK, gamma, lam)

	def get_gp(self, x, fake_x, nn, gamma = 750):
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



###Start Here
PATH = "/scratch/st-mgorges-1/jtoledom/nobackup/SynData/"
datasetName = "KagCerCanRisk"

dir = str(9)
BATCH_SIZE=40
NUM_WORKERS=8
ngpu = 1
lr = 0.0001
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# dict = inOut().newDict(dir)
dictFile = os.listdir(os.path.join(PATH, datasetName, "Dict", dir))[0]
dictFile
dict = inOut().loadDict(os.path.join(PATH, datasetName, "Dict", dir, dictFile))
dict
#Load dataloaders
test, test_label = generateDatasets(s=6).outputDatasets("test")
# train, train_label = generateDatasets().outputDatasets("train")
trainloader, testloader = generateDatasets(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS).getDataLoaders()

#Instatiate network
# clas = MLP().to(device)


#TRAIN CLASS
# error_list, acc, accTrain = train().trainClass(100,25)


# error_list_D, error_list_G  = train(s=6).trainWGANs(100,25)


##########
genDat = generateDatasets(s=6)
# [genDat.test.__getitem__(i)[0][0][0][0].item() for i in range(50)]
ep,err, theModel = inOut().load_model( "Gen", dict)

synSamp = 5000
synDat = theModel(torch.rand(synSamp,100).to(device)).detach().cpu().numpy()

d = np.ceil(synDat.reshape(synSamp,36))
d[:,0].mean()
d = d.astype(int)

dataKag = os.path.join("/scratch/st-mgorges-1/jtoledom", "nobackup/SynData/kag_risk_factors_cervical_cancer.csv")
dataKag
df = pd.read_csv(dataKag)
df.iloc[:,0].mean()
'''
	Raw data contains missing values ('?'). We convert them into -1.
'''
for column in df.columns:
	df[column].loc[df[column] == '?'] = -1
s = df.to_numpy().astype(float)
s[0,:]
df.head()
plt.plot(s[:,3])
plt.hist(s[:,0])
plt.hist(d[:,0])
s.shape[1]
class compare(object):
	def __init__(self, real, synth):
		self.real = real
		self.synth = synth
		self.meanValues()
		self.stdValues()

	def meanValues(self):
		self.meanReal = []
		self.meanSynth = []
		for i in range(self.real.shape[1]):
			self.meanReal.append(np.mean(self.real[:,i]))
			self.meanSynth.append(np.mean(self.synth[:,i]))

	def stdValues(self):
		self.stdReal = []
		self.stdSynth = []
		for i in range(self.real.shape[1]):
			self.stdReal.append(np.std(self.real[:,i]))
			self.stdSynth.append(np.std(self.synth[:,i]))

	def plotMean(self, xmax=0, ymax=0):
		plt.plot(self.meanReal, self.meanSynth, 'b*-', lw=0.0, ms=12)
		plt.plot(range(-1,int(np.ceil(np.max([self.meanReal, self.meanSynth])))), range(-1,int(np.ceil(np.max([self.meanReal, self.meanSynth])))), c='r', lw=5)
		plt.ylabel('Predicted')#, title='Mean values')
		plt.xlabel('Real')#, title='Mean values')
		plt.title('Mean Values')
		if xmax != 0:
			plt.xlim(right=xmax)
		if ymax != 0:
			plt.ylim(top=ymax)

		plt.show()

	def plotStd(self, xmax=0, ymax=0):
		plt.plot(self.stdReal, self.stdSynth, 'b*-', lw=0.0, ms=12)
		plt.plot(range(-1,int(np.ceil(np.max([self.stdReal, self.stdSynth])))), range(-1,int(np.ceil(np.max([self.stdReal, self.stdSynth])))), c='r', lw=5)
		plt.ylabel('Predicted')#, title='Mean values')
		plt.xlabel('Real')#, title='Mean values')
		plt.title('Standard Deviation')
		if xmax != 0:
			plt.xlim(right=xmax)
		if ymax != 0:
			plt.ylim(top=ymax)

		plt.show()


comp = compare(s,d)
comp.plotMean()

comp.plotStd()

d.shape
dfSyn = pd.DataFrame(data=d, columns=df.columns)
dfSyn.to_csv("/scratch/st-mgorges-1/jtoledom/nobackup/SynData/synData-9.csv", index=False)

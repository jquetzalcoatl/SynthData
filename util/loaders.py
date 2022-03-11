import json
import logging
import os
from datetime import datetime

import PIL
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from NNets import select_nn




class MyData(Dataset):
    def __init__(self,  PATH, df, dset = "test", std=0.25, s=6):
        self.df = df
        self.path = os.path.join(PATH, dset)
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
    def __init__(self, PATH, batch_size=40, num_workers=8, std_tr=0.25, s=6):
        self.bsize = batch_size
        self.nworkers = num_workers
        self.df_test = pd.read_csv(os.path.join(PATH, 'test.csv'))
        self.df_train = pd.read_csv(os.path.join(PATH, 'train.csv'))
        self.test = MyData(PATH, self.df_test, dset="test", std=0.0, s=s)
        self.train = MyData(PATH, self.df_train, dset="train",std= std_tr, s=s)

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



class inOut(object):
    def save_model(self, model, module, optimizer, loss, epoch, dict, tag = 'backup'):
        # global PATH
        os.path.isdir(dict['PathRoot'][-1] + "Models/") or os.mkdir(dict['PathRoot'][-1] + "Models/")
        # date = datetime.now().__str__()
        # date = date[:16].replace(':', '-').replace(' ', '-')
        path = dict['PathRoot'][-1] + "Models/" + dict['Dir'] + "/"
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
        dict["Loss"+module] = loss
        self.saveDict(dict)

    def load_model(self, module, dict, device):
        if module == "Class":
            model = select_nn('Class')
            model = model().to(device)
        #             model = MLP().to(device)
            checkpoint = torch.load(dict[module][-1])
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            return last_epoch, loss, model
        elif module == "Gen":
            model = select_nn('Gen')
            model = model().to(device)
        #             model = Gen().to(device)
            checkpoint = torch.load(dict[module][-1])
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            return last_epoch, loss, model
        elif module == "Disc":
            model = select_nn('Disc')
            model = model().to(device)

        #             model = Disc().to(device)
            checkpoint = torch.load(dict[module][-1])
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            return last_epoch, loss, model
        elif module == "Enc":
            exit()
            model = select_nn('Disc')
            model = model().to(device)
        #             model = Enc().to(device)
            checkpoint = torch.load(dict[module][-1])
            model.load_state_dict(checkpoint['model_state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            last_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            return last_epoch, loss, model

    def newDict(self, PATH, dir = "0"):
        # os.path.isdir(PATH) or os.mkdir(PATH)
        os.path.isdir(PATH + "Dict/") or os.mkdir(PATH + "Dict/")
        path = PATH + "Dict/" + dir + "/"
        os.path.isdir(path) or os.mkdir(path)
        date = datetime.now().__str__()
        date = date[:16].replace(':', '-').replace(' ', '-')
        DICT_NAME = f'Dict-{date}.json'
        dict = {
            "Class" : [],
            "Gen" :	[],
            "Disc" : [],
            # "Enc" : [],
            "Path" : [path + DICT_NAME],
            "PathRoot" : [PATH],
            "Dir" : dir
            }
        dictJSON = json.dumps(dict)
        f = open(dict["Path"][-1],"w")
        f.write(dictJSON)
        f.close()
        return dict

    def loadDict(self, dir):
        f = open(dir,"r")
        dict = json.load(f)
        f.close()
        return dict

    def saveDict(self, dict):
        dict["LastUpdated"] = [self.timestamp()]
        dictJSON = json.dumps(dict)
        f = open(dict["Path"][-1],"w")
        f.write(dictJSON)
        f.close()
        
    def timestamp(self):
        date = datetime.now().__str__()
        date = date[:19].replace(' ', '-')
        return date

    def logFunc(self, dict, filename = "train.log"):
        self.initTime = datetime.now()
        os.path.isdir(dict['PathRoot'][-1] + "Logs/") or os.mkdir(dict['PathRoot'][-1] + "Logs/")
        os.path.isdir(dict['PathRoot'][-1] + "Logs/" + dict['Dir']) or os.mkdir(dict['PathRoot'][-1] + "Logs/" + dict['Dir'])
        path = dict['PathRoot'][-1] + "Logs/" + dict['Dir'] + "/"

        self.logging = logging
        self.logging = logging.getLogger()
        self.logging.setLevel(logging.DEBUG)
        self.handler = logging.FileHandler(os.path.join(path, filename))
        self.handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.handler.setFormatter(formatter)
        self.logging.addHandler(self.handler)

        #         self.logging = logging
        #         self.logging.basicConfig(filename=os.path.join(path, 'DiffSolver.log'), level=logging.DEBUG)
        self.logging.info(f'{str(self.initTime).split(".")[0]} - Log')

        dict["Log"] = os.path.join(path, filename)


#         self.logging.info(f'{str(self.initTime).split(".")[0]} - Bulk Export started')
#         self.logging.info(f'bulkImport - Data will be dumped in {self.pathToDump}')
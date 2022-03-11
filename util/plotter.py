import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from loaders import generateDatasets
# from tools import errInSample
import torch



class myPlots():
    def clasPlots(self, error_list, acc, accTrain, epoch):
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        ax1.plot(error_list, 'b*-', lw=3, ms=12)
        ax1.set(ylabel='loss', title='Epoch {}'.format(epoch+1))
        ax2.plot(acc, 'r*-', lw=3, ms=12)
        ax2.plot(accTrain, 'g*-', lw=3, ms=12)
        ax2.set(xlabel='epochs', ylabel='%', title="Accuracy")
        plt.show()

    def plotGANs(self, device, error_list_D, error_list_G, testloader, gen, epoch, dict, bash=False):
        real_batch = next(iter(testloader))
        fake_batch = gen(torch.randn(25,dict['LS']).to(device))

        fig, axs = plt.subplots(3, 2, figsize=(8.5, 8.5))
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
        
        values, base = np.histogram(real_batch[0][:25].view(-1).detach().cpu().numpy(), bins=100)
        valuesFake, baseFake = np.histogram(fake_batch.view(-1).detach().cpu().numpy(), bins=100)
        cumulative = np.cumsum(values)
        cumulativeFake = np.cumsum(valuesFake)
        axs[2, 0].plot(base[:-1], cumulative/np.cumsum(values)[-1], 'b*-',  c='blue', lw=3, ms=5)
        axs[2, 0].plot(baseFake[:-1], cumulativeFake/np.cumsum(valuesFake)[-1],'r*-',  c='red', lw=3, ms=5)
        axs[2, 0].set(xlabel="x", ylabel="Cumulative")
        
        im = axs[2, 1].imshow(vutils.make_grid(fake_batch.to(device) - real_batch[0].to(device)[:25], padding=2, normalize=False, range=(-1,1),  nrow=5).detach().cpu()[1,:,:],cmap='seismic', interpolation='nearest')
        fig.colorbar(im, ax=axs[2, 1])
        
        if bash == False:
            plt.show()
        # Adding this here for the time being
        os.path.isdir(dict['PathRoot'][-1] + "Plots/") or os.mkdir(dict['PathRoot'][-1] + "Plots/")
        path = dict['PathRoot'][-1] + "Plots/" + dict['Dir'] + "/"
        os.path.isdir(path) or os.mkdir(path)
        fig.savefig(path + f'Epoch_{epoch}.png', dpi=fig.dpi)

    def plotDiff(self, PATH, dir, device, error_list, error_list_test, testloader, diff, epoch,
                 transformation="linear", bash=False):
        (x, y) = next(iter(testloader))
        # y = next(iter(testloader))[1]
        yhat = diff(x.to(device))
        yhat, y = transformation_inverse(yhat, y, transformation)
        #if transformation == "sqrt":
        #    yhat = yhat.pow(2)
        #    y = y.pow(2)
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].plot(error_list, 'b*-', lw=3, ms=12)
        # axs[0,0].set(ylabel='Loss', title='Epoch {}'.format(epoch+1))
        # axs[0, 1].imshow(vutils.make_grid(yhat.to(device)[:25], padding=2, normalize=False, range=(-1,1),  nrow=5).cpu()[1,:,:],cmap='cividis', interpolation='nearest')
        # frame1 = fig.gca()
        # frame1.axes.get_xaxis().set_visible(False)
        # frame1.axes.get_yaxis().set_visible(False)
        #
        # axs[1, 0].plot(error_list, 'r*-', lw=3, ms=12)
        # axs[1,0].set(ylabel='Loss', title='')
        #
        # axs[1, 1].imshow(vutils.make_grid(y.to(device), padding=2, normalize=False, range=(-1,1),  nrow=5).detach().cpu()[1,:,:],cmap='cividis', interpolation='nearest')
        # plt.show()
        # plt.imshow(vutils.make_grid((theModel(r[0].to(device))[1:2])/1.85 - r[1].to(device)[1:2], padding=2, normalize=False, range=(-1,1),  nrow=5).detach().cpu().numpy()[1,:,:],cmap='cividis', interpolation='nearest')
        # plt.colorbar()
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8.5, 8.5))
        axs[0, 0].plot(error_list, 'b*-', lw=3, ms=12)
        axs[0, 0].plot(error_list_test, 'r*-', lw=3, ms=12)
        axs[0, 0].set(ylabel='Loss', title='Epoch {}'.format(epoch + 1))

        im = axs[0, 1].imshow(vutils.make_grid(
            yhat.to(device)[:25], padding=2, normalize=False, range=(-1, 1), nrow=5).detach().cpu().numpy()[1, :, :],
                              cmap='cividis', interpolation='nearest')
        axs[0, 1].set_title('Prediction')
        fig.colorbar(im, ax=axs[0, 1])
        # plt.colorbar()

        frame1 = fig.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)

        # axs[1, 0].plot(error_list, 'r*-', lw=3, ms=12)
        # # axs[1,0].colorbar()
        # axs[1,0].set(ylabel='Loss', title='')

        im = axs[1, 1].imshow(vutils.make_grid(y.to(device)[:25], padding=2, normalize=False, range=(-1, 1),
                                               nrow=5).detach().cpu().numpy()[1, :, :], cmap='cividis',
                              interpolation='nearest')
        fig.colorbar(im, ax=axs[1, 1])

        im = axs[1, 0].imshow(vutils.make_grid(x.to(device)[:25], padding=2, normalize=False, range=(-1, 1),
                                               nrow=5).detach().cpu().numpy()[1, :, :], cmap='cividis',
                              interpolation='nearest')
        fig.colorbar(im, ax=axs[1, 0])

        im = axs[2, 0].imshow(
            vutils.make_grid(y.to(device)[:25] - yhat.to(device)[:25], padding=2, normalize=False, range=(-1, 1),
                             nrow=5).detach().cpu().numpy()[1, :, :], cmap='seismic', interpolation='nearest')
        fig.colorbar(im, ax=axs[2, 0])

        im = axs[2, 1].imshow(
            vutils.make_grid(y.to(device)[:25] - (yhat.to(device) / yhat.to(device).max())[:25], padding=2,
                             normalize=False, range=(-1, 1), nrow=5).detach().cpu().numpy()[1, :, :], cmap='seismic',
            interpolation='nearest')
        fig.colorbar(im, ax=axs[2, 1])
        if bash == False:
            plt.show()
        # Adding this here for the time being
        os.path.isdir(PATH + "Plots/") or os.mkdir(PATH + "Plots/")
        path = PATH + "Plots/" + dir + "/"
        os.path.isdir(path) or os.mkdir(path)
        fig.savefig(path + f'Epoch_{epoch}.png', dpi=fig.dpi)


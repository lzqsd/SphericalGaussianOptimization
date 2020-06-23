import torch
import numpy as np
import cv2
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class SGEnvOptim():
    def __init__(self, isCuda = True, gpuId = 0, niter = 10, envNum = 19200,
            envWidth = 32, envHeight = 16, SGRow = 2, SGCol = 6, ch = 3):
        self.SGNum = int(SGRow*SGCol )

        self.envNum = envNum
        self.niter = niter
        self.ch = ch

        phiCenter = ( (np.arange(SGCol ) + 0.5) / SGCol - 0.5 ) * np.pi * 2
        thetaCenter = (np.arange(SGRow ) + 0.5) / SGRow * np.pi / 2.0
        self.phiCenter, self.thetaCenter = np.meshgrid(phiCenter, thetaCenter )
        self.thetaCenter = self.thetaCenter.reshape(1, self.SGNum, 1, 1, 1, 1).astype(np.float32 )
        self.phiCenter = self.phiCenter.reshape(1, self.SGNum, 1, 1, 1, 1).astype(np.float32 )
        self.thetaRange = (np.pi / 2 / SGRow) * 1.5
        self.phiRange = (2 * np.pi / SGCol ) * 1.5

        Az = ( (np.arange(envWidth) + 0.5) / envWidth - 0.5 ) * 2 * np.pi
        El = ( (np.arange(envHeight) + 0.5) / envHeight ) * np.pi / 2.0
        Az, El = np.meshgrid(Az, El )
        Az = Az[:, :, np.newaxis ]
        El = El[:, :, np.newaxis ]
        lx = np.sin(El ) * np.cos(Az )
        ly = np.sin(El ) * np.sin(Az )
        lz = np.cos(El )
        self.ls = np.concatenate( (lx, ly, lz), axis = 2)[np.newaxis, np.newaxis, np.newaxis, :]
        self.ls = Variable(torch.from_numpy(self.ls.astype(np.float32) ) )
        self.envHeight = envHeight
        self.envWidth = envWidth
        self.iterCount = 0

        self.W = Variable(torch.from_numpy(np.sin(El.astype(np.float32) ).reshape( (1, 1, envHeight, envWidth) ) ) )

        self.isCuda = isCuda
        self.gpuId = gpuId
        self.mask = Variable(torch.ones( (self.envNum, 1, 1, 1), dtype=torch.float32) )

        weight = Variable(torch.zeros( (self.envNum, self.SGNum, self.ch), dtype = torch.float32) )
        theta = Variable(torch.zeros( (self.envNum, self.SGNum, 1), dtype = torch.float32 ) )
        phi = Variable(torch.zeros( (self.envNum, self.SGNum, 1 ), dtype = torch.float32 ) )
        lamb = torch.log(Variable(torch.ones(self.envNum, self.SGNum, 1) * np.pi / SGRow ) )

        self.param = torch.cat([weight, theta, phi, lamb], dim = 2)
        self.param = self.param.view(self.envNum, self.SGNum * 6)

        self.thetaCenter = Variable(torch.from_numpy(self.thetaCenter ) )
        self.phiCenter = Variable(torch.from_numpy(self.phiCenter ) )
        self.envmap = Variable(torch.FloatTensor(self.envNum, self.ch, self.envHeight, self.envWidth) )

        self.thetaCenter = self.thetaCenter.expand([self.envNum, self.SGNum, 1, 1, 1, 1] )
        self.phiCenter = self.phiCenter.expand([self.envNum, self.SGNum, 1, 1, 1, 1] )

        if isCuda:
            self.mask = self.mask.cuda(self.gpuId )
            self.ls = self.ls.cuda(self.gpuId )

            self.param = self.param.cuda(self.gpuId )

            self.envmap = self.envmap.cuda(self.gpuId )
            self.thetaCenter = self.thetaCenter.cuda(self.gpuId )
            self.phiCenter = self.phiCenter.cuda(self.gpuId )
            self.W = self.W.cuda(self.gpuId )

        self.param.requires_grad = True
        self.thetaCenter.requires_grad = False
        self.phiCenter.requires_grad = False
        self.mask.requires_grad == False

        self.mseLoss = nn.MSELoss(size_average = False )
        self.optEnv = optim.LBFGS([self.param], lr=0.2, max_iter=100 )

    def renderSG(self, theta, phi, lamb, weight ):
        axisX = torch.sin(theta ) * torch.cos(phi )
        axisY = torch.sin(theta ) * torch.sin(phi )
        axisZ = torch.cos(theta )

        axis = torch.cat([axisX, axisY, axisZ], dim=5)

        mi = lamb.expand([self.envNum, self.SGNum, 1, self.envHeight, self.envWidth] ) * \
                (torch.sum(
                    axis.expand([self.envNum, self.SGNum, 1, self.envHeight, self.envWidth, 3] ) * \
                            self.ls.expand([self.envNum, self.SGNum, 1, self.envHeight, self.envWidth, 3] ),
                            dim = 5) -1 )
        envmaps = weight.expand([self.envNum, self.SGNum, self.ch, self.envHeight, self.envWidth] ) * \
                torch.exp(mi ).expand([self.envNum, self.SGNum, self.ch, self.envHeight, self.envWidth] )
        envmap = torch.sum(envmaps, dim=1 )

        return envmap

    def deparameterize(self ):
        weight, theta, phi, lamb = torch.split(self.param.view(self.envNum, self.SGNum, 6),
                [3, 1, 1, 1], dim=2)
        weight = weight.unsqueeze(-1).unsqueeze(-1)
        theta = theta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        phi = phi.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lamb = lamb.unsqueeze(-1).unsqueeze(-1)
        return theta, phi, weight, lamb


    def optimize(self, envmap, mask = None ):
        assert(envmap.shape[0] == self.envNum
                and envmap.shape[1] == self.ch
                and envmap.shape[2] == self.envHeight
                and envmap.shape[3] == self.envWidth)
        self.envmap.data.copy_(envmap  )
        if not mask is None:
            self.mask.data.copy_(mask )


        minLoss = 2e20
        recImageBest = None
        thetaBest = None
        phiBest = None
        weightBest = None
        lambBest = None
        self.loss = None

        for i in range(0, self.niter ):
            print('Iteration %d' % i )
            def closure( ):
                theta, phi, weight, lamb = self.deparameterize()
                theta = self.thetaRange * torch.tanh(theta ) + self.thetaCenter
                phi = self.phiRange * torch.tanh(phi ) + self.phiCenter
                weight = torch.exp(weight )
                lamb = torch.exp(lamb )

                recImage = self.renderSG(theta, phi, lamb, weight )
                loss = self.mseLoss(
                        torch.log(recImage * self.W.expand_as(recImage) + 1) \
                                * self.mask.expand_as(recImage ),
                        torch.log(self.envmap * self.W.expand_as(recImage) + 1) \
                                * self.mask.expand_as(recImage ) )

                self.loss = loss

                self.optEnv.zero_grad()
                loss.backward(retain_graph = True)

                if self.iterCount % 20 == 0:
                    print ('%d Loss: %f' % (self.iterCount, (loss.item() / self.envNum
                        / self.envWidth / self.envHeight / self.ch ) ) )
                self.iterCount += 1
                return loss

            self.optEnv.step(closure)


            if self.loss.cpu().data.item() < minLoss:
                if torch.isnan(torch.sum(self.param ) ):
                    break
                else:
                    theta, phi, weight, lamb = self.deparameterize()
                    theta = self.thetaRange * torch.tanh(theta ) + self.thetaCenter
                    phi = self.phiRange * torch.tanh(phi ) + self.phiCenter
                    weight = torch.exp(weight )
                    lamb = torch.exp(lamb )

                    recImage = self.renderSG(theta, phi, lamb, weight)

                    recImageBest = recImage.cpu().data.numpy()

                    thetaBest = theta.data.cpu().numpy().reshape( (self.envNum, self.SGNum, 1) )
                    phiBest = phi.data.cpu().numpy().squeeze().reshape( (self.envNum, self.SGNum, 1) )
                    lambBest = lamb.data.cpu().numpy().squeeze().reshape( (self.envNum, self.SGNum, 1) )
                    weightBest = weight.data.cpu().numpy().squeeze().reshape( (self.envNum, self.SGNum, 3) )

                    minLoss = self.loss.cpu()

                    del theta, phi, weight, lamb, recImage
            else:
                break


        return thetaBest, phiBest, lambBest, weightBest, recImageBest

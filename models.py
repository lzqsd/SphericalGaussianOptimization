import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def LSregress(pred, gt, origin):

    nb = pred.size(0)
    origSize = pred.size()
    pred = pred.view(nb, -1)
    gt = gt.view(nb, -1)

    coef = (torch.sum(pred * gt, dim = 1) / torch.clamp(torch.sum(pred * pred, dim=1), min=1e-5) ).detach()
    coef = torch.clamp(coef, 0.001, 1000)
    for n in range(0, len(origSize) -1 ):
        coef = coef.unsqueeze(-1)
    pred = pred.view(origSize )

    predNew = origin * coef.expand(origSize )

    return predNew

def LSregressDiffSpec(diff, spec, im, diffOrig, specOrig):
    nb, nc, nh, nw = diff.size()

    diff = diff.view(nb, -1)
    spec = spec.view(nb, -1)
    im = im.view(nb, -1)

    a11 = torch.sum(diff * diff, dim=1)
    a22 = torch.sum(spec * spec, dim=1)
    a12 = torch.sum(diff * spec, dim=1)

    frac = a11 * a22 - a12 * a12
    b1 = torch.sum(diff * im, dim = 1)
    b2 = torch.sum(spec * im, dim = 1)

    # Compute the coefficients based on linear regression
    coef1 = b1 * a22  - b2 * a12
    coef2 = -b1 * a12 + a11 * b2
    coef1 = coef1 / torch.clamp(frac, min=1e-3 )
    coef2 = coef2 / torch.clamp(frac, min=1e-3 )

    # Compute the coefficients assuming diffuse albedo only
    coef3 = torch.clamp(b1 / a11, 0.01, 100 )
    coef4 = torch.zeros(coef3.size(), dtype = torch.float32 )

    frac = (frac / (nc * ng * nw) ).detach()
    fracInd = (frac > 1e-7).float()

    coefDiffuse = fracInd * coef1 + (1 - fracInd) * coef3
    coefSpecular = fracInd * coef2 + (1 - fracInd) * coef4

    for n in range(0, 3):
        coefDiffuse = coefDiffuse.unsqueeze(-1)
        coefSpecular = coefSpecular.unsqueeze(-1)

    diffOrig = coefDiffuse.expand_as(diffOrig ) * diffOrig
    specOrig = coefDiffuse.expand_as(specOrig ) * specOrig

    return diffOrig, specOrig


class encoder0(nn.Module ):
    def __init__(self, cascadeLevel = 0, isSeg = False):
        super(encoder0, self).__init__()
        self.isSeg = isSeg
        self.cascadeLevel = cascadeLevel

        self.pad1 = nn.ReplicationPad2d(1)
        if self.isSeg == True:
            self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 4, stride =2 , bias = True )
        else:
            if self.cascadeLevel == 0:
                self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=4, stride=2, bias =True)
            else:
                self.conv1 = nn.Conv2d(in_channels = 22, out_channels = 64, kernel_size =4, stride =2, bias = True )

        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=64)

        self.pad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, bias=True)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=128)

        self.pad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn3 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad4 = nn.ZeroPad2d(1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn4 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad5 = nn.ZeroPad2d(1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=True)
        self.gn5 = nn.GroupNorm(num_groups=32, num_channels=512)

        self.pad6 = nn.ZeroPad2d(1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, bias=True)
        self.gn6 = nn.GroupNorm(num_groups=64, num_channels=1024)

    def forward(self, x):
        x1 = F.relu(self.gn1(self.conv1(self.pad1(x) ) ), True)
        x2 = F.relu(self.gn2(self.conv2(self.pad2(x1) ) ), True)
        x3 = F.relu(self.gn3(self.conv3(self.pad3(x2) ) ), True)
        x4 = F.relu(self.gn4(self.conv4(self.pad4(x3) ) ), True)
        x5 = F.relu(self.gn5(self.conv5(self.pad5(x4) ) ), True)
        x6 = F.relu(self.gn6(self.conv6(self.pad6(x5) ) ), True)

        return x1, x2, x3, x4, x5, x6


class decoder0(nn.Module ):
    def __init__(self, mode=0):
        super(decoder0, self).__init__()
        self.mode = mode

        self.dconv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding = 1, bias=True)
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.dconv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size = 4, stride=2, padding = 1, bias=True)
        self.dgn2 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.dconv3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size = 4, stride=2, padding=1, bias=True)
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.dconv4 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size = 4, stride=2, padding = 1, bias=True)
        self.dgn4 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.dconv5 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size = 4, stride=2, padding = 1, bias=True)
        self.dgn5 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.dconv6 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size = 4, stride=2, padding = 1, bias=True)
        self.dgn6 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.dpadFinal = nn.ReplicationPad2d(1)
        self.dconvFinal = nn.Conv2d(in_channels=64, out_channels=3, kernel_size = 3, stride=1, bias=True)

    def computePadding(self, os, ns):
        assert(os <= ns )
        gap = ns - os
        if gap % 2 == 0:
            return [int(gap/2), int(gap / 2) ]
        else:
            return [int((gap+1) / 2), int((gap-1) / 2) ]


    def forward(self, im, x1, x2, x3, x4, x5, x6 ):
        dx1 = F.relu(self.dgn1(self.dconv1(x6 ) ) )

        _, _, oh, ow = dx1.size()
        _, _, nh, nw = x5.size()
        vPadding = self.computePadding(oh, nh)
        hPadding = self.computePadding(ow, nw)
        padding = hPadding + vPadding
        dx1 = F.pad(dx1, padding, mode="replicate" )

        xin1 = torch.cat([dx1, x5], dim = 1)
        dx2 = F.relu(self.dgn2(self.dconv2(xin1 ) ), True)

        _, _, oh, ow = dx2.size()
        _, _, nh, nw = x4.size()
        vPadding = self.computePadding(oh, nh)
        hPadding = self.computePadding(ow, nw)
        padding = hPadding + vPadding
        dx2 = F.pad(dx2, padding, mode="replicate" )

        xin2 = torch.cat([dx2, x4], dim=1 )
        dx3 = F.relu(self.dgn3(self.dconv3(xin2) ), True)

        _, _, oh, ow = dx3.size()
        _, _, nh, nw = x3.size()
        vPadding = self.computePadding(oh, nh)
        hPadding = self.computePadding(ow, nw)
        padding = hPadding + vPadding
        dx3 = F.pad(dx3, padding, mode="replicate" )

        xin3 = torch.cat([dx3, x3], dim=1)
        dx4 = F.relu(self.dgn4(self.dconv4(xin3) ), True)

        _, _, oh, ow = dx4.size()
        _, _, nh, nw = x2.size()
        vPadding = self.computePadding(oh, nh)
        hPadding = self.computePadding(ow, nw)
        padding = hPadding + vPadding
        dx4 = F.pad(dx4, padding, mode="replicate" )

        xin4 = torch.cat([dx4, x2], dim=1 )
        dx5 = F.relu(self.dgn5(self.dconv5(xin4) ) )

        _, _, oh, ow = dx5.size()
        _, _, nh, nw = x1.size()
        vPadding = self.computePadding(oh, nh)
        hPadding = self.computePadding(ow, nw)
        padding = hPadding + vPadding
        dx5 = F.pad(dx5, padding, mode="replicate" )

        xin5 = torch.cat([dx5, x1], dim=1 )
        dx6 = F.relu(self.dgn6(self.dconv6(xin5) ), True)

        _, _, oh, ow = dx6.size()
        _, _, nh, nw = im.size()
        vPadding = self.computePadding(oh, nh)
        hPadding = self.computePadding(ow, nw)
        padding = hPadding + vPadding
        dx6 = F.pad(dx6, padding, mode="replicate" )

        x_orig = self.dconvFinal(self.dpadFinal(dx6) )

        if self.mode == 0:
            x_out = torch.clamp(1.01* (torch.tanh(x_orig ) ), -1, 1)
        elif self.mode == 1:
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
            norm = torch.sqrt(torch.sum(x_orig * x_orig, dim=1).unsqueeze(1) ).expand_as(x_orig);
            x_out = x_orig / norm
        elif self.mode == 2:
            x_orig = torch.clamp(1.01 * torch.tanh(x_orig ), -1, 1)
            x_out = torch.mean(x_orig, dim=1).unsqueeze(1)
        elif self.mode == 3:
            x_out = F.softmax(x_orig, dim=1)
        return x_out


class encoderLight(nn.Module ):
    def __init__(self, cascadeLevel = 0, SGNum = 9 ):
        super(encoderLight, self).__init__()

        self.cascadeLevel = cascadeLevel
        self.SGNum = SGNum

        self.preProcess = nn.Sequential(
                nn.ReplicationPad2d(1),
                nn.Conv2d(in_channels=13, out_channels=32, kernel_size=4, stride=2, bias =True),
                nn.GroupNorm(num_groups=2, num_channels=32),
                nn.ReLU(inplace = True ),

                nn.ZeroPad2d(1),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=True),
                nn.GroupNorm(num_groups=4, num_channels=64),
                nn.ReLU(inplace = True )
                )

        self.pad1 = nn.ReplicationPad2d(1)
        if self.cascadeLevel == 0:
            self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2, bias = True)
        else:
            self.conv1 = nn.Conv2d(in_channels=64 + SGNum * 6, out_channels=128, kernel_size=4, stride=2, bias =True)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.pad2 = nn.ZeroPad2d(1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.pad3 = nn.ZeroPad2d(1)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn3 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad4 = nn.ZeroPad2d(1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=True)
        self.gn4 = nn.GroupNorm(num_groups=32, num_channels=512)

        self.pad5 = nn.ZeroPad2d(1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, bias=True)
        self.gn5 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.pad6 = nn.ZeroPad2d(1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, bias=True)
        self.gn6 = nn.GroupNorm(num_groups=64, num_channels=1024)

    def forward(self, inputBatch, envs = None):

        input1 = self.preProcess(inputBatch )
        input2 = envs

        if self.cascadeLevel == 0:
            x = input1
        else:
            x = torch.cat([input1, input2], dim=1)

        x1 = F.relu(self.gn1(self.conv1(self.pad1(x) ) ), True)
        x2 = F.relu(self.gn2(self.conv2(self.pad2(x1) ) ), True)
        x3 = F.relu(self.gn3(self.conv3(self.pad3(x2) ) ), True)
        x4 = F.relu(self.gn4(self.conv4(self.pad4(x3) ) ), True)
        x5 = F.relu(self.gn5(self.conv5(self.pad5(x4) ) ), True)
        x6 = F.relu(self.gn6(self.conv6(self.pad6(x5) ) ), True)

        return x1, x2, x3, x4, x5, x6



class decoderLight(nn.Module ):
    def __init__(self, SGNum = 9 ):
        super(decoderLight, self).__init__()

        self.SGNum = SGNum

        self.dpad1 = nn.ZeroPad2d(1)
        self.dconv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, bias=True)
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.dconv2 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size = 4, stride=2, padding = 1, bias=True)
        self.dgn2 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.dconv3 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size = 4, stride=2, padding=1, bias=True)
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.dconv4 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size = 4, stride=2, padding = 1, bias=True)
        self.dgn4 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.dconv5 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size = 4, stride=2, padding = 1, bias=True)
        self.dgn5 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.dconv6 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size = 4, stride=2, padding = 1, bias=True)
        self.dgn6 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.dpadFinal = nn.ReplicationPad2d(1)
        self.dconvFinal = nn.Conv2d(in_channels=128, out_channels= SGNum * 6, kernel_size = 3, stride=1, bias=True)


    def computePadding(self, os, ns):
        assert(os <= ns )
        gap = ns - os
        if gap % 2 == 0:
            return [int(gap/2), int(gap / 2) ]
        else:
            return [int((gap+1) / 2), int((gap-1) / 2) ]

    def forward(self, x1, x2, x3, x4, x5, x6, env = None):
        dx1 = F.relu(self.dgn1(self.dconv1(self.dpad1(x6) ) ) )

        _, _, oh, ow = dx1.size()
        _, _, nh, nw = x5.size()
        vPadding = self.computePadding(oh, nh)
        hPadding = self.computePadding(ow, nw)
        padding = hPadding + vPadding
        dx1 = F.pad(dx1, padding, mode="replicate" )

        xin1 = torch.cat([dx1, x5], dim = 1)
        dx2 = F.relu(self.dgn2(self.dconv2(xin1 ) ), True)

        _, _, oh, ow = dx2.size()
        _, _, nh, nw = x4.size()
        vPadding = self.computePadding(oh, nh)
        hPadding = self.computePadding(ow, nw)
        padding = hPadding + vPadding
        dx2 = F.pad(dx2, padding, mode="replicate" )

        xin2 = torch.cat([dx2, x4], dim=1 )
        dx3 = F.relu(self.dgn3(self.dconv3(xin2) ), True)

        _, _, oh, ow = dx3.size()
        _, _, nh, nw = x3.size()
        vPadding = self.computePadding(oh, nh)
        hPadding = self.computePadding(ow, nw)
        padding = hPadding + vPadding
        dx3 = F.pad(dx3, padding, mode="replicate" )

        xin3 = torch.cat([dx3, x3], dim=1)
        dx4 = F.relu(self.dgn4(self.dconv4(xin3) ), True)

        _, _, oh, ow = dx4.size()
        _, _, nh, nw = x2.size()
        vPadding = self.computePadding(oh, nh)
        hPadding = self.computePadding(ow, nw)
        padding = hPadding + vPadding
        dx4 = F.pad(dx4, padding, mode="replicate" )

        xin4 = torch.cat([dx4, x2], dim=1 )
        dx5 = F.relu(self.dgn5(self.dconv5(xin4) ) )

        _, _, oh, ow = dx5.size()
        _, _, nh, nw = x1.size()
        vPadding = self.computePadding(oh, nh)
        hPadding = self.computePadding(ow, nw)
        padding = hPadding + vPadding
        dx5 = F.pad(dx5, padding, mode="replicate" )

        xin5 = torch.cat( [dx5, x1], dim=1 )
        dx6 = F.relu(self.dgn6(self.dconv6(xin5) ), True)

        if not env is None:
            _, _, oh, ow = dx6.size()
            _, _, nh, nw, _, _ = env.size()
            vPadding = self.computePadding(oh, nh)
            hPadding = self.computePadding(ow, nw)
            padding = hPadding + vPadding
            dx6 = F.pad(dx6, padding, mode="replicate" )

        x_out = 1.01 * torch.tanh(self.dconvFinal(self.dpadFinal(dx6) ) )
        x_out = 0.5 * (x_out + 1)
        x_out = torch.clamp(x_out, 0, 1)

        return x_out

class output2env():
    def __init__(self, envWidth = 32, envHeight = 16, isCuda = True, gpuId = 0, SGNum = 9):
        self.envWidth = envWidth
        self.envHeight = envHeight

        Az = ( (np.arange(envWidth) + 0.5) / envWidth - 0.5 )* 2 * np.pi
        El = ( (np.arange(envHeight) + 0.5) / envHeight) * np.pi / 2.0
        Az, El = np.meshgrid(Az, El)
        Az = Az[np.newaxis, :, :]
        El = El[np.newaxis, :, :]
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis = 0)
        ls = ls[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :, :]

        '''
        self.SGNum = int(SGRow * SGCol)
        phiCenter, thetaCenter = np.meshgrid(
                np.linspace( 2 * np.pi / SGCol / 2.0, 2 * np.pi * (1 - 1 / SGCol / 2.0), SGCol),
                np.linspace( 0.5 * np.pi / SGRow / 2.0, 0.5 * np.pi * (1 - 1 / SGRow / 2.0), SGRow) )
        thetaCenter = thetaCenter.reshape( (1, self.SGNum, 1, 1, 1) )
        phiCenter = phiCenter.reshape( (1, self.SGNum, 1, 1, 1) ) - np.pi
        self.thetaCenter = Variable(torch.from_numpy(thetaCenter.astype(np.float32 ) ) )
        self.phiCenter = Variable(torch.from_numpy(phiCenter.astype(np.float32 ) ) )
        self.phiRange = 2 * np.pi / SGCol / 1.6
        self.thetaRange = 0.5 * np.pi / SGRow / 1.6
        '''
        self.SGNum = SGNum

        self.ls = Variable(torch.from_numpy(ls.astype(np.float32 ) ) )
        if isCuda:
            self.ls = self.ls.cuda(gpuId )
            '''
            self.thetaCenter = self.thetaCenter.cuda(gpuId )
            self.phiCenter = self.phiCenter.cuda(gpuId )
            '''

        self.ls.requires_grad = False
        self.gpuId = gpuId

    def output2env(self, output ):
        weightOrig, axisOrig, lambOrig = torch.split(output,  \
                [self.SGNum * 3, self.SGNum * 2, self.SGNum], dim=1 )
        bn, _, envRow, envCol = weightOrig.size()

        weight = 0.9 * weightOrig.view(bn, self.SGNum, 3, envRow, envCol )
        weight = torch.tan(np.pi / 2 * weight )
        weight = weight.unsqueeze(-1).unsqueeze(-1)

        axisOrig = axisOrig.view(bn, self.SGNum, 2, envRow, envCol )
        thetaOrig, phiOrig = axisOrig.split(1, dim = 2)
        theta = thetaOrig * np.pi / 2.0
        phi = (2 * phiOrig - 1) * np.pi
        '''
        theta = (2 * thetaOrig-1) * self.thetaRange + self.thetaCenter.expand_as(thetaOrig )
        phi =  (2 * phiOrig -1) * self.phiRange + self.phiCenter.expand_as(phiOrig )
        '''
        axis_x = torch.sin(theta) * torch.cos(phi)
        axis_y = torch.sin(theta) * torch.sin(phi)
        axis_z = torch.cos(theta)
        axis = torch.cat([axis_x, axis_y, axis_z], dim=2 )
        axis = axis.unsqueeze(-1).unsqueeze(-1)

        lambOrig = torch.clamp(lambOrig.view(bn, self.SGNum, 1, envRow, envCol ), 0, 0.99)
        lamb = torch.tan(np.pi / 2 * lambOrig )
        lamb = lamb.unsqueeze(-1).unsqueeze(-1)

        mi = lamb.expand([bn, self.SGNum, 1, envRow, envCol, self.envHeight, self.envWidth] )* \
                (torch.sum(axis.expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth]) * \
                self.ls.expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth] ), dim = 2).unsqueeze(2) - 1)
        envmaps = weight.expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth] ) * \
            torch.exp(mi).expand([bn, self.SGNum, 3, envRow, envCol, self.envHeight, self.envWidth] )

        envmaps = torch.sum(envmaps, dim=1)

        return envmaps


class renderingLayer():
    def __init__(self, imWidth = 640, imHeight = 480, fov=63.4149, F0=0.05, cameraPos = [0, 0, 0],
            lightPower=3.0, gpuId = 0, envWidth = 32, envHeight = 16, isCuda = True):
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.envWidth = envWidth
        self.envHeight = envHeight

        self.fov = fov/180.0 * np.pi
        self.F0 = F0
        self.cameraPos = np.array(cameraPos, dtype=np.float32).reshape([1, 3, 1, 1])
        self.lightPower = lightPower
        self.xRange = 1 * np.tan(self.fov/2)
        self.yRange = float(imHeight) / float(imWidth) * self.xRange
        self.isCuda = isCuda
        x, y = np.meshgrid(np.linspace(-self.xRange, self.xRange, imWidth),
                np.linspace(-self.yRange, self.yRange, imHeight ) )
        y = np.flip(y, axis=0)
        z = -np.ones( (imHeight, imWidth), dtype=np.float32)

        pCoord = np.stack([x, y, z]).astype(np.float32)
        self.pCoord = pCoord[np.newaxis, :, :, :]
        v = self.cameraPos - self.pCoord
        v = v / np.sqrt(np.maximum(np.sum(v*v, axis=1), 1e-12)[:, np.newaxis, :, :] )
        v = v.astype(dtype = np.float32)

        self.v = Variable(torch.from_numpy(v) )
        self.pCoord = Variable(torch.from_numpy(self.pCoord) )

        self.up = torch.Tensor([0,1,0] )

        Az = ( (np.arange(envWidth) + 0.5) / envWidth - 0.5 )* 2 * np.pi
        El = ( (np.arange(envHeight) + 0.5) / envHeight) * np.pi / 2.0
        Az, El = np.meshgrid(Az, El)
        Az = Az.reshape(-1, 1)
        El = El.reshape(-1, 1)
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis = 1)

        envWeight = np.sin(El ) * np.pi * np.pi / envWidth / envHeight

        self.ls = Variable(torch.from_numpy(ls.astype(np.float32 ) ) )
        self.envWeight = Variable(torch.from_numpy(envWeight.astype(np.float32 ) ) )
        self.envWeight = self.envWeight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        if isCuda:
            self.v = self.v.cuda(gpuId)
            self.pCoord = self.pCoord.cuda(gpuId )
            self.up = self.up.cuda(gpuId )
            self.ls = self.ls.cuda(gpuId )
            self.envWeight = self.envWeight.cuda(gpuId )

        self.gpuId = gpuId


    def forward(self, diffusePred, normalPred, roughPred, distPred, segBatch, lightPos):

        lightPos = np.array(lightPos, dtype=np.float32).reshape([1, 3, 1, 1])
        lightPos = Variable(torch.from_numpy(lightPos) )
        temp = Variable(torch.FloatTensor(1, 1, 1, 1) )

        if self.isCuda:
            temp = temp.cuda(self.gpuId)
            lightPos = lightPos.cuda(self.gpuId)

        coord3D = self.pCoord.expand_as(diffusePred) * distPred.expand_as(diffusePred)
        l = lightPos.expand_as(self.pCoord)  - coord3D
        l = l / torch.sqrt(torch.clamp(torch.sum(l*l, dim = 1).unsqueeze(1), 1e-6, 20) )
        h = (self.v + l) / 2;
        h = h / torch.sqrt(torch.clamp(torch.sum(h*h, dim=1), min = 1e-6).unsqueeze(1) )

        vdh = torch.sum( (self.v * h), dim = 1).unsqueeze(1)
        temp.data[0] = 2.0
        frac0 = self.F0 + (1-self.F0) * torch.pow(temp.expand_as(vdh), (-5.55472*vdh-6.98316)*vdh)

        diffuseBatch = diffusePred / np.pi
        roughBatch = (roughPred + 1.0)/2.0

        k = (roughBatch + 1) * (roughBatch + 1) / 8.0
        alpha = roughBatch * roughBatch
        alpha2 = alpha * alpha

        ndv = torch.clamp(torch.sum(normalPred * self.v.expand_as(normalPred), dim = 1), 0, 1)
        ndh = torch.clamp(torch.sum(normalPred * h.expand_as(normalPred), dim = 1), 0, 1)
        ndl = torch.clamp(torch.sum(normalPred * l.expand_as(normalPred), dim = 1), 0, 1)

        if len(ndv.size()) == 3:
            ndv = ndv.unsqueeze(1)
            ndh = ndh.unsqueeze(1)
            ndl = ndl.unsqueeze(1)

        frac = alpha2 * frac0.expand_as(alpha)
        nom0 = ndh * ndh * (alpha2 - 1) + 1
        nom1 = ndv * (1 - k) + k
        nom2 = ndl * (1 - k) + k
        nom = torch.clamp(4*np.pi*nom0*nom0*nom1*nom2, 1e-6, 4*np.pi)
        specPred = frac / nom

        dist2Pred = torch.sum( (lightPos.expand_as(coord3D) - coord3D) \
                * (lightPos.expand_as(coord3D) - coord3D), dim=1).unsqueeze(1)
        color = (diffuseBatch + specPred.expand_as(diffusePred) ) * ndl.expand_as(diffusePred) * \
                self.lightPower / torch.clamp(dist2Pred.expand_as(diffusePred), 1e-6)
        color = color * segBatch.expand_as(diffusePred)
        return torch.clamp(color, 0, 1)


    def forwardEnv(self, diffusePred, normalPred, roughPred, envmap):

        envR, envC = envmap.size(2), envmap.size(3)
        bn = diffusePred.size(0)

        diffusePred = F.adaptive_avg_pool2d(diffusePred, (envR, envC) )
        normalPred = F.adaptive_avg_pool2d(normalPred, (envR, envC) )
        normalPred = normalPred / torch.sqrt( torch.clamp(
            torch.sum(normalPred * normalPred, dim=1 ), 1e-6, 1).unsqueeze(1) )
        roughPred = F.adaptive_avg_pool2d(roughPred, (envR, envC ) )

        temp = Variable(torch.FloatTensor(1, 1, 1, 1,1) )

        if self.isCuda:
            temp = temp.cuda(self.gpuId)

        ldirections = self.ls.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        camyProj = torch.einsum('b,abcd->acd',(self.up, normalPred)).unsqueeze(1).expand_as(normalPred) * normalPred
        camy = F.normalize(self.up.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(camyProj) - camyProj, dim=1)
        camx = -F.normalize(torch.cross(camy, normalPred,dim=1), p=2, dim=1)

        l = ldirections[:, :, 0:1, :, :] * camx.unsqueeze(1) \
                + ldirections[:, :, 1:2, :, :] * camy.unsqueeze(1) \
                + ldirections[:, :, 2:3, :, :] * normalPred.unsqueeze(1)

        h = (self.v.unsqueeze(1) + l) / 2;
        h = h / torch.sqrt(torch.clamp(torch.sum(h*h, dim=2), min = 1e-6).unsqueeze(2) )

        vdh = torch.sum( (self.v * h), dim = 2).unsqueeze(2)
        temp.data[0] = 2.0
        frac0 = self.F0 + (1-self.F0) * torch.pow(temp.expand_as(vdh), (-5.55472*vdh-6.98316)*vdh)

        diffuseBatch = (diffusePred )/ np.pi
        roughBatch = (roughPred + 1.0)/2.0

        k = (roughBatch + 1) * (roughBatch + 1) / 8.0
        alpha = roughBatch * roughBatch
        alpha2 = alpha * alpha

        ndv = torch.clamp(torch.sum(normalPred * self.v.expand_as(normalPred), dim = 1), 0, 1).unsqueeze(1).unsqueeze(2)
        ndh = torch.clamp(torch.sum(normalPred.unsqueeze(1) * h, dim = 2), 0, 1).unsqueeze(2)
        ndl = torch.clamp(torch.sum(normalPred.unsqueeze(1) * l, dim = 2), 0, 1).unsqueeze(2)

        frac = alpha2.unsqueeze(1).expand_as(frac0) * frac0
        nom0 = ndh * ndh * (alpha2.unsqueeze(1).expand_as(ndh) - 1) + 1
        nom1 = ndv * (1 - k.unsqueeze(1).expand_as(ndh) ) + k.unsqueeze(1).expand_as(ndh)
        nom2 = ndl * (1 - k.unsqueeze(1).expand_as(ndh) ) + k.unsqueeze(1).expand_as(ndh)
        nom = torch.clamp(4*np.pi*nom0*nom0*nom1*nom2, 1e-6, 4*np.pi)
        specPred = frac / nom


        envmap = envmap.view([bn, 3, envR, envC, self.envWidth * self.envHeight ] )
        envmap = envmap.permute([0, 4, 1, 2, 3] )

        brdfDiffuse = diffuseBatch.unsqueeze(1).expand([bn, self.envWidth * self.envHeight, 3, envR, envC] ) * \
                    ndl.expand([bn, self.envWidth * self.envHeight, 3, envR, envC] )
        colorDiffuse = torch.sum(brdfDiffuse * envmap * self.envWeight.expand_as(brdfDiffuse), dim=1)

        brdfSpec = specPred.expand([bn, self.envWidth * self.envHeight, 3, envR, envC ] ) * \
                    ndl.expand([bn, self.envWidth * self.envHeight, 3, envR, envC] )
        colorSpec = torch.sum(brdfSpec * envmap * self.envWeight.expand_as(brdfSpec), dim=1)

        return colorDiffuse, colorSpec



class renderingLayerSG():
    def __init__(self, roughSGDict, roughOutVec, diffuseSGDict, diffuseOutVec = -0.9999247,
            fov=63.4149, F0=0.05, cameraPos = [0, 0, 0], gpuId = 0, imWidth = 160, imHeight = 120, isCuda = True ):
        self.imHeight = imHeight
        self.imWidth = imWidth

        self.fov = fov / 180.0 * np.pi
        self.F0 = F0
        self.cameraPos = np.array(cameraPos, dtype=np.float32).reshape([1, 3, 1, 1])
        self.xRange = 1 * np.tan(self.fov/2)
        self.yRange = float(imHeight) / float(imWidth) * self.xRange
        self.isCuda = isCuda
        x, y = np.meshgrid(np.linspace(-self.xRange, self.xRange, imWidth),
                np.linspace(-self.yRange, self.yRange, imHeight ) )
        y = np.flip(y, axis=0)
        z = -np.ones( (imHeight, imWidth), dtype=np.float32)

        pCoord = np.stack([x, y, z]).astype(np.float32)
        self.pCoord = pCoord[np.newaxis, :, :, :]
        v = self.cameraPos - self.pCoord
        v = v / np.sqrt(np.maximum(np.sum(v*v, axis=1), 1e-12)[:, np.newaxis, :, :] )
        v = v.astype(dtype = np.float32 )

        self.v = Variable(torch.from_numpy(v) )

        self.up = Variable(torch.Tensor([0,1,0] ) )

        assert(roughSGDict.shape[0] == 20
                and roughSGDict.shape[1] == 4  # Number of basis
                and roughSGDict.shape[2] == 4  # Number of SG lobs
                and roughSGDict.shape[3] == 4) # Number of SG parameters for each lobe
        self.roughSGDict = Variable(torch.from_numpy(roughSGDict ) )

        assert(roughOutVec.shape[0] == 20
                and roughOutVec.shape[1] == 4
                and roughOutVec.shape[2] == 16)
        self.roughOutVec = Variable(torch.from_numpy(roughOutVec ) )

        assert(diffuseSGDict.shape[0] == 4
                and diffuseSGDict.shape[1] == 4)
        self.diffuseSGDict = Variable(torch.from_numpy(diffuseSGDict ) )

        self.diffuseOutVec = diffuseOutVec


        if isCuda:
            self.v = self.v.cuda(gpuId )
            self.up = self.up.cuda(gpuId )
            self.roughSGDict = self.roughSGDict.cuda(gpuId )
            self.roughOutVec = self.roughOutVec.cuda(gpuId )
            self.diffuseSGDict = self.diffuseSGDict.cuda(gpuId )

        diffuseSG_theta, diffuseSG_phi, diffuseSG_weight, diffuseSG_lamb = torch.split(self.diffuseSGDict, 1, dim=1)
        diffuseSG_axisX = torch.sin(diffuseSG_theta ) * torch.cos(diffuseSG_phi )
        diffuseSG_axisY = torch.sin(diffuseSG_theta ) * torch.sin(diffuseSG_phi )
        diffuseSG_axisZ = torch.cos(diffuseSG_theta )
        diffuseSG_axis = torch.cat([diffuseSG_axisX, diffuseSG_axisY, diffuseSG_axisZ], dim=1 )
        diffuseSG_weightedAxis = diffuseSG_lamb.expand_as(diffuseSG_axis ) * diffuseSG_axis

        self.diffuseSG_weightedAxis = diffuseSG_weightedAxis.view([1, 1, 1, 1, 4, 1, 3] )
        self.diffuseSG_weight = diffuseSG_weight.view([1, 1, 1, 1, 4, 1, 1] )
        self.diffuseSG_lamb = diffuseSG_lamb.view([1, 1, 1, 1, 4, 1, 1] )

        self.gpuId = gpuId


    def forwardEnv(self, diffuse, normal, rough, theta, phi, weight, lamb ):
        # The size of theta should be
        # BatchSize, 1, envR, envC, 12, 1
        envR, envC = theta.size(2), theta.size(3)
        bn = diffuse.size(0)
        assert(envR == self.imHeight and envC == self.imWidth )

        diffuse = F.adaptive_avg_pool2d(diffuse, (envR, envC) ) / np.pi
        normal = F.adaptive_avg_pool2d(normal, (envR, envC) )
        normal = normal / torch.sqrt( torch.clamp(
            torch.sum(normal * normal, dim=1 ), 1e-6, 1).unsqueeze(1) )
        rough = F.adaptive_avg_pool2d(rough, (envR, envC ) )
        rough = 0.5 * (rough + 1)

        camyProj = torch.einsum('b,abcd->acd',(self.up, normal) ).unsqueeze(1).expand_as(normal) * normal
        camy = F.normalize(self.up.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(camyProj) - camyProj, dim=1)
        camx = -F.normalize(torch.cross(camy, normal, dim=1 ), dim=1 )

        # Rotate the environmental map
        vx = torch.sum(self.v.expand_as(camx ) * camx, dim=1).unsqueeze(1)
        vy = torch.sum(self.v.expand_as(camy ) * camy, dim=1).unsqueeze(1)
        phiDelta = (np.pi - torch.atan2(vy, vx ) ).unsqueeze(-1).unsqueeze(-1)
        phi = phi + phiDelta.expand_as(phi )

        # Get the output theta and thetaInd
        thetaV = torch.acos(torch.sum(self.v.expand_as(normal ) * normal, dim=1).unsqueeze(1) ).detach()
        thetaVId = thetaV / (np.pi / 2) * 16 - 1
        thetaVId.requires_grad = False
        thetaVId_u = torch.clamp(torch.ceil(thetaVId ), 0, 15)
        thetaVId_l = torch.clamp(torch.floor(thetaVId ), 0, 15)

        tw_u = torch.clamp(thetaVId - thetaVId_l, 0, 1)
        tw_l = 1 - tw_u

        thetaVId_u = thetaVId_u.unsqueeze(-1).unsqueeze(-1)
        thetaVId_l = thetaVId_l.unsqueeze(-1).unsqueeze(-1)
        thetaVId_u = torch.cat( [thetaVId_u, thetaVId_u, thetaVId_u], dim=4 )
        thetaVId_l = torch.cat( [thetaVId_l, thetaVId_l, thetaVId_l], dim=4 )

        # Turn the phi and theta of environmental map
        axis_x = torch.sin(theta ) * torch.cos(phi )
        axis_y = torch.sin(theta ) * torch.sin(phi )
        axis_z = torch.cos(theta )
        axis = torch.cat([axis_x, axis_y, axis_z], dim=5 )
        weightedAxis = axis * lamb.expand_as(axis )

        #################################################################
        ###### Compute the Diffuse Image ######
        lamb = lamb.unsqueeze(4)
        lamb = lamb.expand([bn, 1, self.imHeight, self.imWidth, 4, 12, 3] )

        weightedAxis = weightedAxis.unsqueeze(4)
        weightedAxis = weightedAxis.expand([bn, 1, self.imHeight, self.imWidth, 4, 12, 3] )

        weight = weight.unsqueeze(4)
        weight = weight.expand([bn, 1, self.imHeight, self.imWidth, 4, 12, 3] )

        diffuseSG_coef = self.diffuseSG_weight.expand_as(weight ) * weight
        diffuseSG_decay = -(self.diffuseSG_lamb.expand_as(lamb ) + lamb )
        diffuseSG_r = torch.norm(
                weightedAxis + self.diffuseSG_weightedAxis.expand_as(weightedAxis ), p=2, dim=6).unsqueeze(-1)
        diffuseSG_intL = torch.sum(torch.sum(4 * np.pi * diffuseSG_coef *
            (torch.exp(diffuseSG_r + diffuseSG_decay)  - torch.exp(-diffuseSG_r + diffuseSG_decay) )
            / 2 / torch.clamp(diffuseSG_r, 1e-12, 1e10), dim = 4), dim=4).squeeze(-1).squeeze(1)

        diffuseSG_intL = diffuseSG_intL.transpose(2, 3).transpose(1, 2)
        diffIm = self.diffuseOutVec * diffuseSG_intL * diffuse
        diffIm = torch.clamp(diffIm, min=0)


        #################################################################
        ###### Compute the Specular Image ######
        lamb = lamb.unsqueeze(4)
        lamb = lamb.expand([bn, 1, self.imHeight, self.imWidth, 4, 4, 12, 3] )

        weightedAxis = weightedAxis.unsqueeze(4)
        weightedAxis = weightedAxis.expand([bn, 1, self.imHeight, self.imWidth, 4, 4, 12, 3] )

        weight = weight.unsqueeze(4)
        weight = weight.expand(bn, 1, self.imHeight, self.imWidth, 4, 4, 12, 3)

        # From roughness to SG
        roughInd = (rough * 20 - 1).detach()
        roughInd_u = torch.ceil(roughInd ).detach()
        roughInd_u = torch.clamp(roughInd_u, 0, 19)
        roughInd_u.requires_grad = False
        roughInd_l = torch.floor(roughInd ).detach()
        roughInd_l = torch.clamp(roughInd_l, 0, 19)
        roughInd_l.requires_grad = False

        rw_u = torch.clamp(roughInd - roughInd_l, 0, 1)
        rw_l = 1.0 - rw_u

        roughSG_u = torch.index_select(self.roughSGDict, dim=0, index = roughInd_u.view(-1).long() )
        roughSG_u = roughSG_u.view([bn, 1, envR, envC, 4, 4, 4] )
        roughSG_l = torch.index_select(self.roughSGDict, dim=0, index = roughInd_l.view(-1).long() )
        roughSG_l = roughSG_l.view([bn, 1, envR, envC, 4, 4, 4] )

        roughOut_u = torch.index_select(self.roughOutVec, dim=0, index = roughInd_u.view(-1).long() )
        roughOut_u = roughOut_u.view([bn, 1, envR, envC, 4, 1, 16] )
        roughOut_l = torch.index_select(self.roughOutVec, dim=0, index = roughInd_l.view(-1).long() )
        roughOut_l = roughOut_l.view([bn, 1, envR, envC, 4, 1, 16] )

        ###### Do the integral ######
        # 1. Split the roughSG into component and build the axis
        roughSG_u_theta, roughSG_u_phi, roughSG_u_weight, roughSG_u_lamb = torch.split(roughSG_u, 1, dim=6)
        roughSG_u_axisX = torch.sin(roughSG_u_theta ) * torch.cos(roughSG_u_phi )
        roughSG_u_axisY = torch.sin(roughSG_u_theta ) * torch.sin(roughSG_u_phi )
        roughSG_u_axisZ = torch.cos(roughSG_u_theta )
        roughSG_u_axis = torch.cat([roughSG_u_axisX, roughSG_u_axisY, roughSG_u_axisZ], dim=6 )
        roughSG_u_weightedAxis = roughSG_u_lamb.expand_as(roughSG_u_axis) * roughSG_u_axis

        roughSG_l_theta, roughSG_l_phi, roughSG_l_weight, roughSG_l_lamb = torch.split(roughSG_l, 1, dim=6)
        roughSG_l_axisX = torch.sin(roughSG_l_theta ) * torch.cos(roughSG_l_phi )
        roughSG_l_axisY = torch.sin(roughSG_l_theta ) * torch.sin(roughSG_l_phi )
        roughSG_l_axisZ = torch.cos(roughSG_l_theta )
        roughSG_l_axis = torch.cat([roughSG_l_axisX, roughSG_l_axisY, roughSG_l_axisZ], dim=6 )
        roughSG_l_weightedAxis = roughSG_l_lamb.expand_as(roughSG_l_axis) * roughSG_l_axis

        # 2. Do the integral
        roughSG_u_weight = roughSG_u_weight.unsqueeze(dim = 6)
        roughSG_u_weightedAxis = roughSG_u_weightedAxis.unsqueeze(dim = 6)
        roughSG_u_lamb = roughSG_u_lamb.unsqueeze(dim = 6)

        roughSG_u_coef = roughSG_u_weight.expand_as(weight ) * weight
        roughSG_u_decay = -(lamb + roughSG_u_lamb.expand_as(lamb ) )
        roughSG_u_weightedAxis = roughSG_u_weightedAxis.expand(bn, 1, self.imHeight, self.imWidth, 4, 4, 12, 3)
        roughSG_u_r = torch.norm(roughSG_u_weightedAxis.expand(bn, 1, self.imHeight, self.imWidth, 4, 4, 12, 3) +
                weightedAxis.expand(bn, 1, self.imHeight, self.imWidth, 4, 4, 12, 3), p=2, dim=7).unsqueeze(-1)

        roughSG_u_r_log = torch.log(torch.clamp(roughSG_u_r, min=1e-20) )
        roughSG_u_intL = torch.sum(torch.sum(4 * np.pi * roughSG_u_coef *
            (torch.exp( torch.clamp(roughSG_u_r + roughSG_u_decay, max=0) - roughSG_u_r_log )
            - torch.exp(-roughSG_u_r + roughSG_u_decay - roughSG_u_r_log ) )
            /2, dim = 5), dim=5).unsqueeze(-1)

        roughSG_l_weight = roughSG_l_weight.unsqueeze(dim = 6)
        roughSG_l_weightedAxis = roughSG_l_weightedAxis.unsqueeze(dim = 6)
        roughSG_l_lamb = roughSG_l_lamb.unsqueeze(dim = 6)

        roughSG_l_coef = roughSG_l_weight.expand_as(weight ) * weight
        roughSG_l_decay = -(lamb + roughSG_l_lamb.expand_as(lamb ) )
        roughSG_l_r = torch.norm(roughSG_l_weightedAxis.expand(bn, 1, self.imHeight, self.imWidth, 4, 4, 12, 3) +
                weightedAxis.expand(bn, 1, self.imHeight, self.imWidth, 4, 4, 12, 3), p=2, dim=7).unsqueeze(-1)
        roughSG_l_r_log = torch.log(torch.clamp(roughSG_l_r, min=1e-20) )
        roughSG_l_intL = torch.sum(torch.sum(4 * np.pi * roughSG_l_coef *
            (torch.exp( torch.clamp(roughSG_l_r + roughSG_l_decay, max=0) - roughSG_l_r_log )
            - torch.exp(-roughSG_l_r + roughSG_l_decay - roughSG_l_r_log ) )
            /2, dim = 5), dim=5).unsqueeze(-1)

        # 3. Get the output
        spec_u = torch.sum(
                roughOut_u.expand(bn, 1, self.imHeight, self.imWidth, 4, 3, 16 ) *
                roughSG_u_intL.expand(bn, 1, self.imHeight, self.imWidth, 4, 3, 16), dim = 4)
        spec_u_u = torch.gather(spec_u, dim = 5, index = thetaVId_u.long() ).squeeze(1).squeeze(-1)
        spec_u_l = torch.gather(spec_u, dim = 5, index = thetaVId_l.long() ).squeeze(1).squeeze(-1)
        spec_u_u = spec_u_u.transpose(2, 3).transpose(1, 2)
        spec_u_l = spec_u_l.transpose(2, 3).transpose(1, 2)
        spec_u = tw_u.expand_as(spec_u_u) * spec_u_u + tw_l.expand_as(spec_u_l) * spec_u_l

        spec_l = torch.sum(
                roughOut_l.expand(bn, 1, self.imHeight, self.imWidth, 4, 3, 16 ) *
                roughSG_l_intL.expand(bn, 1, self.imHeight, self.imWidth, 4, 3, 16), dim = 4)
        spec_l_u = torch.gather(spec_l, dim = 5, index = thetaVId_u.long() ).squeeze(1).squeeze(-1)
        spec_l_l = torch.gather(spec_l, dim = 5, index = thetaVId_l.long() ).squeeze(1).squeeze(-1)
        spec_l_u = spec_l_u.transpose(2, 3).transpose(1, 2)
        spec_l_l = spec_l_l.transpose(2, 3).transpose(1, 2)
        spec_l = tw_u.expand_as(spec_l_u) * spec_l_u + tw_l.expand_as(spec_l_l) * spec_l_l

        specIm = spec_l * rw_l.expand_as(spec_l) + spec_u * rw_u.expand_as(spec_u )
        specIm = torch.clamp(specIm, min=0 )

        return diffIm, specIm



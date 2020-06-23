import SGOptim
import torch
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import argparse
import random
import os
import dataLoader
from torch.utils.data import DataLoader
import time
import os.path as osp
import h5py
import cv2

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot', default=None, help='path to images')
# The basic training setting
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--envRow', type=int, default=120, help='the number of samples of envmaps in y direction')
parser.add_argument('--envCol', type=int, default=160, help='the number of samples of envmaps in x direction')
parser.add_argument('--SGNum', type=int, default=12, help='the number of SG parameters used for approximation environmental maps')
parser.add_argument('--envHeight', type=int, default=16, help='the size of envmaps in y direction')
parser.add_argument('--envWidth', type=int, default=32, help='the size of envmaps in x direction')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network')
parser.add_argument('--rs', type=int, default=0, help='start point of optimizing the environmental map')
parser.add_argument('--re', type=int, default=10000, help='end point of optimizing the environmental map')

# The detail network setting
opt = parser.parse_args()
print(opt)


opt.gpuId = opt.deviceIds[0]
assert(opt.batchSize == 1)
assert(opt.envRow == 120 and opt.envCol == 160 )
assert(opt.envHeight == 16 and opt.envWidth == 32 )
assert(opt.SGNum == 12 )

opt.seed = 0
print("Random Seed: ", opt.seed)
random.seed(opt.seed )
torch.manual_seed(opt.seed )

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

####################################
brdfDataset = dataLoader.BatchLoader(
        imWidth = opt.envCol, imHeight = opt.envRow,
        dataRoot = opt.dataRoot, rs = opt.rs, re = opt.re,
        envHeight = opt.envHeight, envWidth = opt.envWidth,
        envRow = opt.envRow, envCol = opt.envCol,
        isAllLight = True, isLight = True )
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize, num_workers = 0, shuffle = False)
envNum = opt.envRow * opt.envCol

for i, dataBatch in enumerate(brdfLoader):
    maskBatch = dataBatch['segObj']
    envBatch = dataBatch['envmaps']
    envIndBatch = dataBatch['envmapsInd']
    nameBatch = dataBatch['name']

    env = envBatch[0, :]
    env = torch.transpose(torch.transpose(env, 0, 1), 1, 2)
    env = env.view(opt.envRow * opt.envCol, 3, opt.envHeight, opt.envWidth )

    mask = maskBatch[0, :]
    mask = mask.view(opt.envRow * opt.envCol, 1, 1, 1)
    name = nameBatch[0]
    envmapInd = envIndBatch[0].data.item()

    nameParts = name.split('/')
    nameParts[-1] = nameParts[-1].replace('im_', 'imsgEnv_'
            ).replace('.hdr', '.h5')
    nameNew = '/'.join(nameParts )
    
    print(nameNew )
    if osp.isfile(nameNew ):
        continue

    if envmapInd == 1:

        start = time.time()

        env1 = env[0:int(envNum/2), : ]
        mask1 = mask[0:int(envNum/2), : ]

        envOptim1 = SGOptim.SGEnvOptim(
                envNum = env1.shape[0],
                envWidth = opt.envWidth,
                envHeight = opt.envHeight,
                niter = 10 )
        theta1, phi1, lamb1, weight1,  recImage1 = envOptim1.optimize(env1, mask1 )
        del envOptim1

        env2 = env[ int(envNum/2) : envNum, : ]
        mask2 = mask[ int(envNum/2) : envNum, : ]

        envOptim2 = SGOptim.SGEnvOptim(
                envNum = env2.shape[0],
                envWidth = opt.envWidth,
                envHeight = opt.envHeight,
                niter = 10 )
        theta2, phi2, lamb2, weight2, recImage2 = envOptim2.optimize(env2, mask2 )
        del envOptim2 

        if (theta1 is None) or (theta2 is None):
            continue

        theta = np.concatenate([theta1, theta2], axis=0 )
        phi = np.concatenate([phi1, phi2], axis=0 )
        lamb = np.concatenate([lamb1, lamb2], axis=0 )
        weight = np.concatenate([weight1, weight2], axis=0 )
        recImage = np.concatenate([recImage1, recImage2], axis=0 )

        SGParams = np.concatenate([theta, phi, lamb, weight], axis=2 )
        SGParams = SGParams.reshape([opt.envRow, opt.envCol, opt.SGNum, 6] )

        print(nameNew )
        hf = h5py.File(nameNew, 'w')
        hf.create_dataset('data', data = SGParams, compression = 'lzf')
        hf.close()

        # Debug
        '''
        recImage = recImage.transpose([0, 2, 3, 1] )
        recImage = recImage.reshape([opt.envRow, opt.envCol, \
                opt.envHeight, opt.envWidth, 3] )
        recImage = recImage.transpose([0, 2, 1, 3, 4] )
        recImage = recImage.reshape([opt.envHeight * opt.envRow, \
                opt.envWidth * opt.envCol, 3] )
        cv2.imwrite(nameNew.replace('.h5', '.hdr'), recImage )
        '''
        end = time.time()
        print('Time Interval: %.5s s' % (end -start) )
    else:
        print('Warning: %s envmap will be skipped' % name )

import glob
import numpy as np
import os.path as osp
from PIL import Image
import random
import struct
from torch.utils.data import Dataset
import scipy.ndimage as ndimage
import cv2
from skimage.measure import block_reduce
import h5py
import scipy.ndimage as ndimage


class BatchLoader(Dataset):
    def __init__(self, dataRoot, rs, re,
            dirs = ['main_xml', 'main_xml1',
                'mainDiffLight_xml', 'mainDiffLight_xml1',
                'mainDiffMat_xml', 'mainDiffMat_xml1'],
            imHeight = 240, imWidth = 320,
            rseed = None, cascadeLevel = 0,
            isLight = False, isAllLight = False,
            envHeight = 8, envWidth = 16, envRow = 120, envCol = 160,
            SGNum = 6 ):

        self.dataRoot = dataRoot
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.cascadeLevel = cascadeLevel
        self.isLight = isLight
        self.isAllLight = isAllLight
        self.envWidth = envWidth
        self.envHeight = envHeight
        self.envRow = envRow
        self.envCol = envCol
        self.envWidth = envWidth
        self.envHeight = envHeight
        self.SGNum = SGNum

        shapeList = []
        for d in dirs:
            shapeList = shapeList + glob.glob(osp.join(self.dataRoot, d, 'scene*') )
        shapeList = sorted(shapeList )

        self.rs = min(rs, len(shapeList ) )
        self.re = min(re, len(shapeList ) )

        shapeList = shapeList[self.rs : self.re]
        print('Shape Num: %d' % len(shapeList ) ) 

        self.imList = []
        for shape in shapeList:
            imNames = sorted(glob.glob(osp.join(shape, 'im_*.hdr') ) )
            self.imList = self.imList + imNames

        if isAllLight:
            self.imList = [x for x in self.imList if
                    osp.isfile(x.replace('im_', 'imenv_') ) ] 
            self.imList = [x for x in self.imList if not
                    osp.isfile(x.replace('im_', 'imsgEnv_').replace('.hdr', '.h5') ) ] 


        print('Image Num: %d' % len(self.imList ) )

        self.segList = [x.replace('im_', 'immask_').replace('mainDiffMat', 'main').replace('hdr', 'png') for x in self.imList ]
        self.envList = [x.replace('im_', 'imenv_') for x in self.imList ]

        # Permute the image list
        self.count = len(self.imList )
        self.perm = list(range(self.count ) )

    def __len__(self):
        return len(self.perm )

    def __getitem__(self, ind):
        # Read segmentation
        seg = 0.5 * (self.loadImage(self.segList[self.perm[ind] ] ) + 1)[0:1, :, :]
        segArea = np.logical_and(seg > 0.49, seg < 0.51 ).astype(np.float32 )
        segEnv = (seg < 0.001).astype(np.float32 )
        segObj = (seg > 0.999)

        if self.isLight:
            segObj = segObj.squeeze()
            segObj = ndimage.binary_erosion(segObj, structure=np.ones((7, 7) ), border_value=1)
            segObj = segObj[np.newaxis, :, :]

        segObj = segObj.astype(np.float32 )

        envmaps, envmapsInd = self.loadEnvmap(self.envList[self.perm[ind] ] )
        batchDict = {
                'segArea': segArea,
                'segEnv': segEnv,
                'segObj': segObj,
                'envmaps': envmaps,
                'envmapsInd': envmapsInd,
                'name': self.imList[self.perm[ind ] ]
                }

        return batchDict


    def loadImage(self, imName, isGama = False):
        if not(osp.isfile(imName ) ):
            print(imName )
            assert(False )

        im = Image.open(imName)
        im = im.resize([self.imWidth, self.imHeight], Image.ANTIALIAS )

        im = np.asarray(im, dtype=np.float32)
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1] )

        return im

    def loadEnvmap(self, envName ):
        if not osp.isfile(envName ):
            env = np.zeros( [3, self.envRow, self.envCol,
                self.envHeight, self.envWidth], dtype = np.float32 )
            envInd = np.zeros([1, 1, 1], dtype=np.float32 )
            print('Warning: the envmap %s does not exist.' % envName )
            return env, envInd
        else:
            envHeightOrig, envWidthOrig = 16, 32
            assert( (envHeightOrig / self.envHeight) == (envWidthOrig / self.envWidth) )
            assert( envHeightOrig % self.envHeight == 0)

            env = cv2.imread(envName, -1 )

            if not env is None:
                env = env.reshape(self.envRow, envHeightOrig, self.envCol,
                    envWidthOrig, 3)
                env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3] ) )

                scale = envHeightOrig / self.envHeight
                if scale > 1:
                    env = block_reduce(env, block_size = (1, 1, 1, 2, 2), func = np.mean )

                envInd = np.ones([1, 1, 1], dtype=np.float32 )
                return env, envInd
            else:
                env = np.zeros( [3, self.envRow, self.envCol,
                    self.envHeight, self.envWidth], dtype = np.float32 )
                envInd = np.zeros([1, 1, 1], dtype=np.float32 )
                print('Warning: the envmap %s does not exist.' % envName )
                return env, envInd

            return env, envInd

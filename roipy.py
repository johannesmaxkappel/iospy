'''
Basic image processing functions for processing ROI images

Copyright E.Chong & J.Kappel 2017
'''

import pickle as p
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def mask_image(img):
    assert os.path.exists(img), \
        'Path does not exist!'
    assert os.path.exists(os.path.join('C:/voyeur_rig_config','PolygonMask.png')), \
        'No PolygonMask.png in voyeur_rig_config!'
    maskimg = cv2.imread('PolygonMask.png')
    maskimg = cv2.cvtColor(maskimg, cv2.COLOR_BGR2GRAY)
    y, x = np.where(maskimg == 255)

    mask = {'dx_start': x.min(),
            'dx_stop': x.max(),
            'dy_start': y.min(),
            'dy_stop': y.max()
            }

    return img[mask['dy_start']:mask['dy_stop'],
           mask['dx_start']:mask['dx_stop']]
    pass

def create_projection(img, imgname, imgsource, bmp=False):
    assert os.path.exists(os.path.join('C:/voyeur_rig_config','Matrix.pkl')), \
        'No Matrix.pkl in cwd!'
    matrix = open('Matrix.pkl', 'rb')
    CAM2DMD = p.load(matrix)
    img = mask_image(img)
    warpim = cv2.warpPerspective(img, CAM2DMD, (684, 608), borderMode=1, borderValue=1)
    # switch rows and columns
    warpim = np.transpose(warpim)
    for y, row in enumerate(warpim):
        for x, value in enumerate(row):
            if not value in [0, 255]:
                warpim[y][x] = 255
    if bmp:
        cv2.imwrite(os.path.join(imgsource, 'transformed_{0}.bmp'.format(imgname.split('.')[0])), warpim)
    cv2.imwrite(os.path.join(imgsource, 'transformed_{0}.png'.format(imgname.split('.')[0])), warpim)
    pass

def shift_frame(img,rowshift, colshift):
    ### shifting rows and columns to move spots into FOS ###
    width, height = img.shape
    print 'Rows shifted: {0}. Columns shifted: {1}.'.format(rowshift, colshift)
    img = np.insert(img, height, np.zeros((rowshift, width)), axis=0)
    img = np.insert(img, 0, np.zeros((colshift, height + rowshift)), axis=1)
    img = img[rowshift:height + rowshift, 0:width] * (2 ** 16)
    return img

def shiftandwarp(mouse, date, rowshift=0, colshift=0):
    path = 'C:/VoyeurData/{0}/spots/{1}'.format(mouse, date)
    assert os.path.exists(path), 'Path for {0} does not exist!'.format(mouse)
    for imgname in os.listdir(path):

        if imgname.startswith('ref'):
            img = plt.imread(os.path.join(path, imgname))
            img = shift_frame(img,rowshift,colshift)
            cv2.imwrite(os.path.join(path, 'shifted_{0}.png'.format(imgname.split('.')[0])), img.astype('uint16'))
        elif imgname.startswith('Mask'):
            img = plt.imread(os.path.join(path, imgname))
            assert len( np.unique( img ) ) == 2, 'Data format not binary!'
            img = shift_frame(img,rowshift,colshift)
            create_projection(img, imgname, path)
        else:
            continue
    pass
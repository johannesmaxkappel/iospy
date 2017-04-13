'''
Basic image processing functions for processing ROI images

Copyright E.Chong & J.Kappel 2017
'''

import pickle as p
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import zipfile

def mask_image(img):

    path = 'C:/voyeur_rig_config/PolygonMask.png'
    assert os.path.exists(path), 'No PolygonMask.png in voyeur_rig_config!'
    maskimg = cv2.imread(path)
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

    matrixpath = 'C:/voyeur_rig_config/Matrix.pkl'
    assert os.path.exists(matrixpath), 'No Matrix.pkl in cwd!'
    matrix = open((matrixpath), 'rb')
    CAM2DMD = p.load(matrix)
    img = mask_image(img)
    warpimg = cv2.warpPerspective(img, CAM2DMD, (684, 608), borderMode=1, borderValue=1)
    # switch rows and columns
    warpimg = np.transpose(warpimg)
    for y, row in enumerate(warpimg):
        for x, value in enumerate(row):
            if not value in [0, 255]:
                warpimg[y][x] = 255
    if bmp:
        cv2.imwrite(os.path.join(imgsource, 'transformed_{0}.bmp'.format(imgname.split('.')[0])), warpimg)
    cv2.imwrite(os.path.join(imgsource, 'transformed_{0}.png'.format(imgname.split('.')[0])), warpimg)
    pass


def shift_frame(img,rowshift, colshift):

    ### shifting rows and columns to move spots into FOS ###
    width, height = img.shape

    print 'Rows shifted: {0}. Columns shifted: {1}.'.format(rowshift, colshift)
    img = np.insert(img, height, np.zeros((rowshift, width)), axis=0)
    img = np.insert(img, 0, np.zeros((colshift, height + rowshift)), axis=1)
    img = img[rowshift:height + rowshift, 0:width] * (2 ** 16)
    return img


def transform_masks(mouse, date, rowshift=0, colshift=0):

    path = 'C:/VoyeurData/{0}/spots/{1}'.format(mouse, date)
    refimg = False
    assert os.path.exists(path), 'Path for {0} does not exist!'.format(mouse)
    for imgname in os.listdir(path):

        if imgname.startswith('ref'):
            refimg = True
            img = plt.imread(os.path.join(path, imgname))
            img = shift_frame(img,rowshift,colshift)
            img = img.astype('uint16')
            cv2.imwrite(os.path.join(path, 'shifted_{0}.png'.format(imgname.split('.')[0])), img)
        elif imgname.startswith('Mask'):
            print 'Transforming image: {0}'.format(imgname)
            img = plt.imread(os.path.join(path, imgname))
            assert len(img.shape) == 2, 'Data format not 2-dimensional!'
            assert len( np.unique( img ) ) == 2, 'Data format not binary!'
            img = shift_frame(img,rowshift,colshift)
            create_projection(img, imgname, path)
        else:
            continue
    if not refimg:
        print 'WARNING: No reference image was shifted!'
    pass

def read_roi(fileobj):

    if fileobj[:4] != 'Iout':
        raise IOError('Magic number not found')

    y1 = ord(fileobj[9:10])
    x1 = ord(fileobj[11:12])
    y2 = ord(fileobj[13:14])
    x2 = ord(fileobj[15:16])

    frame = np.zeros(256**2)
    frame = frame.reshape(256, 256)
    frame[y1:y2, x1:x2] = 255
    return frame

def create_masks(mouse, date, transform=True):

    path = 'C:/VoyeurData/{0}/spots/{1}'.format(mouse, date)
    zf = zipfile.ZipFile(os.path.join(path,'RoiSet.zip'))
    for roi in zf.namelist():
        rfile = zf.open(roi, 'r')
        fileobj = rfile.read()
        frame = read_roi(fileobj)
        cv2.imwrite(os.path.join(path, 'Mask_{0}.png'.format(roi[:3])), frame.astype('uint8'))
        if transform:
            create_projection(frame, 'Mask_{0}.png'.format(roi[:3]), path)
    pass
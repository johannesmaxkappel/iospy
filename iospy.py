'''
Basic image processing functions for imaging data from NeuroCCD camera

Copyright J. Kappel & E.Chong 2017

'''

from scipy.ndimage import fourier_gaussian
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import sys
import os
sns.set_style('white')

def read_data(filepath, dFrame = True):

    fn = open(filepath, 'rb')
    header = fn.read(2880)
    pixels = np.fromstring(fn.read(), dtype='int16')

    header = [header[i:i + 80] for i in range(0, len(header), 80)]
    # remove all whitespaces
    for i, h in enumerate(header):
        header[i] = h.replace(' ', '')

    for h in header:
        if h == 'END':
            break
        else:
            param, value = h.split('=')
        if param == 'NAXIS1':
            width = int(value)
        elif param == 'NAXIS2':
            height = int(value)
        elif param == 'NAXIS3':
            nFrames = int(value)
        else:
            pass

    nPixels = width * height
    print 'Resolution:{0}x{1}'.format(width, height)
    print 'Number of pixels:', nPixels
    sys.stdout.flush()
    trialframes = []
    for frameno in range(0, nFrames+1):

        frame = pixels[frameno * nPixels:(frameno + 1) * nPixels]
        frame = np.reshape(frame, (width, height))
        frame = frame.astype('uint16')

        trialframes.append(frame)

    print 'Number of processed frames:', len(trialframes)
    sys.stdout.flush()
    darkframe = trialframes[nFrames]
    if dFrame:
        for frameno, frame in enumerate(trialframes):
            frame = frame - darkframe
            trialframes[frameno] = frame
    return trialframes


def compute_average(trialframes, t1=500, t2=1600):

    blframes_mean = np.mean(trialframes[10:t1-10], axis=0)
    oframes_mean = np.mean(trialframes[t1+10:t2-10], axis=0)
    odor_normed = (oframes_mean-blframes_mean)/oframes_mean
    odor_normed[~ np.isfinite(odor_normed)] = odor_normed.min()
    return odor_normed


def plot_signal(trialframes, t1, t2):

    blframes_mean = np.mean(trialframes[10:t1-10], axis=0)
    oframes = trialframes[t1+10:t2-10]
    oframes_seq = []
    for oframe in oframes:
        oframe_r = (oframe - blframes_mean)/blframes_mean
        oframes_seq.append(np.mean(oframe_r))

    plt.plot([x for x in range(0, len(oframes_seq))], [np.mean(frame) for frame in oframes_seq])
    plt.show()
    pass


def rescale_int(im, lower=2, upper=98):

    v_min, v_max = np.percentile(im, (lower, upper))
    return exposure.rescale_intensity(im, in_range=(v_min, v_max))


def normalize(im):

    min_img = im - im.min()
    return (min_img / min_img.max())


def convert_dtype(im):

    im = im * 2 ** 16
    return im.astype('uint16')


def process_im(im,
               scale=85,
               int_lower=2,
               int_upper=98,
               cut=(5,251),
               bpass=True,
               resc_int=True
               ):
    width, height = im.shape
    odor_normed = im[cut[0]:cut[1], ]
    lframe = np.split(odor_normed, 2, axis=1)[0]
    rframe = np.split(odor_normed, 2, axis=1)[1]
    pframes = []

    for split in [lframe, rframe]:
        split = normalize(split)
        if resc_int:
            split = rescale_int(split, lower=int_lower, upper=int_upper)
        pframes.append(split)

    a = np.concatenate(pframes, axis=1)
    if bpass:
        a = bp_fft(a, scale=scale)
        a = normalize(a)
    a = np.ravel(a)

    r1 = np.zeros(width * (cut[0]))
    r2 = np.zeros(width * (width - cut[1]))

    a = np.insert(a, 0, r1, axis=0)
    a = np.insert(a, cut[1] * width, r2, axis=0)

    a = np.reshape(a, (width, height))
    a = convert_dtype(a)
    return a


def bp_fft(im, scale=85):

    sigma = scale * 0.15
    input_im = np.fft.fft2(im)
    low = fourier_gaussian(input_im, sigma=sigma)
    input_im = input_im - low
    high = fourier_gaussian(input_im, sigma=0.3)
    highr = np.fft.ifft2(high)
    return highr.real


def process_ref(mouse,date, extension=''):

    path = 'C:/Turbo-SM/SMDATA/{0}_{1}_ref{2}'.format(mouse,date,extension)
    assert os.path.exists(path), 'File path not found!'
    spotpath = 'C:/VoyeurData/{0}/spots/{1}'.format(mouse, date)
    if not os.path.exists(spotpath):
        os.makedirs(spotpath)
    tsmcount = 0
    for tsm in os.listdir(path):
        if not tsm.endswith('tsm'):
            continue
        tsmcount += 1
        print 'processing file no. {0}: {1}'.format(tsmcount, tsm)
        ref_frames = read_data(os.path.join(path, tsm))
        ref_average = np.mean(ref_frames, axis=0)
        ref_average = process_average(ref_average, bpass=False)
        cv2.imwrite(os.path.join(path, 'ref_{0}_{1}_{2}.png'.format(mouse, date, tsmcount)), ref_average)
        cv2.imwrite(os.path.join(spotpath, 'ref_{0}_{1}_{2}.png'.format(mouse, date, tsmcount)), ref_average)
    return


def process_single_odorant(mouse,
                           date,
                           odorant,
                           ref=True,
                           average=True,
                           bpass=True,
                           resc_int=True,
                           scale=85,
                           cut=(5,251),
                           path=''
                           ):

    if path == '':
        path = 'C:/Turbo-SM/SMDATA/{0}_{1}_{2}'.format(mouse, date, odorant)
    else:
        path = os.path.join(path, '{0}_{1}_{2}'.format(mouse, date, odorant))
    assert os.path.exists(path), 'File path not found!'

    spotpath = 'C:/VoyeurData/{0}/spots/{1}'.format(mouse, date)
    if not os.path.exists(spotpath):
        os.makedirs(spotpath)
    odor_averaged = []
    tsmcount = 0
    if ref:
        process_ref(mouse, date)
    for tsm in os.listdir(path):
        if not tsm.endswith('tsm'):
            continue
        tsmcount += 1
        print 'processing file no. {0}: {1}'.format(tsmcount, tsm)
        trialframes = read_data(os.path.join(path,tsm))
        odor_normed = compute_average(trialframes, t1=500, t2=len(trialframes))
        odor_final = process_im(odor_normed, cut=cut, bpass=bpass, scale=scale, resc_int=resc_int)
        cv2.imwrite(os.path.join(path, '{0}_{1}_{2}_trial{3}.tif'.format(mouse, date, odorant, tsmcount)), odor_final)
        cv2.imwrite(os.path.join(spotpath, '{0}_{1}_{2}_trial{3}.tif'.format(mouse, date, odorant, tsmcount)), odor_final)
        plt.imshow(odor_final)
        plt.show()
        odor_averaged.append(odor_normed)

    if len(odor_averaged) > 1:
        if not average:
            pass
        average_final = np.mean(odor_averaged, axis=0)
        average_final = process_im(average_final, cut=cut, bpass=bpass, scale=scale, resc_int=resc_int)
        cv2.imwrite(os.path.join(path, '{0}_{1}_{2}_averaged.tif'.format(mouse, date, odorant)), average_final)
        cv2.imwrite(os.path.join(spotpath, '{0}_{1}_{2}_averaged.tif'.format(mouse, date, odorant)), average_final)
        plt.imshow(average_final)
        plt.show()
        pass
    else:
        pass


def process_imaging_sess(mouse,
                         date,
                         path='',
                         bpass=True,
                         resc_int = True,
                         ref=False
                         ):
    if path == '':
        path = 'C:/Turbo-SM/SMDATA/'
    for imgfolder in os.listdir(path):
        if imgfolder.startswith('{0}_{1}'.format(mouse, date)):
            if imgfolder.endswith('ref'):
                continue
            odorant = '_'.join(imgfolder.split('_')[2:])
            process_single_odorant(mouse, date, odorant, ref=False, bpass=bpass, path=path, resc_int=resc_int)
    if ref:
        process_ref(mouse, date)
    pass
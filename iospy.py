'''
Basic image processing functions for imaging data from NeuroCCD camera

Copyright E.Chong & J.Kappel 2017

'''


import scipy.ndimage.filters as filters
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

    plt.plot([x for x in range(0, len(oframes_seq))], [np.mean(frame.astype('uint16')) for frame in oframes_seq])
    plt.show()
    pass

def process_average(odor_normed,
                    lowpass=True,
                    rescale_int=True,
                    int_lower=2,
                    int_upper=98,
                    row1=0,
                    row2=256,
                    scale = 150
                    ):
    width, height = odor_normed.shape
    # cut rows
    odor_normed = odor_normed[row1:row2,]
    lframe = np.split(odor_normed, 2, axis=1)[0]
    rframe = np.split(odor_normed, 2, axis=1)[1]
    pframes = []

    for split in [lframe, rframe]:

        if lowpass:
            low_freq_component = filters.gaussian_filter( split, sigma = scale * 0.175 )
            split = split - low_freq_component

        # subtract baseline
        min_img = split - split.min()
        # normalize and 16bit convert
        norm_img = ((min_img / min_img.max()) * (2 ** 16)).round()
        # convert data type
        bit16_img = norm_img.astype('uint16')

        if rescale_int:
            v_min, v_max = np.percentile(bit16_img, (int_lower, int_upper))
            bit16_img = exposure.rescale_intensity(bit16_img, in_range=(v_min, v_max))

        pframes.append(bit16_img)

    #insert black rows to keep the shape
    a = np.concatenate(pframes, axis=1)
    a = a.flatten()
    r1 = np.zeros(width * (row1))
    r2 = np.zeros(width * (width - row2))

    a = np.insert(a, 0, r1, axis=0)
    a = np.insert(a, row2, r2, axis=0)
    finalimage = np.reshape(a, (width, height))
    return finalimage

def process_ref(mouse,date):
    path = 'C:/Turbo-SM/SMDATA/{0}_{1}_ref'.format(mouse,date)
    ref_frames = read_data(path)
    ref_average = np.mean(ref_frames, axis=0)
    ref_average = process_average(ref_average, lowpass=False)
    cv2.imwrite(os.path.join(path, 'ref_{0}_{1}.png'.format(mouse, date)), ref_average)
    pass

def process_single_odorant(mouse, date, odorant):

    path = 'C:/Turbo-SM/SMDATA/{0}_{1}_{2}'.format(mouse, date, odorant)
    #path = 'R:/Rinberglab/rinberglabspace/Edmund/intrinsic-imaging/{0}/{0}_{1}_{2}'.format(mouse, date, odorant)
    odor_averaged = []
    tsmcount = 0

    for tsm in os.listdir(path):
        if not tsm.endswith('tsm'):
            continue
        tsmcount += 1
        print 'processing file no. 1: {0}'.format(tsm)
        trialframes = read_data(os.path.join(path,tsm))
        odor_normed = compute_average(trialframes,t1=500, t2=len(trialframes))
        odor_final = process_average(odor_normed,row1=5)
        cv2.imwrite(os.path.join(path, '{0}_{1}_{2}_trial{3}.tif'.format(mouse, date, odorant, tsmcount)), odor_final)
        plt.imshow(odor_final)
        plt.show()
        odor_averaged.append(odor_normed)

    if len(odor_averaged) > 1:
        average_final = np.mean(odor_averaged, axis=0)
        average_final = process_average(average_final, row1=5)
        cv2.imwrite(os.path.join(path, '{0}_{1}_{2}_averaged.tif'.format(mouse, date, odorant)), average_final)
        plt.imshow(average_final)
        plt.show()
        pass
    else:
        pass

# def read_h5(mouse, date, session):
#     for h5 in [x for x in os.listdir(self.path) if x.endswith('h5')]:
#         odordict = h5py.File(os.path.join(self.path, h5), 'r')
#         for trialno, odor in enumerate(odordict['Trials']['odor']):
import h5py
import os
from collections import defaultdict as ddict
import pickle
from scipy.stats import binom
from scipy.optimize import minimize
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
import numpy as np

def weibull(x, p):

    t, b, a, g = p
    k = -log(((1.0 - a) / (1.0 - g)) ** (1.0 / b))

    y = 1.0 - (1.0 - g) * (e ** - (((k * x) / t) ** b))
    return y


def weibull_chris(x, p):
    """

    :param i: parameter value for the stimulus (ie intensity)
    :param alpha:
    :param beta:
    :param guess:
    :param lapse:
    :return:
    """

    alpha, beta, guess, lapse = p
    return ((1. - lapse) - (1. - guess - lapse) *
            np.exp(-(x / alpha) ** beta))

def find_fit(p, data):

    bounds=[(0.001,None),(0.001,None),(0.001,1),(0.001,1)]
    return minimize(logmaxlikelihood, p, args=data, method='L-BFGS-B', bounds=bounds)


def binom_logpmf(k, n, p):
    """

    :param x: number of successes
    :param n: number of trials shape parameter
    :param p: probability shape parameter
    :return:
    """
    # v1 = log(binom_coeff(n, x))
    # v1 = log(comb(n, x))
    # v2 = x * log(p)
    # v3 = (n - x) * log1p(-p)  # log(1+(-p))
    return binom.logpmf(k, n, p)


def logmaxlikelihood(p, args, f='weibull'):
    '''
    data structure:
    x: array of independent variables, i.e. stimuli
    k: array of number of successes for each x
    n: array of total number of trials for each x
    '''

    t, b, a, g = p
    x, k, n = args
    res = weibull_chris(x, p)
    return -np.sum(binom_logpmf(k, n, res))

def binP(N, p, x1, x2):
    p = float(p)
    q = p/(1-p)
    k = 0.0
    v = 1.0
    s = 0.0
    tot = 0.0

    while(k<=N):
            tot += v
            if(k >= x1 and k <= x2):
                    s += v
            if(tot > 10**30):
                    s = s/10**30
                    tot = tot/10**30
                    v = v/10**30
            k += 1
            v = v*q*(N+1-k)/k
    return s/tot

def binomial_CI(vx, vN, vCL = 95):
    '''
    Calculate the exact (Clopper-Pearson) confidence interval for a binomial proportion

    Usage:
    >>> calcBin(13,100)
    (0.07107391357421874, 0.21204372406005856)
    >>> calcBin(4,7)
    (0.18405151367187494, 0.9010086059570312)
    '''
    vx = float(vx)
    vN = float(vN)
    #Set the confidence bounds
    vTU = (100 - float(vCL))/2
    vTL = vTU

    vP = vx/vN
    if(vx==0):
            dl = 0.0
    else:
            v = vP/2
            vsL = 0
            vsH = vP
            p = vTL/100

            while((vsH-vsL) > 10**-5):
                    if(binP(vN, v, vx, vN) > p):
                            vsH = v
                            v = (vsL+v)/2
                    else:
                            vsL = v
                            v = (v+vsH)/2
            dl = v

    if(vx==vN):
            ul = 1.0
    else:
            v = (1+vP)/2
            vsL =vP
            vsH = 1
            p = vTU/100
            while((vsH-vsL) > 10**-5):
                    if(binP(vN, v, 0, vx) < p):
                            vsH = v
                            v = (vsL+v)/2
                    else:
                            vsL = v
                            v = (v+vsH)/2
            ul = v
    return (dl, ul)


def new_pfmcheck(pfm, sw, thresh):
    if len(pfm) > sw:
        pfmtrue = [p1 for p1 in pfm if not p1 == 'x']
        avrg = np.mean(pfmtrue[-sw:])
        if avrg < thresh:
            return False
        else:
            return True
    else:
        return True


def new_analyse(mouse, session, plot=False, sw=20, thresh=0.59, multiple_c=False):

    path = 'R:\\Rinberglab\\rinberglabspace\\Users\\Johannes\\taar_stim\\behavior'
    resdict = ddict(lambda: [0, 0, 0])
    pfm = []
    filepath = '{0}\\{1}\\{0}_{2}_{3}.h5'.format(
        mouse,
        session,
        session.split('-')[0],
        session.split('-')[1]
    )

    assert os.path.exists(os.path.join(path, filepath)), '{0}/{1} does not exist!'.format(path, filepath)

    h5file = h5py.File(os.path.join(path, filepath), 'r')
    results = h5file['Trials']['result']
    concs = h5file['Trials']['odorconc']
    dmd = h5file['Trials']['stim_desc']
    stimids =  h5file['Trials']['stimid']
    lpfm = []
    rpfm = []
    tpfm = []
    cdict = {
        '0.02': '0_100',
        '0.01': '5_95',
        '0.05': '25_75',
        '0.076': '38_62',
        '0.086': '43_57',
        '0.114': '57_43',
        '0.124': '62_38',
        '0.15': '75_25',
        '0.19': '95_5',
        '0.2': '100_0'
    }
    ldict = {
        '0.114': '57_43',
        '0.124': '62_38',
        '0.15': '75_25',
        '0.19': '95_5',
        '0.2': '100_0'
    }
    rdict = {
        '0.02': '0_100',
        '0.01': '5_95',
        '0.05': '25_75',
        '0.076': '38_62',
        '0.086': '43_57',
    }

    lstims = [l for l in ldict.keys()]
    rstims = [l for l in rdict.keys()]

    stimiddict = {
        'left' : [4, 9, 12, 15],
        'right' : [2, 5, 10, 15]
    }
    for rno, r in enumerate(results):
        if rno < sw:
            continue
        light = dmd[rno]
        conc = concs[rno]
        stimid = stimids[rno]
        if str(conc) in lstims:
            trialdir = 'left'
        elif str(conc) in rstims:
            trialdir = 'right'
        else:
            print 'Concentration not found in cdict:', str(conc)
            break

        if 'DMDoff' in light:
            if multiple_c:
                if int(stimid) in stimiddict[trialdir]:
                    stimtype = '{0}_{1}'.format(cdict[str(conc)], '1.5')
                else:
                    stimtype = cdict[str(conc)]
            else:
                stimtype = cdict[str(conc)]
        else:
            stimtype = '{0}_{1}'.format(cdict[str(conc)], light[:2])

        if r in [1, 4]:  # left
            if r == 4:
                r = 0
                lpfm.append(r)
                tpfm.append(r)
            else:
                lpfm.append(r)
                tpfm.append(r)
            rpfm.append('x')
            checkl = new_pfmcheck(lpfm, sw, thresh)
            checkr = new_pfmcheck(rpfm, sw, thresh)
            if checkl and checkr:
                resdict[stimtype][0] += 1
        elif r in [2, 3]:  # right
            if r == 3:
                r = 0
                rpfm.append(r)
                tpfm.append(r)
            else:
                r = 1
                rpfm.append(r)
                tpfm.append(r)
            lpfm.append('x')
            checkl = new_pfmcheck(lpfm, sw, thresh)
            checkr = new_pfmcheck(rpfm, sw, thresh)
            if checkl and checkr:
                resdict[stimtype][1] += 1
        else:  # no response
            if concs[rno] in lstims:
                lpfm.append(0)
                rpfm.append('x')
                tpfm.append(0)
            else:
                rpfm.append(0)
                lpfm.append('x')
                tpfm.append(0)
            check = new_pfmcheck(tpfm, sw, thresh)
            if check:
                resdict[stimtype][2] += 1
    if plot:
        plt.figure(figsize=(14, 8))
        for pfm in [lpfm, rpfm, tpfm]:
            pfm_sw = []
            for pno, p in enumerate(pfm):
                if pno > sw:
                    pfmtrue = [p1 for p1 in pfm[:pno + 1] if not p1 == 'x']
                    avrg = round(np.mean(pfmtrue[-sw:]), 2)
                elif pfm != []:
                    pfmtrue = [p1 for p1 in pfm[:pno + 1] if not p1 == 'x']
                    avrg = round(np.mean(pfmtrue[:]), 2)
                elif p == 2:
                    avrg = 0
                else:
                    avrg = p
                pfm_sw.append(avrg)

            plt.plot(pfm_sw)
            plt.ylim(0, 1)
            plt.legend(['left', 'right', 'total'])
        plt.show()
    return resdict


class Data:

    def __init__(self, name, sw, thresh, multiple_c=False):

        self.name = name
        self.multiple_c = multiple_c
        self.path = 'R:\\Rinberglab\\rinberglabspace\\Users\\Johannes\\taar_stim\\behavior'
        assert os.path.exists(self.path), 'Network drive not connected!'
        if not os.path.exists(os.path.join(self.path, 'mouse_data_{0}.p'.format(self.name))):
            self.mouse_data = {'8202': {}}
            with open(os.path.join(self.path, 'mouse_data_{0}.p'.format(self.name)), 'wb') as f:
                pickle.dump(self.mouse_data, f)
        else:
            with open(os.path.join(self.path, 'mouse_data_{0}.p'.format(self.name)), 'rb') as f:
                self.mouse_data = pickle.load(f)
        self.sw = sw
        self.thresh = thresh

    def add_data(self, sessions):

        for mouse in sessions.keys():

            try:
                mouse_dict = self.mouse_data[mouse]

            except:
                self.mouse_data[mouse] = {}
                mouse_dict = self.mouse_data[mouse]

            for session in sessions[mouse]:
                if isinstance(session, list):
                    session = session[0]
                if session in self.mouse_data[mouse].keys():
                    continue
                else:
                    resdict = new_analyse(mouse, session, sw=self.sw, thresh=self.thresh, multiple_c=self.multiple_c)
                    self.mouse_data[mouse][session] = dict(resdict)

        with open(os.path.join(self.path, 'mouse_data_{0}.p'.format(self.name)), 'wb') as f:
            pickle.dump(self.mouse_data, f)

    def plot_data(self, curve=True, probe=True, multiple_c=False, probe_curve=False, scatter=False):

        plots = []
        colors = {'odor + T3 stim': 'r', 'odor + T4 stim': 'g', 'odor': 'b', 'odor 1.5-fold': 'c'}
        for mouse, sessions in sorted(self.mouse_data.items()):
            fig, ax = plt.subplots(figsize=(10, 8))
            sdict = ddict(list)
            pdict = ddict(list)
            x = []
            curvedict = ddict(lambda: [[],[],[]])

            for session in sessions:
                for stim, result in self.mouse_data[mouse][session].items():
                    pdict[stim].append(result)
            totalt = 0
            for stim, result in sorted(pdict.items()):

                mix = stim.split('_')
                if len(mix) < 3:
                    mode = 'odor'
                elif mix[2] == '1.5':
                    mode = 'odor 1.5-fold'
                elif probe:
                    if 'T' in mix[2]:
                        mode = 'odor + {0} stim'.format(mix[2])
                    else:
                        continue
                else:
                    continue
                pea_c = round(float(stim.split('_')[0]) * 0.01, 2)
                if not multiple_c:
                    if mode == 'odor 1.5-fold':
                        continue
                performance = []
                totalr = 0
                silent = 0
                left = 0

                for [l, r, s] in result:

                    totalt += l + r + s
                    subtotal = l + r
                    totalr += subtotal
                    silent += s
                    left += l
                    if l > 0:
                        performance.append(float(l) / float(subtotal))
                    else:
                        performance.append(0.0)

                    if not mode == 'odor':
                        sf = float(s) / float(l + r + s)
                        sdict[stim].append(sf)
                if scatter:
                    scatter_i = ax.scatter([pea_c] * len(performance), [performance], color=colors[mode], alpha=.3)
                # plot mean
                mean_l = float(left) / float(totalr)
                scatter_m = ax.scatter([pea_c], [mean_l], color=colors[mode], marker='.', s=200, alpha=.9, label=mode)
                # calculate,plot binomial confidence intervals
                confint = binomial_CI(left, totalr)
                c = confint[1] - confint[0]
                c0 = confint[0] + c / 2
                errb = ax.errorbar([pea_c], [c0], yerr=c / 2, linestyle='None', color=colors[mode], alpha=.6)
                errb[-1][0].set_linestyle('--')
                if mode == 'odor':
                    x.append(float(pea_c))
                if not probe_curve:
                    if 'stim' in mode:
                        continue
                curvedict[mode][0].append(float(pea_c))
                curvedict[mode][1].append(float(totalr))
                curvedict[mode][2].append(float(left))
            # find curve fit
            if curve:
                p_chris = [.5, 3.5, .01, 0.0]
                for mode in curvedict.keys():
                    fdata = [
                        np.array(curvedict[mode][0]),
                        np.array(curvedict[mode][2]),
                        np.array(curvedict[mode][1])
                    ]
                    fit = find_fit(p_chris, fdata)

                    if fit.success:
                        x1 = np.arange(0., 1.01, 0.01)
                        p = fit.x
                        ax.plot(x1, weibull_chris(x1, p), alpha=.5, color=colors[mode])

            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='lower right')

            ax.set_title(mouse)
            ax.set_xlim(0, 1.0)
            x.append(0.0)
            x.append(1.0)
            ax.set_xticks([x0 for x0 in x])
            ax.set_yticks([y for y in np.arange(0.0, 1.1, 0.1)])
            ax.set_ylim(0, 1.05)
            ax.set_xlabel('% PEA in PEA/IPA', fontsize=12)
            ax.set_ylabel('% responds like PEA', fontsize=12)
            sns.despine()
            sns.despine(left=True)
            plt.savefig('MixtureAnalysis_{0}_{1}.pdf'.format(self.name, mouse), format='pdf', dpi=1200)
            plt.savefig('MixtureAnalysis_{0}_{1}.png'.format(self.name, mouse), dpi=300)
            plots.append(fig)
            plt.show()
            print 'Total # trials:', totalt
        return plots

    def plot_av(self, curve=True, probe=True, multiple_c=False, probe_curve=False, scatter=False):

        avdata = ddict(lambda: np.array([0, 0, 0]))
        for mouse in self.mouse_data.keys():

            for date in self.mouse_data[mouse].keys():

                for r in self.mouse_data[mouse][date].keys():
                    avdata[r] += np.array(self.mouse_data[mouse][date][r])
        colors = {'odor + T3 stim': 'r', 'odor + T4 stim': 'g', 'odor': 'b', 'odor 1.5-fold': 'c'}
        totalt = 0
        x = []
        fig, ax = plt.subplots(figsize=(10, 8))
        curvedict = ddict(lambda: [[], [], []])

        for stim, result in sorted(avdata.items()):

            mix = stim.split('_')
            if len(mix) < 3:
                mode = 'odor'
            elif mix[2] == '1.5':
                mode = 'odor 1.5-fold'
            elif probe:
                if 'T' in mix[2]:
                    mode = 'odor + {0} stim'.format(mix[2])
                else:
                    continue
            else:
                continue
            pea_c = round(float(stim.split('_')[0]) * 0.01, 2)

            performance = []
            [l, r, s] = result

            total = l + r + s
            totalt += l + r + s
            subtotal = l + r

            # plot mean
            mean_l = float(l) / float(subtotal)
            scatter_m = ax.scatter([pea_c], [mean_l], color=colors[mode], marker='.', s=200, alpha=.9, label=mode)
            # calculate,plot binomial confidence intervals
            confint = binomial_CI(l, total)
            c = confint[1] - confint[0]
            c0 = confint[0] + c / 2
            errb = ax.errorbar([pea_c], [c0], yerr=c / 2, linestyle='None', color=colors[mode], alpha=.6)
            errb[-1][0].set_linestyle('--')
            if mode == 'odor':
                x.append(float(pea_c))
            if not probe_curve:
                if 'stim' in mode:
                    continue

            curvedict[mode][0].append(float(pea_c))
            curvedict[mode][1].append(float(subtotal))
            curvedict[mode][2].append(float(l))
        # find curve fit

        p_chris = [.5, 3.5, .01, 0.0]
        for mode in curvedict.keys():
            fdata = [
                np.array(curvedict[mode][0]),
                np.array(curvedict[mode][2]),
                np.array(curvedict[mode][1])
            ]
            fit = find_fit(p_chris, fdata)

            if fit.success:
                x1 = np.arange(0., 1.01, 0.01)
                p = fit.x
                ax.plot(x1, weibull_chris(x1, p), alpha=.5, color=colors[mode])

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='lower right')

        ax.set_title('Average')
        ax.set_xlim(0, 1.0)
        x.append(0.0)
        x.append(1.0)
        ax.set_xticks([x0 for x0 in x])
        ax.set_yticks([y for y in np.arange(0.0, 1.1, 0.1)])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('% PEA in PEA/IPA', fontsize=12)
        ax.set_ylabel('% responds like PEA', fontsize=12)
        sns.despine()
        sns.despine(left=True)
        plt.savefig('MixtureAnalysis_{0}_{1}.pdf'.format(self.name, 'average'), format='pdf', dpi=1200)
        plt.savefig('MixtureAnalysis_{0}_{1}.png'.format(self.name, 'average'), dpi=300)
        plt.show()
        print 'Total # trials:', totalt

    def plot_accp(self, mice=[]):

        plots = []
        for mouse, sessions in sorted(self.mouse_data.items()):
            if not mouse in mice:
                continue
            fuckyoudict = ddict(list)
            for session in sorted(sessions):
                for stim, result in self.mouse_data[mouse][session].items():
                    fuckyoudict[stim].append(result)
            pdict = ddict(list)
            for stim, result in fuckyoudict.items():
                totalr = 0
                silent = 0
                left = 0
                for [l, r, s] in result:

                    mix = stim.split('_')
                    if len(mix) < 3:
                        mode = 'odor'
                    else:
                        mode = 'odor + {0} stim'.format(mix[2])
                    pea_c = round(float(stim.split('_')[0]) * 0.01, 2)

                    totalr += l + r
                    silent += s
                    left += l
                    if not mode == 'odor':
                        pdict[stim].append([float(l) / float(l + r), float(left) / float(totalr)])

            nrows, ncols = 2, 2
            fig = plt.figure(figsize=(12, 9))
            stims = sorted([s for s in pdict.keys()])
            for i in range(len(stims)):
                ax = fig.add_subplot(nrows, ncols, i + 1)
                ax.plot(range(len(sessions)), [x[0] for x in pdict[stims[i]]], alpha=.5)
                ax.plot(range(len(sessions)), [x[1] for x in pdict[stims[i]]])
                ax.set_xticklabels(sorted(sessions))
                ax.set_yticks([y for y in np.arange(0.0, 1.1, 0.1)])
                ax.set_ylim(0, 1.05)
                ax.set_title(stims[i])
            fig.tight_layout()
            plots.append(fig)
            plt.show()
        return plots

    def calc_stats(self):

        for mouse, sessions in sorted(self.mouse_data.items()):
            for session in sorted(sessions):
                total = 0
                for stim, result in self.mouse_data[mouse][session].items():
                    total += sum(result)
                print mouse, session, 'Total # trials:', total
        pass

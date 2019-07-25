import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import queue as q
import os
from os.path import isfile, join
import seaborn as sns
import matplotlib.pylab as pylab
import sys

sys.path.append('/Users/aabir/anaconda/envs/pca/simulations')

sns.set_style('whitegrid')

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10,10),
         'axes.labelsize': '24',
         'axes.titlesize':'24',
         'figure.titlesize': '24',
         'xtick.labelsize':'22',
         'ytick.labelsize':'22',
         'text.color':'k',
         'axes.formatter.useoffset':False,
         'grid.color' : 'k',
		 'grid.linestyle' : ':',
		 'grid.linewidth' : 0.5,
		 'legend.fancybox' : True,
		 'legend.framealpha' : 0.5,
         'axes.labelcolor' : 'k'}

pylab.rcParams.update(params)

from invmgmt.invsim import *
from invmgmt.simanalytics import *
from invmgmt.runsim import *

def MKBformatter(num, round=True):
    if any([int(np.abs(num))%(10**i) for i in range(len(str(int(np.abs(num)))))]) and not round:
        raise ValueError('Cannot round and num {} is not a power multiple of 10'.format(num))
    Kval = int(num/1000)
    if np.abs(Kval) <= 1:
        return num 
    elif np.abs(Kval) < 1000:
        return str(Kval)+'K'
    else: 
        Mval = int(num/10**6)
        if Mval < 1000:
            return str(Mval)+'M'
        else:
            Bval = int(num/10**9)
            return str(Bval)+'B'
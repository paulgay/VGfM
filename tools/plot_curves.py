import matplotlib.pyplot as plt
import pickle, sys, numpy as np, json, argparse, h5py, os
from prettytable import PrettyTable
import sys
sys.path.append('data_tools/')
import utils_prep

def reshape(rs):
    xs = sorted(rs['split_2'].keys())
    gtq = {}
    for i in range(len(xs)):
        it = xs[i]
        for (p,v) in rs['split_0'][it].items():
            if 'split_0'+p not in gtq:
                gtq['split_0'+p] = []
            gtq['split_0'+p].append(v)
        for (p,v) in rs['split_2'][it].items():
            if 'split_2'+p not in gtq:
                gtq['split_2'+p] = []
            gtq['split_2'+p].append(v)
    return (xs, gtq)



# specify which .res files and which colors
l = [ ('checkpoints/final2/final2.res','g'), ('checkpoints/final3/final3.res','r'), ('checkpoints/rel_object/rel_object.res','k'),('checkpoints/rel_object_fc/rel_object_fc.res','b')]


f = 1
for p in ['part_of', 'support', 'same_set', 'same_plan', 'in_front_of', 'left', 'below']:
    plt.figure(f)
    f = f + 1
    pls = []
    for (fil,col) in l: 
        rs = pickle.load(open(fil))
        (si_xs, si) = reshape(rs)
        pl, = plt.plot(si_xs,si['split_2'+p], col, label=os.path.basename(fil).replace('.res',''))
        pls.append(pl)
        plt.plot(si_xs,si['split_0'+p],col+'o')
    plt.title(p)
    plt.legend(pls, [ os.path.basename(fil).replace('.res','') for (fil, col) in l])
plt.show()





#import pdb; pdb.set_trace()

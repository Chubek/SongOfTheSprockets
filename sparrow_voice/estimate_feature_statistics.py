import os

from sprocket.model import GV, F0statistics
from sprocket.util import HDF5

from .misc import read_feats


def efs_main(speaker, list_file, pair_dir):
    # open h5 files
    h5_dir = os.path.join(pair_dir, 'h5')
    statspath = os.path.join(pair_dir, 'stats', speaker + '.h5')
    h5 = HDF5(statspath, mode='a')

    # estimate and save F0 statistics
    f0stats = F0statistics()
    f0s = read_feats(list_file, h5_dir, ext='f0')
    f0stats = f0stats.estimate(f0s)
    h5.save(f0stats, ext='f0stats')
    print("f0stats save into " + statspath)

    # estimate and save GV of orginal and target speakers
    gv = GV()
    mceps = read_feats(list_file, h5_dir, ext='mcep')
    gvstats = gv.estimate(mceps)
    h5.save(gvstats, ext='gv')
    print("gvstats save into " + statspath)

    h5.close()
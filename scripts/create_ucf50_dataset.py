"""Create HDF5 dataset for UCF-101.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import cPickle as pickle
import h5py
from spikefuel import dataset


def printname(name):
    """print name."""
    print name

# paths
db_name = "UCF50_30fps_20160409"
save_path = "/home/inilab/data"
ucf50_data_path = "/home/inilab/data/UCF50"
ucf50_path = "/home/inilab/data/ARCHIVE/UCF-50-ARCHIVE/ucf50_recordings_30fps"
ucf50_stats_path = "./data/ucf50_stats.pkl"

# reading dataset statistaic
f = file(ucf50_stats_path, mode="r")
ucf50_stats = pickle.load(f)
f.close()

# Produce bounding boxes

ucf50_list = ucf50_stats["ucf50_list"]

# inite dataset
dataset.create_ucf50_db(db_name, save_path, ucf50_path, ucf50_stats,
                        ucf50_data_path)

db = h5py.File(os.path.join(save_path, db_name+".hdf5"), mode="r")
db.visit(printname)

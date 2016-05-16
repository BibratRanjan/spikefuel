"""Create HDF5 dataset for Caltech-256.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import cPickle as pickle
import h5py
from spikefuel import dataset, dvsproc


def printname(name):
    """print name."""
    print name

# paths
data_path = os.environ["SPIKEFUEL_DATA"]
db_name = "Caltech256_10fps_20160411.hdf5"
save_path = "./data"
caltech256_rec = "INI_caltech256_recordings_10fps_20160424"
caltech256_path = os.path.join(data_path, caltech256_rec)
caltech256_stats_path = os.path.join(data_path, "sf_data",
                                     "caltech256_stats.pkl")
db_path = os.path.join(data_path, db_name)

f = file(caltech256_stats_path, mode="r")
caltech256_stats = pickle.load(f)
f.close()
caltech256_list = caltech256_stats["caltech256_list"]

option = "fix-error-recordings"

if option == "gen-full-dataset":
    # reading dataset statistaic
    f = file(caltech256_stats_path, mode="r")
    caltech256_stats = pickle.load(f)
    f.close()

    # Produce bounding boxes

    caltech256_list = caltech256_stats["caltech256_list"]

    # inite dataset
    dataset.create_caltech256_db(db_name, save_path, caltech256_path,
                                 caltech256_stats)

    db = h5py.File(os.path.join(save_path, db_name+".hdf5"), mode="r")
    db.visit(printname)

if option == "fix-error-recordings":
    database = h5py.File(db_path, mode="a")
    class_name = caltech256_list[62]
    img_name = caltech256_stats[class_name][63 - 1]

    img_n, img_ex = os.path.splitext(img_name)

    # find number of frames
    num_frames = 10
    metadata = {"num_frames": num_frames}
    record_path = os.path.join(caltech256_path, class_name,
                               img_n+".aedat")
    print "[MESSAGE] Loading %s" % (record_path)
    record_data = dvsproc.loadaerdat(record_path)
    print record_data[0]
    dataset.save_recording(img_n, record_data, database,
                           group_path=class_name,
                           metadata=metadata, renew=True)
    print "[MESSAGE] Sequence %s is saved" % img_n

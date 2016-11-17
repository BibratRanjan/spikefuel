"""Create necessary statistics for certain dataset.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import glob
import cPickle as pickle
import h5py
import numpy as np

from spikefuel import dvsproc, helpers


def lipreading_stats(save_path, directory=".",
                     filename="lipreading_stats.pkl"):
    """Generate statistic for Lipreading datasets.

    Parameters
    ----------
    save_path : string
        Lipreading dataset save path
    directory : string
        directory you want to save the stats, in default,
        it's in current folder
    filename : string
        file name you want to save the stats, in default,
        it's named "lipreading_stats.pkl"

    Returns
    -------
    A file named with respected filename
    """
    if not os.path.isdir(save_path):
        raise ValueError("Either path is not existed or is a relative path")

    LIPREADING_STATS = {}
    LIPREADING_LIST = next(os.walk(save_path))[1]
    LIPREADING_STATS["lipreading_list"] = LIPREADING_LIST

    for list_name in LIPREADING_LIST:
        fns_temp = glob.glob(os.path.join(save_path, list_name)+"/"+"*.mpg")
        file_names = []
        for fn in fns_temp:
            file_names.append(os.path.basename(fn))
        LIPREADING_STATS[list_name] = file_names

    F = open(os.path.join(directory, filename), mode="wb")
    pickle.dump(LIPREADING_STATS, F, protocol=pickle.HIGHEST_PROTOCOL)
    F.close()


def caltech256_stats(save_path, directory=".",
                     filename="caltech256_stats.pkl"):
    """Generate statistic for Caltech-256 dataset.

    Parameters
    ---------
    save_path : string
        Caltech-256 dataset save path
    directory : string
        directory you want to save the stats, in default,
        it's in current folder
    filename : string
        filename you want to save the stats, in default,
        it's named "ucf50_stats.pkl"

    Returns
    -------
    A file named with respective filename
    """
    if not os.path.isdir(save_path):
        raise ValueError("Either path is not existed or is a relative path")

    caltech256_stats = {}
    caltech256_list = next(os.walk(save_path))[1]
    caltech256_stats["caltech256_list"] = caltech256_list

    for list_name in caltech256_list:
        fns_temp = glob.glob(os.path.join(save_path, list_name)+"/"+"*.jpg")
        file_names = []
        for fn in fns_temp:
            file_names.append(os.path.basename(fn))
        caltech256_stats[list_name] = file_names

    f = open(os.path.join(directory, filename), mode="wb")
    pickle.dump(caltech256_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def ucf50_stats(save_path, directory=".", filename="ucf50_stats.pkl"):
    """Generate statistaic for UCF-50 dataset.

    Parameters
    ---------
    save_path : string
        UCF-50 dataset save path
    directory : string
        directory you want to save the stats, in default,
        it's in current folder
    filename : string
        filename you want to save the stats, in default,
        it's named "ucf50_stats.pkl"

    Returns
    -------
    A file named with respective filename
    """
    ucf50_stats = {}
    ucf50_list = ["BaseballPitch", "Basketball", "BenchPress", "Biking",
                  "Billiards", "BreastStroke", "CleanAndJerk", "Diving",
                  "Drumming", "Fencing", "GolfSwing", "HighJump", "HorseRace",
                  "HorseRiding", "HulaHoop", "JavelinThrow", "JugglingBalls",
                  "JumpRope", "JumpingJack", "Kayaking", "Lunges",
                  "MilitaryParade", "Mixing", "Nunchucks", "PizzaTossing",
                  "PlayingGuitar", "PlayingPiano", "PlayingTabla",
                  "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
                  "Punch", "PushUps", "RockClimbingIndoor", "RopeClimbing",
                  "Rowing", "SalsaSpin", "SkateBoarding", "Skiing", "Skijet",
                  "SoccerJuggling", "Swing", "TaiChi", "TennisSwing",
                  "ThrowDiscus", "TrampolineJumping", "VolleyballSpiking",
                  "WalkingWithDog", "YoYo"]

    ucf50_stats["ucf50_list"] = ucf50_list
    for list_name in ucf50_list:
        fns_temp = glob.glob(os.path.join(save_path, list_name)+"/"+"*.avi")
        file_names = []
        for fn in fns_temp:
            file_names.append(os.path.basename(fn))
        ucf50_stats[list_name] = file_names

    f = open(os.path.join(directory, filename), mode="wb")
    pickle.dump(ucf50_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def ucf101_stats(save_path, directory=".", filename="ucf101_stats.pkl"):
    """Generate statistaic for UCF-101 dataset.

    Parameters
    ---------
    save_path : string
        UCF-101 dataset save path
    directory : string
        directory you want to save the stats, in default,
        it's in current folder
    filename : string
        filename you want to save the stats, in default,
        it's named "ucf101_stats.pkl"

    Returns
    -------
    A file named with respective filename
    """
    ucf101_stats = {}
    ucf101_list = ["ApplyEyeMakeup", "ApplyLipstick", "Archery",
                   "BabyCrawling", "BalanceBeam", "BandMarching",
                   "BaseballPitch", "Basketball", "BasketballDunk",
                   "BenchPress", "Biking", "Billiards", "BlowDryHair",
                   "BlowingCandles", "BodyWeightSquats", "Bowling",
                   "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke",
                   "BrushingTeeth", "CleanAndJerk", "CliffDiving",
                   "CricketBowling", "CricketShot", "CuttingInKitchen",
                   "Diving", "Drumming", "Fencing", "FieldHockeyPenalty",
                   "FloorGymnastics", "FrisbeeCatch", "FrontCrawl",
                   "GolfSwing", "Haircut", "HammerThrow", "Hammering",
                   "HandstandPushups", "HandstandWalking", "HeadMassage",
                   "HighJump", "HorseRace", "HorseRiding", "HulaHoop",
                   "IceDancing", "JavelinThrow", "JugglingBalls", "JumpRope",
                   "JumpingJack", "Kayaking", "Knitting", "LongJump", "Lunges",
                   "MilitaryParade", "Mixing", "MoppingFloor", "Nunchucks",
                   "ParallelBars", "PizzaTossing", "PlayingCello",
                   "PlayingDaf", "PlayingDhol", "PlayingFlute",
                   "PlayingGuitar", "PlayingPiano", "PlayingSitar",
                   "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse",
                   "PullUps", "Punch", "PushUps", "Rafting",
                   "RockClimbingIndoor", "RopeClimbing", "Rowing", "SalsaSpin",
                   "ShavingBeard", "Shotput", "SkateBoarding", "Skiing",
                   "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
                   "StillRings", "SumoWrestling", "Surfing", "Swing",
                   "TableTennisShot", "TaiChi", "TennisSwing", "ThrowDiscus",
                   "TrampolineJumping", "Typing", "UnevenBars",
                   "VolleyballSpiking", "WalkingWithDog", "WallPushups",
                   "WritingOnBoard", "YoYo"]

    ucf101_stats["ucf101_list"] = ucf101_list
    for list_name in ucf101_list:
        fns_temp = glob.glob(os.path.join(save_path, list_name)+"/"+"*.avi")
        file_names = []
        for fn in fns_temp:
            file_names.append(os.path.basename(fn))
        ucf101_stats[list_name] = file_names

    f = open(os.path.join(directory, filename), mode="wb")
    pickle.dump(ucf101_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def tracking_stats(save_path, directory=".", filename="tracking_stats.pkl"):
    """Generate statistics for Tracking Dataset.

    You can obtain the dataset from here:
    http://cmp.felk.cvut.cz/~vojirtom/dataset/

    Parameters
    ----------
    save_path : string
        Tracking dataset save path
    directory : string
        directory you want to save the stats, in default,
        it's in current folder
    filename : string
        filename you want to save the stats, in default,
        it's named "tracking_stats.pkl"

    Returns
    -------
    A file named with respective filename
    """
    tracking_stats = {}
    tracking_primary_list = ["Babenko", "BoBot", "Cehovin", "Ellis_ijcv2011",
                             "Godec", "Kalal", "Kwon", "Kwon_VTD", "Other",
                             "PROST", "Ross", "Thang", "Wang"]
    tracking_stats["primary_list"] = tracking_primary_list
    secondary_list = {}
    for primary_class in tracking_primary_list:
        pc_class = os.path.join(save_path, primary_class)
        secondary_class = next(os.walk(pc_class))[1]
        secondary_list[primary_class] = secondary_class

        for category in secondary_class:
            file_path = os.path.join(save_path, primary_class, category, "*.*")
            fns_temp = glob.glob(file_path)
            file_names = []
            for fn in fns_temp:
                if fn[-4:] != ".txt" and fn[-3:] != ".db":
                    file_names.append(os.path.basename(fn))
            tracking_stats[category] = file_names

    tracking_stats["secondary_list"] = secondary_list

    f = open(os.path.join(directory, filename), mode="wb")
    pickle.dump(tracking_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def vot_stats(listfile, directory=".", filename="vot_stats.pkl"):
    """Generate statistics for VOT challenge dataset.

    Caution: number of frames for each sequence is hand-coded for VOT2015
    dataset only.

    Parameters
    ----------
    listfile : String
        path of list of sequence names.
    directory : String
        directory you want to save the stats, in default,
        it's in current folder
    filename : String
        filename you want to save the stats, in default,
        it's named "vot_stats.pkl".
        I would encourage you to use .pkl as extension.

    Returns
    -------
    A file named with respective filename
    """
    vot_stats = {}
    vot_list = np.loadtxt(listfile, dtype="string")
    num_frames = [196, 105, 41, 725, 339, 539, 225, 76, 350, 293, 175, 151,
                  742, 393, 131, 326, 292, 366, 310, 519, 682, 1500, 120, 366,
                  844, 567, 240, 118, 465, 267, 377, 402, 708, 661, 707, 63,
                  201, 100, 164, 61, 999, 291, 140, 713, 158, 156, 558, 365,
                  251, 351, 366, 131, 392, 129, 138, 201, 365, 191, 312, 341]

    vot_stats['vot_list'] = vot_list
    vot_stats['num_frames'] = num_frames

    f = open(os.path.join(directory, filename), mode="wb")
    pickle.dump(vot_stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def init_database(db_name, save_path):
    """Initialize database for a given dataset.

    Parameters
    ----------
    db_name : string
        Database name
    save_path : string
        the destination of the database

    Returns
    -------
    database : h5py.File
        a HDF5 file object
    """
    # append extension if needed
    if db_name[-5:] != ".hdf5" and db_name[-3:] != ".h5":
        db_name += ".hdf5"

    # create destination folder if needed
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    db_name = os.path.join(save_path, db_name)
    database = h5py.File(db_name, "a")

    return database


def save_recording(record_name, record_data,
                   database, group=None, group_path=None,
                   bounding_box=None, metadata=None, renew=False):
    """Save a given recording to a given group.

    Parameters
    ----------
    record_name : string
        the name of the recording
    record_data : tuple
        a tuple with 4 elements that contains recording data
    database : h5py.File
        a HDF5 file object.
    group : h5py.Group
        a HDF5 group object.
    group_path : string
        The path to the given group from root group.
    bounding_box : numpy.ndarray
        Bounding box information for tracking or detection dataset
    metadata : dictionary
        the metadata that is associated with the recording
    renew : bool
        indicate if the data should be renewed

    Returns
    -------
    A recording saved in the given group
    """
    # Direct to the given group if no Group object presented
    if group is None:
        if group_path is None:
            raise ValueError("Group path must not be None!")
        group = database[group_path]

    if record_name not in group or renew is True:
        if record_name not in group:
            gp = group.create_group(record_name)
        else:
            gp = group[record_name]

        if renew is True:
            del gp["timestamps"]
            del gp["x_pos"]
            del gp["y_pos"]
            del gp["pol"]
        gp.create_dataset("timestamps", data=record_data[0].astype(np.int32),
                          dtype=np.int32)
        gp.create_dataset("x_pos", data=record_data[1].astype(np.uint8),
                          dtype=np.uint8)
        gp.create_dataset("y_pos", data=record_data[2].astype(np.uint8),
                          dtype=np.uint8)
        gp.create_dataset("pol", data=record_data[3].astype(np.bool),
                          dtype=np.bool)

        # record bounding boxes if available
        # the format may vary from dataset to dataset
        if bounding_box is not None:
            gp.create_dataset("bounding_box",
                              data=bounding_box.astype(np.float32),
                              dtype=np.float32)
        # Set metadata for the recording
        if metadata is not None:
            for key, value in metadata.iteritems():
                gp.attrs[key] = value

        database.flush()


def gen_tracking_db(database, tracking_stats):
    """Generate TrackingDataset structure.

    Parameters
    ----------
    database : h5py.File
        HDF5 file object
    tracking_stats : dictionary
        the dictionary that contains TrackingDataset's stats

    Returns
    -------
    database : h5py.File
        HDF5 file object with multiple groups
    """
    primary_list = tracking_stats["primary_list"]

    for pc in primary_list:
        if pc not in database:
            database.create_group(pc)
            print "[MESSAGE] Primary group %s is created" % (pc)

    print "[MESSAGE] TrackingDataset HDF5 structure is generated."


def create_tracking_db(db_name, save_path, tracking_path, tracking_stats,
                       bounding_box=None, device="DAViS240C",
                       fps=30, monitor_id="SAMSUNG SyncMaster 2343BW",
                       monitor_feq=60):
    """Create HDF5 database for TrackingDataset.

    Main function wrappers for TrackingDataset.

    Parameters
    ----------
    db_name : string
        Database name
    save_path : string
        the destination of the database
    tracking_path : string
        the destination of the tracking record
    tracking_stats : dictionary
        the dictionary that contains TrackingDataset's stats
    bounding_box : dictionary
        collection of bounding boxes for each sequence
    device : string
        DVS camera model, in default, it's DAViS240C
    fps : int
        Internal refreshing rate set by program, in default, it's 30Hz
        This factor will approximately determine how long the frame is gonna
        display
    monitor_id : string
        The model name of the monitor used, as default,
        it's SAMSUNG SyncMaster 2343BW
    monitor_feq : int
        Monitor display frequency, 60Hz as default

    return
    ------
    database : h5py.File
        a HDF5 file object saved in the given destination
    """
    # tracking statistics
    primary_list = tracking_stats["primary_list"]
    secondary_list = tracking_stats["secondary_list"]

    database = init_database(db_name, save_path)
    gen_tracking_db(database, tracking_stats)

    # Set dataset metadata
    database.attrs["device"] = device
    database.attrs["fps"] = fps
    database.attrs["monitor_id"] = monitor_id
    database.attrs["monitor_feq"] = monitor_feq

    for pc in primary_list:
        # I didn't record primary group Kalal
        # FIXME: fix this if I recorded primary group Kalal
        if pc != "Kalal":
            for sc in secondary_list[pc]:
                if sc not in database[pc]:
                    metadata = {"num_frames": len(tracking_stats[sc])}
                    record_path = os.path.join(tracking_path, sc+".aedat")
                    print "[MESSAGE] Loading %s" % (record_path)
                    record_data = dvsproc.loadaerdat(record_path)
                    save_recording(sc, record_data, database,
                                   group_path=pc,
                                   metadata=metadata,
                                   bounding_box=bounding_box[sc])
                    print "[MESSAGE] Sequence %s is saved" % sc

    database.flush()
    database.close()
    print "[MESSAGE] TrackingDataset is saved to %s" % (save_path)


def gen_caltech256_db(database, caltech256_stats):
    """Generate Caltech-256 structure.

    Parameters
    ----------
    database : h5py.File
        HDF5 file object
    caltech256_stats : dictionary
        the dictionary that contains Caltech-256's stats

    Returns
    -------
    database : h5py.File
        HDF5 file object with multiple groups
    """
    caltech256_list = caltech256_stats["caltech256_list"]

    for class_name in caltech256_list:
        if class_name not in database:
            database.create_group(class_name)
            print "[MESSAGE] Class %s is created" % (class_name)

    print "[MESSAGE] Caltech-256 HDF5 structure is generated."


def create_caltech256_db(db_name, save_path, caltech256_path, caltech256_stats,
                         device="DAViS240C", fps=10,
                         monitor_id="SAMSUNG SyncMaster 2343BW",
                         monitor_feq=60):
    """Create HDF5 database for Caltech-256.

    Main function wrappers for Caltech-256.

    Parameters
    ----------
    db_name : string
        Database name
    save_path : string
        the destination of the database
    caltech256_path : string
        the destination of the Caltech-256 record
    caltech256_stats : dictionary
        the dictionary that contains Caltech-256's stats
    caltech256_data_path : string
        destination of Caltech-256 dataset
    device : string
        DVS camera model, in default, it's DAViS240C
    fps : int
        Internal refreshing rate set by program, in default, it's 30Hz
        This factor will approximately determine how long the frame is gonna
        display
    monitor_id : string
        The model name of the monitor used, as default,
        it's SAMSUNG SyncMaster 2343BW
    monitor_feq : int
        Monitor display frequency, 60Hz as default

    return
    ------
    database : h5py.File
        a HDF5 file object saved in the given destination
    """
    # tracking statistics
    caltech256_list = caltech256_stats["caltech256_list"]

    database = init_database(db_name, save_path)
    gen_caltech256_db(database, caltech256_stats)

    # Set dataset metadata
    database.attrs["device"] = device
    database.attrs["fps"] = fps
    database.attrs["monitor_id"] = monitor_id
    database.attrs["monitor_feq"] = monitor_feq

    for class_name in caltech256_list:
        for img_name in caltech256_stats[class_name]:
            img_n, img_ex = os.path.splitext(img_name)
            if img_n not in database[class_name]:
                # find number of frames
                num_frames = fps
                metadata = {"num_frames": num_frames}
                record_path = os.path.join(caltech256_path, class_name,
                                           img_n+".aedat")
                print "[MESSAGE] Loading %s" % (record_path)
                record_data = dvsproc.loadaerdat(record_path)
                save_recording(img_n, record_data, database,
                               group_path=class_name,
                               metadata=metadata)
                print "[MESSAGE] Sequence %s is saved" % img_n

    database.flush()
    database.close()
    print "[MESSAGE] Caltech-256 is saved to %s" % (save_path)


def gen_ucf50_db(database, ucf50_stats):
    """Generate UCF50 structure.

    Parameters
    ----------
    database : h5py.File
        HDF5 file object
    ucf50_stats : dictionary
        the dictionary that contains UCF50's stats

    Returns
    -------
    database : h5py.File
        HDF5 file object with multiple groups
    """
    ucf50_list = ucf50_stats["ucf50_list"]

    for category in ucf50_list:
        if category not in database:
            database.create_group(category)
            print "[MESSAGE] Category %s is created" % (category)

    print "[MESSAGE] UCF-50 HDF5 structure is generated."


def create_ucf50_db(db_name, save_path, ucf50_path, ucf50_stats,
                    ucf50_data_path, device="DAViS240C", fps=30,
                    monitor_id="SAMSUNG SyncMaster 2343BW", monitor_feq=60):
    """Create HDF5 database for UCF-50.

    Main function wrappers for UCF-50.

    Parameters
    ----------
    db_name : string
        Database name
    save_path : string
        the destination of the database
    ucf50_path : string
        the destination of the ucf-50 record
    ucf50_stats : dictionary
        the dictionary that contains UCF-50's stats
    ucf50_data_path : string
        destination of UCF-50 dataset
    device : string
        DVS camera model, in default, it's DAViS240C
    fps : int
        Internal refreshing rate set by program, in default, it's 30Hz
        This factor will approximately determine how long the frame is gonna
        display
    monitor_id : string
        The model name of the monitor used, as default,
        it's SAMSUNG SyncMaster 2343BW
    monitor_feq : int
        Monitor display frequency, 60Hz as default

    return
    ------
    database : h5py.File
        a HDF5 file object saved in the given destination
    """
    # tracking statistics
    ucf50_list = ucf50_stats["ucf50_list"]

    database = init_database(db_name, save_path)
    gen_ucf50_db(database, ucf50_stats)

    # Set dataset metadata
    database.attrs["device"] = device
    database.attrs["fps"] = fps
    database.attrs["monitor_id"] = monitor_id
    database.attrs["monitor_feq"] = monitor_feq

    for category in ucf50_list:
        for vidn in ucf50_stats[category]:
            vid_n, vid_ex = os.path.splitext(vidn)
            if vid_n not in database[category]:
                # find number of frames
                vid_path = os.path.join(ucf50_data_path, category, vidn)
                num_frames = helpers.count_video_frames(vid_path)
                metadata = {"num_frames": num_frames}
                record_path = os.path.join(ucf50_path, category,
                                           vid_n+".aedat")
                print "[MESSAGE] Loading %s" % (record_path)
                record_data = dvsproc.loadaerdat(record_path)
                save_recording(vid_n, record_data, database,
                               group_path=category,
                               metadata=metadata)
                print "[MESSAGE] Sequence %s is saved" % vid_n

    database.flush()
    database.close()
    print "[MESSAGE] UCF-50 is saved to %s" % (save_path)


def create_vot_db(db_name, save_path, vot_path, vot_stats,
                  vot_data_path, bounding_box=None, device="DAViS240C", fps=30,
                  monitor_id="SAMSUNG SyncMaster 2343BW", monitor_feq=60):
    """Create HDF5 database for VOT Challenge Dataset.

    Main function wrappers for VOT Challenge Dataset.

    Parameters
    ----------
    db_name : string
        Database name
    save_path : string
        the destination of the database
    vot_path : string
        the destination of the vot record
    vot_stats : dictionary
        the dictionary that contains VOT's stats
    vot_data_path : string
        destination of VOT dataset
    bounding_box : dictionary
        collection of bounding boxes for each sequence
    device : string
        DVS camera model, in default, it's DAViS240C
    fps : int
        Internal refreshing rate set by program, in default, it's 30Hz
        This factor will approximately determine how long the frame is gonna
        display
    monitor_id : string
        The model name of the monitor used, as default,
        it's SAMSUNG SyncMaster 2343BW
    monitor_feq : int
        Monitor display frequency, 60Hz as default

    return
    ------
    database : h5py.File
        a HDF5 file object saved in the given destination
    """
    # tracking statistics
    vot_list = vot_stats["vot_list"]
    num_frames = vot_stats['num_frames']
    num_seq = len(vot_list)

    database = init_database(db_name, save_path)

    # Set dataset metadata
    database.attrs["device"] = device
    database.attrs["fps"] = fps
    database.attrs["monitor_id"] = monitor_id
    database.attrs["monitor_feq"] = monitor_feq

    for i in xrange(num_seq):
        if vot_list[i] not in database:
            metadata = {"num_frames": num_frames[i]}
            record_path = os.path.join(vot_path, vot_list[i]+".aedat")
            print "[MESSAGE] Loading %s" % (record_path)
            record_data = dvsproc.loadaerdat(record_path)
            save_recording(vot_list[i], record_data, database, group=database,
                           bounding_box=bounding_box[vot_list[i]],
                           metadata=metadata)
            print "[MESSAGE] Sequence %s is saved" % vot_list[i]

    database.flush()
    database.close()
    print "[MESSAGE] VOT Dataset is saved to %s" % (save_path)

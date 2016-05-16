"""Error test for UCF-50 and Caltech-256.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import cPickle as pickle
from spikefuel import dvsproc

# UCF50 paths
ucf50_path = "/home/inilab/data/ARCHIVE/UCF-50-ARCHIVE/ucf50_recordings_30fps"
ucf50_stats_path = "./data/ucf50_stats.pkl"

if not os.path.isfile("/home/inilab/data/mis_ucf50_recordings.pkl"):
    # reading dataset statistaic
    f = file(ucf50_stats_path, mode="r")
    ucf50_stats = pickle.load(f)
    f.close()

    # Produce bounding boxes

    ucf50_list = ucf50_stats["ucf50_list"]

    ucf_mis_vid = []
    ucf50_num_vid = 0
    mis_ucf50_num_vid = 0
    for class_name in ucf50_list:
        for video_name in ucf50_stats[class_name]:
            ucf50_num_vid += 1
            vid_path = os.path.join(ucf50_path, class_name,
                                    video_name[:-4]+".aedat")

            _, flag = dvsproc.check_aedat(vid_path)
            if flag is False:
                print "Recording "+video_name[:-4]+" is wrong."
                mis_ucf50_num_vid += 1
                ucf_mis_vid.append(class_name+"/"+video_name[:-4])

    print "------------------------------------------------------"
    print "Number of Wrong Recordings: %d" % (mis_ucf50_num_vid)
    print "Total Number of Recordings: %d" % (ucf50_num_vid)
    percentage = mis_ucf50_num_vid/float(ucf50_num_vid)
    print "Percentage of Wrong Recordings: %f" % (percentage)
    print ucf_mis_vid
    f = file("/home/inilab/data/mis_ucf50_recordings.pkl", mode="wb")
    pickle.dump(ucf_mis_vid, f)
    f.close()
    print "------------------------------------------------------"

caltech_path = "/home/inilab/data/caltech256_recordings"
caltech_stats_path = "./data/caltech256_stats.pkl"

if not os.path.isfile("/home/inilab/data/mis_caltech_recordings.pkl"):
    f = file(caltech_stats_path, mode="r")
    caltech256_stats = pickle.load(f)
    f.close()

    caltech256_list = caltech256_stats["caltech256_list"]

    caltech_mis_vid = []
    caltech_num_vid = 0
    mis_caltech_num_vid = 0
    for class_name in caltech256_list:
        for img_name in caltech256_stats[class_name]:
            caltech_num_vid += 1
            print "CHECKING "+class_name+" "+img_name[:-4]
            img_path = os.path.join(caltech_path, class_name,
                                    img_name[:-4]+".aedat")
            if os.path.isfile(img_path):
                _, flag = dvsproc.check_aedat(img_path)
                if flag is False:
                    print "Recording "+class_name+" "+img_name[:-4]+" is wrong"
                    mis_caltech_num_vid += 1
                    caltech_mis_vid.append(class_name+"/"+img_name[:-4])

    print "------------------------------------------------------"
    print "Number of Wrong Recordings: %d" % (mis_caltech_num_vid)
    print "Total Number of Recordings: %d" % (caltech_num_vid)
    percentage = mis_caltech_num_vid/float(caltech_num_vid)
    print "Percentage of Wrong Recordings: %f" % (percentage)
    print caltech_mis_vid
    f = file("/home/inilab/data/mis_caltech_recordings.pkl", mode="wb")
    pickle.dump(caltech_mis_vid, f)
    f.close()
    print "------------------------------------------------------"

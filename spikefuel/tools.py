"""Tools to simplify workflow.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import cv2
import time
import socket

from spikefuel import dvsproc

# PUBLIC PARAMETERS
UDP_PORT = 8997
REMOTE_IP = "localhost"
BUFSIZE = 1024


def create_vot_image_list(save_path, num_frames):
    """Create a image list for given VOT video sequence.

    Parameters
    ----------
    save_path : String
        the directory of the given VOT video sequence
    num_frames : int
        number of frames in the given video sequence

    Returns
    -------
    image_list : list
        One ordered list that contains path of frames
    """
    image_list = []
    for i in xrange(1, num_frames+1):
        images_path = os.path.join(save_path, "%08d" % (i,) + ".jpg")
        image_list.append(images_path)

    return image_list


def create_image_sequence(frames, save_path, title, form=".png"):
    """Create a image sequence by given frames.

    Parameters
    ----------
    frames : list
        An ordered list that contains all frames. Frames are either in
        OpenCV Mat or numpy array
    save_path : string
        directory of where you want to write the image sequence
    title : string
        title of the image
    form : string
        format of output images, in default as PNG image

    Returns
    -------
    A sequence of images written in the given directory and title.
    """
    image_base = os.path.join(save_path, title+"-")

    num_frames = len(frames)

    for i in xrange(num_frames):
        image_add = os.path.join(image_base, "%08d" % (i+1,)+form)
        print "[MESSAGE] Writing "+image_add
        cv2.imwrite(image_add, frames[i])

    print "[MESSAGE] Images are written"


def init_dvs():
    """Initialize a socket that can send commands to jAER.

    Returns
    -------
    s : socket
        An empty socket that can connect jAER
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('', 0))
    print "[MESSAGE] The socket is established."

    return s


def destroy_dvs(s):
    """Destroy dvs socket.

    Returns
    -------
    A flag that indicates if the socket is close successfully.
    """
    s.close()
    print "[MESSAGE] The socket is closed."


def reset_dvs_time(conn, viewer_id=2, wait=0.3):
    """Send a command to reset time scale across all viewers.

    FIXME: the reset print a message that is not supposed to be there.

    Parameters
    ----------
    conn : socket
        An socket that can connect jAER
    viewer_id : int
        the n-th viewer you've created, for Mac, default is 2
    wait : float
        positive float for wait jAER viewers to reset time stamp
    """
    addr = REMOTE_IP, UDP_PORT+viewer_id-1
    line = 'zerotimestamps'
    conn.sendto(line, addr)
    time.sleep(wait)


def start_log_dvs(conn, save_path, title, viewer_id=2):
    """Send start logging command to jAER.

    Parameters
    ----------
    conn : socket
        An socket that can connect jAER
    save_path : string
        the directory you want to put the rocording
        use absolute path since the file saving is with jAER
    title : string
        the title of your recording, no extension
    viewer_id : int
        the n-th viewer you've created, for Mac, default is 2

    Returns
    -------
    rec_path : string
        absolute path to saved recording

    flag that indicates if the function is sent successfully
    """
    addr = REMOTE_IP, UDP_PORT+viewer_id-1
    rec_path = os.path.join(save_path, title)

    # send start logging command
    reset_dvs_time(conn, viewer_id)
    line = 'startlogging '+rec_path
    conn.sendto(line, addr)
    data, fromaddr = conn.recvfrom(BUFSIZE)
    print ('[MESSAGE] client received %r from %r' % (data, fromaddr))

    return rec_path


def stop_log_dvs(conn, viewer_id=2):
    """Send stop logging command to jAER.

    Parameters
    ----------
    conn : socket
        An socket that can connect jAER
    viewer_id : int
        the n-th viewer you've created, for Mac, default is 2

    Returns
    -------
    flag that indicates if the function is sent successfully
    """
    addr = REMOTE_IP, UDP_PORT+viewer_id-1

    # Send stop logging command
    line = 'stoplogging'
    conn.sendto(line, addr)
    data, fromaddr = conn.recvfrom(BUFSIZE)
    print ('[MESSAGE] client received %r from %r' % (data, fromaddr))


def log_dvs(conn, save_path, title, duration, viewer_id=2):
    """Log a DVS recording for certain duration.

    Parameters
    ----------
    conn : socket
        An socket that can connect jAER
    save_path : string
        the directory you want to put the rocording
        use absolute path since the file saving is with jAER
    title : string
        the title of your recording, no extension
    duration : float
        duration of the total recording in seconds
    viewer_id : int
        the n-th viewer you've created, for Mac, default is 2

    Returns
    -------
    A flag that tells if you log the data successfully.
    """
    rec_path = start_log_dvs(conn, save_path, title, viewer_id)
    time.sleep(duration)
    stop_log_dvs(conn)

    return dvsproc.check_aedat(rec_path+".aedat")

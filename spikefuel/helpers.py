"""Help functions related to image labeling, bounding box labeling.

This script also includes some experiment helpers.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import sys
import cv2
import av
import numpy as np
if os.name == "posix" and sys.version[0] < 3:
    import subprocess32 as subprocess
else:
    import subprocess


def cal_bound_box_ratio(pts, img_h, img_w):
    """Calculate relative position ratio around image.

    Parameters
    ----------
    pts : numpy.ndarray
        consists of bounding box information with size (n points, 2)
    img_h : int
        height of image
    img_w : int
        width of image

    Returns
    -------
    ratio : numpy.ndarray
        bounding box relative position ratio in a image
        represent as (ratio_value : 1)
        Example:
        |********|**********|
        |----8---|
        |--------18---------|
        (0.44:1)
    """
    ratio = pts.copy()

    if pts.ndim == 3:
        ratio[:, :, 0] = pts[:, :, 0]/float(img_w)
        ratio[:, :, 1] = pts[:, :, 1]/float(img_h)
    elif pts.ndim == 2:
        ratio[:, 0] = pts[:, 0]/float(img_w)
        ratio[:, 1] = pts[:, 1]/float(img_h)

    return ratio


def cal_bound_box_position(ratio, img_h, img_w):
    """Calculate relative position in axis around image.

    Parameters
    ----------
    pts : numpy.ndarray
        consists of bounding box information with size (n points, 2)
    img_h : int
        height of image
    img_w : int
        width of image

    Returns
    -------
    pts : numpy.ndarray
        new bounding boxes in the image
    """
    pts = ratio.copy()

    if ratio.ndim == 3:
        pts[:, :, 0] = ratio[:, :, 0]*float(img_w)
        pts[:, :, 1] = ratio[:, :, 1]*float(img_h)
    elif ratio.ndim == 2:
        pts[:, 0] = ratio[:, 0]*float(img_w)
        pts[:, 1] = ratio[:, 1]*float(img_h)

    return pts


def cal_img_shift(frame_size, img_size):
    """Calculate image shifts.

    Parameters
    ---------
    frame_size : tuple
        (frame height, frame width)
    img_size : tuple
        (image height, image width)

    Returns
    -------
    shift : tuple
        (x_shift, y_shift)
    """
    if len(frame_size) == 3:
        # color image
        f_h, f_w, _ = frame_size
    else:
        f_h, f_w = frame_size

    if len(img_size) == 3:
        # color image
        i_h, i_w, _ = img_size
    else:
        i_h, i_w = img_size

    # if the ratio is different, then append border
    if (float(f_h)/float(f_w)) != (float(i_h)/float(i_w)):
        if (float(f_h)/float(f_w)) > (float(i_h)/float(i_w)):
            x_shift = int((i_w*f_h-i_h*f_w)/(f_h))
            y_shift = 0
        elif (float(f_h)/float(f_w)) < (float(i_h)/float(i_w)):
            x_shift = 0
            y_shift = int((i_h*f_w-f_h*i_w)/(f_w))
    else:
        x_shift = 0
        y_shift = 0

    return (x_shift, y_shift)


def trans_groundtruth(groundtruth, method="size"):
    """Transform size based groundtruth to pixel positions.

    A popular bounding box labeling is based on 4 numbers:
    (topleftX, topleftY, bottomRightX, bottomRightX) <-- position based
    or
    (topleftX, topleftY, width, height) <-- size based

    Parameters
    ----------
    groundtruth : numpy.ndarray
        origonal groundtruth in numpy arrray, each row is a bounding box
    method : string
        groundtruth coding method,
            "size" for size based coding (default)
            "position" for position based coding

    Returns
    -------
    gt : numpy.ndarray
        new transformed groundtruth, each row contains 4 positions coded in
        (topleftX, topleftY, toprightX, toprightY,
         bottomRightX, bottomRightX, bottomLeftX, bottomLeftY)
    """
    gt_old = groundtruth.copy()
    gt = np.zeros((groundtruth.shape[0], 8), dtype=np.float)

    if method == "position":
        gt_old[:, 2] = gt_old[:, 2]-gt_old[:, 0]
        gt_old[:, 3] = gt_old[:, 3]-gt_old[:, 1]

    gt[:, :2] = gt_old[:, :2]
    gt[:, 2] = gt_old[:, 0]+gt_old[:, 2]
    gt[:, 3] = gt_old[:, 1]
    gt[:, 4] = gt_old[:, 0]+gt_old[:, 2]
    gt[:, 5] = gt_old[:, 1]+gt_old[:, 3]
    gt[:, 6] = gt_old[:, 0]
    gt[:, 7] = gt_old[:, 1]+gt_old[:, 3]

    return gt


def calibration(win_h, win_w, scale=0.9, window_title="test",
                bg_color=[255, 0, 0]):
    """Calibrate experiment setup.

    This function created a smaller window in the screen,
    and this smaller window is the sammpling window, by default,
    the ratio of the width and height is 4:3. A DVS should
    fill up this window as much as possible.

    Parameters
    ----------
    win_h : int
        Height of the window
    win_w : int
        Width of the window
    window_title : string
        A window title is needed if outside program has created a window
        display
    scale : float
        the scale of the sub-window

    Returns
    -------
    swin_h : int
        Height of the sub-window
    swin_w : int
        Width of the sub-window
    """
    # Put text on screen
    message = "Experiment Setup Calibration"
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    font_scale = 1
    thickness = 2
    font_size = cv2.getTextSize(message, font, font_scale, thickness)
    text_x = win_w/2 - font_size[0][0]/2
    text_y = win_h/2 + font_size[0][1]/2

    # Check if input window is 4:3
    if float(win_h)/float(win_w) != 0.75:
        raise ValueError("the input window is not in ratio 4:3")

    # get stats of smaller window
    swin_h = int(scale*win_h)
    swin_w = int(scale*win_w)

    window = np.ones((win_h, win_w, 3))*bg_color
    diff_y = (win_h-swin_h)/2
    diff_x = (win_w-swin_w)/2
    work_win = np.zeros((swin_h, swin_w, 3))
    window[diff_y:swin_h+diff_y, diff_x:swin_w+diff_x, :] = work_win
    window = np.array(window, dtype=np.uint8)

    cv2.putText(window, message, (text_x, text_y), font, fontScale=1,
                color=[255, 0, 255], thickness=1)

    flag = True
    while (1):
        # draw such window
        if flag is True:
            temp_win = window.copy()
            cv2.rectangle(temp_win, (diff_x, diff_y),
                          (diff_x+swin_w, diff_y+swin_h), color=[0, 255, 0],
                          thickness=1)
            flag = False
        elif flag is False:
            temp_win = window.copy()
            flag = True

        cv2.imshow(window_title, temp_win)

        k = cv2.waitKey(delay=10) & 0xFF
        if k == 27:
            break

    print "[MESSAGE] Experiment setup calibration is finished."

    return swin_h, swin_w


def start_jaer(jaer_path, jaer_exec="jAERViewer1.5_linux.sh"):
    """Start jAER from Python.

    This script is written for Linux usage,
    An error will be raised if it's Windows OS.
    Instead, windows user needs to manually setup jAER.

    Parameters
    ----------
    jaer_path : string
        absolute save path of jAER.
        e.g. /Users/dgyHome/Documents/workspace/jaer/trunk
    jaer_exec : string
        The executable of jAER. Version 1.5 is assumed.

    Returns
    -------
    An opened jAER viewer.
    """
    # Check OS type
    if os.name != "posix":
        raise ValueError("The Operating System is not a POSIX platform")

    commands = "cd "+jaer_path+"; bash "+jaer_exec
    process = subprocess.Popen(commands, stdout=subprocess.PIPE, shell=True)

    return process


def read_video(v_name):
    """A workaround function for reading video.

    Apparently precompiled OpenCV couldn't read AVI videos on Mac OS X
    and Linux,
    therefore I use PyAV, a ffmpeg binding to extract video frames

    Parameters
    ----------
    v_name : string
        absolute path to video

    Returns
    -------
    frames : list
        An ordered list for storing frames
    num_frames : int
        number of frames in the video
    """
    container = av.open(v_name)
    video = next(s for s in container.streams if s.type == b'video')

    frames = []
    for packet in container.demux(video):
        for frame in packet.decode():
            frame_t = np.array(frame.to_image())
            frames.append(cv2.cvtColor(frame_t, cv2.COLOR_RGB2BGR))

    return frames, len(frames)


def count_video_frames(v_name):
    """Counting video frames using PyAV."""
    container = av.open(v_name)
    video = next(s for s in container.streams if s.type == b'video')

    num_frames = 0
    for packet in container.demux(video):
        for frame in packet.decode():
            num_frames += 1

    return num_frames

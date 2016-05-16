"""This module consists of GUI functionalities.

+ Add border to certain image or frame with OpenCV Mat

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import numpy as np
import cv2


def create_border(frame, height, width, color):
    """Add constant border for the frame.

    The original frame will be displayed at the center of the monitor

    Parameters
    ----------
    frame : OpenCV Mat
        the original frame of image or video
    height : int
        height of the monitor
    width : int
        width of the monitor
    color : list
        the color of the border

    Returns
    -------
    new_frame : OpenCV Mat
        border added OpenCV Mat
    """
    vert = height/2-frame.shape[0]/2
    hor = width/2-frame.shape[1]/2

    new_frame = cv2.copyMakeBorder(src=frame, top=vert, bottom=vert, left=hor,
                                   right=hor, borderType=cv2.BORDER_CONSTANT,
                                   value=color)

    return new_frame


def create_border_sequence(frames, height, width, color):
    """Create borders for image sequence given monitor's parameters.

    Parameters
    ----------
    frames : list
        An ordered list that contains all frames. Frames are either in
        OpenCV Mat or numpy array
    height : int
        height of the monitor
    width : int
        width of the monitor
    color : list
        the color of the border

    Returns
    -------
    new_frames : list
        A list of frames that with borders
    """
    new_frames = []
    num_frames = len(frames)

    for i in xrange(num_frames):
        new_frames.append(create_border(frames[i], height, width, color))

    return new_frames


def gen_image_frames(img_path, fps, duration):
    """Generate frames from a image so that can play like a video.

    Parameters
    ----------
    img_path : string
        absolute path to the image.
    fps : int
        frame rates per second
    duration : float
        duration of the sequence in second
    """
    num_frames = int(fps*duration)

    img = cv2.imread(img_path)
    frames = []
    for i in xrange(num_frames):
        frames.append(img)

    return frames, len(frames)


def create_saccade_sequence(frames, border_size, color):
    """Create a saccade sequence using border.

    Parameters
    ----------
    frames : list
        given video sequence
    border_size : int
        border size in pixel
    color : list
        A three values list indicates border's color

    Returns
    -------
    new_frames : list
        a list of saccade frames
    """
    height = frames[0].shape[0]+2*border_size
    width = frames[0].shape[1]+2*border_size
    frames = create_border_sequence(frames, height, width, color)

    new_frames = []
    for frame in frames:
        h_ran = np.random.randint(3)
        v_ran = np.random.randint(3)

        if h_ran != 0:
            shift = {1: frame.shape[1]-border_size, 2: border_size}.get(h_ran)
            frame = np.roll(frame, shift, axis=1)
        if v_ran != 0:
            shift = {1: frame.shape[0]-border_size, 2: border_size}.get(v_ran)
            frame = np.roll(frame, shift, axis=0)

        new_frames.append(frame)

    return new_frames


def draw_poly_box(frame, pts, color=[0, 255, 0]):
    """Draw polylines bounding box.

    Parameters
    ----------
    frame : OpenCV Mat
        A given frame with an object
    pts : numpy array
        consists of bounding box information with size (n points, 2)
    color : list
        color of the bounding box, the default is green

    Returns
    -------
    new_frame : OpenCV Mat
        A frame with given bounding box.
    """
    new_frame = frame.copy()
    temp_pts = np.array(pts, np.int32)
    temp_pts = temp_pts.reshape((-1, 1, 2))
    cv2.polylines(new_frame, [temp_pts], True, color, thickness=2)

    return new_frame


def draw_poly_box_sequence(frames, pts_array, color=[0, 255, 0]):
    """Label bounding box for a sequence of frames.

    Parameters
    ----------
    frames : list
        An ordered list that contains all frames. Frames are either in
        OpenCV Mat or numpy array
    pts_array : numpy.ndarray
        contains bounding boxes for each frame respectively,
        in size of (frames, number of points, 2)
    color : list
        color of the bounding box, the default is green

    Returns
    -------
    new_frames : list
        A list of new frames with bounding boxes
    """
    new_frames = []

    num_frames = len(frames)

    for i in xrange(num_frames):
        new_frames.append(draw_poly_box(frames[i], pts_array[i], color))

    return new_frames


def rescale_image(frame, swin_h, swin_w, color=[255, 0, 0]):
    """Rescale image to size of working window.

    Parameters
    ---------
    frame : numpy.ndarray
        given image or frame
    swin_h : int
        width of the working window
    swin_w : int
        height of the working window
    color : list
        The background color of appended border
    """
    # new_frame = frame.copy()
    # try to save memory
    new_frame = frame
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    # if the ratio is different, then append border
    if (float(swin_h)/float(swin_w)) != (float(frame_h)/float(frame_w)):
        # do something
        if (float(frame_h)/float(frame_w)) > (float(swin_h)/float(swin_w)):
            w_append = int((frame_h*swin_w-swin_h*frame_w)/swin_h)
            new_frame = cv2.copyMakeBorder(src=new_frame, top=0, bottom=0,
                                           left=w_append/2, right=w_append/2,
                                           borderType=cv2.BORDER_CONSTANT,
                                           value=color)

        elif (float(frame_h)/float(frame_w)) < (float(swin_h)/float(swin_w)):
            h_append = int((swin_h*frame_w-frame_h*swin_w)/swin_w)
            new_frame = cv2.copyMakeBorder(src=new_frame, top=h_append/2,
                                           bottom=h_append/2, left=0, right=0,
                                           borderType=cv2.BORDER_CONSTANT,
                                           value=color)

    new_frame = cv2.resize(new_frame, (swin_w, swin_h),
                           interpolation=cv2.INTER_AREA)

    return new_frame


def rescale_image_sequence(frames, swin_h, swin_w, color=[255, 0, 0]):
    """Rescale a image sequence by given working window size.

    Parameters
    ----------
    frames : list
        contains list of frames in the image sequence
    swin_h : int
        width of the working window
    swin_w : int
        height of the working window
    color : list
        The background color of appended border

    Returns
    -------
    rescaled_frames : lists
        contains list of rescaled frames of given sequence
    """
    rescaled_frames = []
    for i in xrange(len(frames)):
        rescaled_frames.append(rescale_image(frames[i], swin_h, swin_w, color))

    return rescaled_frames

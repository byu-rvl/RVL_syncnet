from typing import Union, List

import numpy as np

import numpy as np
from skimage import img_as_ubyte
from skimage.transform import rescale, resize
import numpy as np
from scipy.spatial import distance

import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np


class ImageSequence:

    def __init__(self, arr: np.ndarray, fps: int, info: dict = None):
        self.arr = arr
        _shape = arr.shape
        self.seq_len = _shape[0]
        self.height = _shape[1]
        self.width = _shape[2]

        self.fps = fps
        self.duration = None
        self.time_stamps = None
        self.time_base = None

        self.info = info
        # self.__dict__.update(info)

        self.fps15 = self._get_fps15()

    def save(self):
        pass

    def _get_fps15(self):
        dsr = round(self.fps / 15)
        # TODO: add upsampling case
        msk = np.arange(self.seq_len)
        msk = msk % dsr == 0
        return self.arr[msk]

__here__ = Path(__file__).parent


if "blazeData" in globals() and blazeData is not None:
    detectFace_base_options =python.BaseOptions(model_asset_buffer=blazeData)
    del globals()['blazeData']
else:
    path = Path(str(__here__.parent.parent) +'/facetools/data/blaze_face_short_range.tflite')
    if not path.exists():
        path = Path(str(__here__) +'/data/blaze_face_short_range.tflite')
    detectFace_base_options = python.BaseOptions(model_asset_path=str(path))

detectFace_options = vision.FaceDetectorOptions(base_options=detectFace_base_options)
DETECT_FACE_DETECTOR = vision.FaceDetector.create_from_options(detectFace_options)

def detectFace_helper(frame):
    roi = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    facesFound = DETECT_FACE_DETECTOR.detect(roi).detections

    if facesFound:
        boundedBox = facesFound[0].bounding_box
        detected = np.array((boundedBox.origin_x , boundedBox.origin_y , (boundedBox.origin_x + boundedBox.width) , (boundedBox.origin_y + boundedBox.height)))
        detected = np.resize(detected, (1,4))

        return detected
    else:
        return np.array(None)


def findFace(image, door=None):
    detected = detectFace_helper(image)
    if not len(detected.shape):
        return False, detected
    scaleFactor = 1
    if door is not None:
        scaleFactor = door.get_face_scale()

    ymin = detected[0][1]
    ymax = detected[0][3]
    imageHeight = image.shape[0]
    faceHeight = (ymax - ymin)/imageHeight
    return faceHeight > scaleFactor, detected



def rescale_to_ceiling512(img: np.ndarray):
    size = 512
    _h, _w = img.shape[:2]
    _r = size / _h if _h > _w else size / _w
    img_ = rescale(img, scale=_r, order=1, mode='edge', preserve_range=False, multichannel=True,
                   anti_aliasing=True)  # bi-linear
    img_ = img_as_ubyte(img_)
    return img_


def resize_to_ceiling(img: np.ndarray, target_size: int):
    h0, w0 = img.shape[:2]
    w_ = target_size if w0 >= h0 else round(w0 * target_size / h0)
    h_ = target_size if w0 <= h0 else round(h0 * target_size / w0)
    size_ = (h_, w_)
    # TODO: test this interpolation method (bi-linear)
    img_ = resize(image=img, output_shape=size_, order=1, mode='edge', preserve_range=False, anti_aliasing=True)
    img_ = img_as_ubyte(img_)
    return img_



def _find_larger_face_idx(face_locations):
    idx_max = -1
    s_max = 0
    for idx, bbox in enumerate(face_locations):
        print(bbox)
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        _w = xmax - xmin + 1
        _h = ymax - ymin + 1
        _S = _h * _w
        if _S > s_max:
            idx_max = idx
            s_max = _S
    return idx_max


def _crop_face_region(arr, bbox, expansion=1.2):
    h0, w0 = arr.shape[:2]
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[2]
    ymax = bbox[3]
    _w = xmax - xmin + 1
    _h = ymax - ymin + 1
    _top = ymin
    _left = xmin

    _half_size = max(_w, _h) * expansion / 2
    _rc = _top + (_h / 2)
    _cc = _left + (_w / 2)

    top_ = int(round(_rc - _half_size))
    left_ = int(round(_cc - _half_size))
    bottom_ = int(round(_rc + _half_size))
    right_ = int(round(_cc + _half_size))

    top_new = 0 if top_ < 0 else top_
    left_new = 0 if left_ < 0 else left_
    bottom_new = h0 if bottom_ > h0 else bottom_
    right_new = w0 if right_ > w0 else right_
    # TODO: create a test

    return arr[top_new:bottom_new, left_new:right_new]


def _resize_face_region(roi: np.ndarray, target_size: int):
    _shape = list(roi.shape)
    roi_ = resize_to_ceiling(roi, target_size)
    h_, w_ = roi_.shape[:2]
    # canvas
    _shape[0] = target_size
    _shape[1] = target_size
    x = np.zeros(shape=_shape, dtype=np.uint8)
    if h_ == w_:
        x[:h_, :w_] = roi_
    elif h_ < w_:
        top_ = (target_size - h_) // 2
        bottom_ = top_ + h_
        x[top_:bottom_, :w_] = roi_
    else:
        left_ = (target_size - w_) // 2
        right_ = left_ + w_
        x[:h_, left_:right_] = roi_
    return x


def get_cropped_face_img(img: np.ndarray, face_bbox: List, expansion: float = 1.3,
                         target_size: int = 112, largest=True):
    """
    Crop, resize, and center the face image(s) using detected bounding box(es).

    :param img: A whole frame image.
    :param face_bbox: Bounding box or a list of bounding boxes.
    :param expansion: The rate to expand the detected box.
    :param target_size: The size of the returned square face image.
    :param largest: If True, only return the the face with largest area.
    :return:
    """

    def _crop_face(_bbox):
        roi = _crop_face_region(arr=img, bbox=_bbox, expansion=expansion)
        roi_ = _resize_face_region(roi=roi, target_size=target_size)
        return roi_

    if isinstance(face_bbox[0], int):
        return _crop_face(face_bbox)

    n_loc = len(face_bbox)
    if n_loc == 0:
        return

    if largest:
        idx_face = 0
        if n_loc > 1:
            print("MULTIPLE FACES DETECTED")
            idx_face = _find_larger_face_idx(face_bbox)
        bbox = face_bbox[idx_face]
        return _crop_face(bbox)

    return list(map(_crop_face, face_bbox))


def _find_mean_face_location(frame_seq):
    pass


def rescale_bbox(face_bbox: Union[dict, List[dict]], size_orig: int, size_det: int):
    """Rescale the detected bbox(es) (H and W share the same rate)."""

    def _rescale_bbox(_bbox):
        r = size_orig / size_det
        keys = ['height', 'width', 'r', 'c']
        for k in keys:
            _bbox[k] = round(_bbox[k] * r)
        return _bbox

    if isinstance(face_bbox, dict):
        return _rescale_bbox(face_bbox)

    return list(map(_rescale_bbox, face_bbox))

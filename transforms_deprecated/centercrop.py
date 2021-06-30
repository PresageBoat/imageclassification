import numpy as np
import numbers
from typing import Sequence,List
from transforms.utils import center_crop

def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class CenterCrop(object):
    """Crops the given image at the center.
    If the image is opencv image, it is expected
    to have [H, W, C] shape.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """
    def __init__(self,size):
        super().__init__()
        self.size=_setup_size(size,error_msg="Please provide only two dimensions (h, w) for size.")

    def __call__(self, img:np.array)->np.array:
        return center_crop(img,self.size)


def test():
    import cv2
    import numpy as np
    img=cv2.imread('./3.jpg')
    cv2.imwrite('org.tmp.jpg',img)
    c_crop=CenterCrop(size=224)
    input_img=img[::]
    image=c_crop(input_img)

    cv2.imwrite('centercrop.tmp.jpg',image)


if __name__ == '__main__':
    test()
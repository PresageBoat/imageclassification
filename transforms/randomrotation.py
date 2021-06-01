
import numpy as np
import random
import numbers
from collections.abc import Sequence
import cv2 as cv
from typing import List

def _setup_angle(x):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.")
        x = [-x, x]
    else:
        raise TypeError("input type must number")
    return [float(d) for d in x]


class RandomRotation(object):
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number): Pixel fill value for the area outside the rotated
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
        resample (int, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``interpolation`` parameter instead.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self,degrees,fill=0):
        super().__init__()
        if not isinstance(degrees, (int, float)):
            raise TypeError("Argument angle should be int or float")

        self.degrees=_setup_angle(degrees)

        # if fill is None:
        #     fill=0
        # elif not isinstance(fill, (Sequence, numbers.Number)):
        #     raise TypeError("Fill should be either a sequence or a number.")
        # self.fill=fill

    @staticmethod
    def get_params(degrees:List[float])->float:
        angle=random.uniform(float(degrees[0]),float(degrees[1]))
        return angle
    
    def __call__(self, img:np.array)->np.array:
        angle = self.get_params(self.degrees)
        height,width,_=img.shape
        matRotate = cv.getRotationMatrix2D((width*0.5, height*0.5), angle, 1.) # 旋转变化矩阵
        dst = cv.warpAffine(img, matRotate, (width,height))#旋转
        return dst


def test():
    import cv2
    import numpy as np
    img=cv2.imread('./2.jpg')
    cv2.imwrite('org.tmp.jpg',img)
    r_rotate=RandomRotation(degrees=45)
    input_img=img[::]
    image=r_rotate(input_img)

    cv2.imwrite('RandomRotation.tmp.jpg',image)


if __name__ == '__main__':
    test()
import numpy as np
from typing import List, Tuple
import random
import math
import cv2

from utils import resize


class RandomResizedCrop(object):
    """Crop a random portion of image and resize it to a given size.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.

    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)) -> None:
        super().__init__()
        self.size=size
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: np.array, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (opencv Image ): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """

        height, width, _ = img.shape
        area = height*width

        for _ in range(10):
            target_area = area*random.uniform(scale[0], scale[1])
            aspect_ratio = np.exp(random.uniform(ratio[0], ratio[1]))

            w = int(round(math.sqrt(target_area*aspect_ratio)))
            h = int(round(math.sqrt(target_area * aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height-h+1)
                j = random.randint(0, width-w+1)
                return i, j, w, h
            
            #Fallback to central crop 
            in_ratio=float(width)/float(height)

            if in_ratio<min(ratio):
                w=width
                h=int(round(w/min(ratio)))
            elif in_ratio>max(ratio):
                h=height
                w=int(round(h*max(ratio)))
            else: # whole image
                w=width
                h=height
            i=(height-h)//255
            j=(width-w)//255
            return i, j,h,w

    def __call__(self, img:np.array)->np.array:
        i,j,h,w=self.get_params(img,self.scale,self.ratio)
        #crop images
        img_crop=img[i:i+h,j:j+w]
        return resize(img_crop,self.size)


def test():
    import cv2
    import numpy as np
    img=cv2.imread('./2.jpg')
    cv2.imwrite('org.tmp.jpg',img)
    rsc=RandomResizedCrop(size=256)
    input_img=img[::]
    image=rsc(input_img)

    cv2.imwrite('randomresizedcrop.tmp.jpg',image)

if __name__=='__main__':
    test()

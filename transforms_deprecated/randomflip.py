from typing import List
import numpy as np
import random
import cv2 as cv

class RandomFlip(object):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """
    def __init__(self,p=0.5)->None:
        super().__init__()
        self.p=p
    
    def __call__(self, img:np.array)->np.array:
        """
        Args:
            img (opencv Image ): Image to be flipped.

        Returns:
            opencv Image : Randomly flipped image.
        """
        if random.uniform(0,1)>self.p:
            return cv.flip(img,-1)



import numpy as np
from typing import List,Optional, Type
import cv2 as cv
import math

def resize(img:np.array,size:List[int],max_size:Optional[int]=None)->np.array:

    if not isinstance(size, (int, tuple, list)):
        raise TypeError("Got inappropriate size arg")

    if img is None:
        raise ValueError('Input images is null')

    if isinstance(size, tuple):
        size = list(size)
    
    if isinstance(size, list):
        if len(size) not in [1, 2]:
            raise ValueError("Size must be an int or a 1 or 2 element tuple/list, not a "
                             "{} element tuple/list".format(len(size)))
        if max_size is not None and len(size) != 1:
            raise ValueError(
                "max_size should only be passed if size specifies the length of the smaller edge, "
                "i.e. size should be an int or a sequence of length 1 in torchscript mode."
            )
    
    h,w,_=img.shape
    if isinstance(size,int) or len(size)==1: #specified size only for the smallest edge
        short ,long=(w,h) if w<=h else(h,w)
        requested_new_short=size if isinstance(size,int) else size[0]

        if short ==requested_new_short:
            return img
        
        new_short, new_long = requested_new_short,int(requested_new_short*long/short)

        if max_size is not None:
            if max_size <= requested_new_short:
                raise ValueError(
                    f"max_size = {max_size} must be strictly greater than the requested "
                    f"size for the smaller edge size = {size}"
                )
            if new_long > max_size:
                new_short, new_long = int(max_size * new_short / new_long), max_size

        new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
    else:  # specified both h and w
        new_w, new_h = size[1], size[0]

    img=cv.resize(img,[new_h,new_w],interpolation = cv.INTER_LINEAR)

    return img

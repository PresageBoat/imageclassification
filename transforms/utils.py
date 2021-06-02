import numpy as np
from typing import List,Optional, Type
import cv2 as cv
import math
import numbers

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



def center_crop(img:np.array,output_size:List[int])->np.array:
    """Crops the given image at the center.
    If the image is opencv image, it is expected
    to have [H, W, C] shape.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        img (opencv Image ): Image to be cropped.
        output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
            it is used for both directions.

    Returns:
        opencv Image : Cropped image.
    """

    if isinstance(output_size,numbers.Number):
        output_size = (int(output_size),int(output_size))
    elif isinstance(output_size,(tuple,list)) and len(output_size)==1:
        output_size =(output_size[0],output_size[0])
    
    image_height,image_width,_=img.shape
    crop_height,crop_width=output_size
    if crop_width > image_width or crop_height > image_height:
        padding_ltrb = [
            (crop_width - image_width) // 2 if crop_width > image_width else 0,
            (crop_height - image_height) // 2 if crop_height > image_height else 0,
            (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
            (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        ]
        img=cv.copyMakeBorder(img,padding_ltrb[1],padding_ltrb[3],padding_ltrb[0],padding_ltrb[2],cv.BORDER_CONSTANT,value=(0,0,0))
        image_height,image_width,_=img.shape
        if crop_width == image_width and crop_height == image_height:
            return img

    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return img[crop_top:crop_top+crop_height,crop_left:crop_left+crop_width]

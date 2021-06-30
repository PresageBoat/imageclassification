from typing import List
import numpy as np

class Normalize(object):
    """Normalize a image with mean and standard deviation.
    This transform only test opencv Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutate the input image.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
    def __init__(self,mean: List[float],std: List[float])->None:
        super().__init__()
        self.mean=mean
        self.std=std

    def __call__(self,img:np.array)->np.array:
        channels=len(self.mean)
        div_255=[255.0]*channels
        if img is None:
            raise ValueError('Input images is null')
        img=(img/div_255-self.mean)/self.std
        return img
    

def test():
    import cv2
    import numpy as np
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    img=cv2.imread('./1.jpg')
    cv2.imwrite('org.tmp.jpg',img)
    Norm=Normalize(mean=MEAN,std=STD)
    input_img=img[::]
    image=Norm(input_img)

    cv2.imwrite('normalize.tmp.jpg',image*255)


if __name__=='__main__':
    test()
    


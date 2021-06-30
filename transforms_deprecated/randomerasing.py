import math
import random
import numpy as np


class RandomErasing_Tensor(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
         img: RGB format,torch.tensor type
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        print(img.shape)

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
         img: RGB format ；from cv2.imread,just 3 channels
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465])->None:
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img:np.array)->np.array:
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            rows,cols,channels=img.shape
            area = rows * cols

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < cols and h < rows:
                x1 = random.randint(0, rows - h)
                y1 = random.randint(0, cols - w)
                if channels == 3:
                    # img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    # img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    # img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]
                    img[x1:x1+h,y1:y1+w]=self.mean
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def RandomErasing_Tensor_test():
    import cv2
    import numpy as np
    import torch
    from torchvision import transforms

    def toTensor(img):
        assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img)
        return img.float().div(255).unsqueeze(0)  # 255也可以改为256

    def tensor_to_np(tensor):
        img = tensor.mul(255).byte()
        img = img.cpu().numpy().squeeze(0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


    img=cv2.imread('./1.jpg')
    cv2.imwrite('org.tmp.jpg',img)
    RE=RandomErasing_Tensor(probability=1)
    img_tensor=toTensor(img)
    image=RE(img_tensor)

    cv2.imwrite('randomearsing.tmp.jpg',tensor_to_np(image))


def RandomErasing_test():
    import cv2
    import numpy as np
    mean=[0,0,0]
    img=cv2.imread('./1.jpg')
    cv2.imwrite('org.tmp.jpg',img)
    RE=RandomErasing(probability=1,mean=mean)
    input_img=img[::]
    image=RE(input_img)

    cv2.imwrite('randomearsing.tmp.jpg',input_img)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--module',type=str,default='numpy')
    opt=parser.parse_args()
    if opt.module =='torch.tensor':
        RandomErasing_Tensor_test()
    else:
        RandomErasing_test()

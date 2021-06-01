# from torchvision import transforms
# # from RandomErasing import RandomErasing

# MEAN = [0.485, 0.456, 0.406]
# STD = [0.229, 0.224, 0.225]

# class ImageTransform():

#     def __init__(self):
#         pass

#     def __call__(self):
#         trans_train = transforms.Compose([
#         transforms.RandomResizedCrop(256, scale=(0.08, 1.0)),
#         transforms.RandomRotation(15),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#         RandomErasing(probability=0.5, mean=[0, 0, 0])
#     ])

#     trans_val = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])


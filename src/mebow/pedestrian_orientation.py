from torch import torch
import numpy as np
import torchvision

from numba import jit

# transforming the image to the correct size for network input
class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad)
                             for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return torchvision.transforms.functional.pad(image, padding, 0, 'constant')

class SquarePadNumpy:
    def __call__(self, image):
        wh = (image.shape[0], image.shape[1])
        max_wh = max(wh)
        p_left, p_top = [(max_wh - s) // 2 for s in wh]
        p_right, p_bottom = [max_wh - (s+pad)
                             for s, pad in zip(wh, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return torchvision.transforms.functional.pad(image, padding, 0, 'constant')

def init_pedestrian_orientation_model():
    model = torchvision.models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 72),
        torch.nn.Softmax(),
    )


    # ! because the model was trained and saved with nn.DataParallel, dict key names are different 
    # ! e.g. module.conv1.weight instead of conv1.weight, this is a hack to load without parallel
    # original saved file with DataParallel
    pre_model = '../models/pose_model_resnet18.pth'
    state_dict = torch.load(pre_model)
    # create new OrderedDict that does not contain module.
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove module.
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

    # if 'state_dict' in checkpoint:
    #     model.load_state_dict(checkpoint['state_dict'], strict=True)
    # else:
    #     model.load_state_dict(checkpoint, strict=True)

    # ! or we can load with DataParallel
    # model = torch.nn.DataParallel(model, device_ids=(0,)).cuda()

    model.cuda()
    # model = model.cuda()
    model.eval()
    print("Pytorch model loaded")

    return model


def convert_to_tensor(PIL_image):
    # image transformations
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    target_image_size = (224, 224)
    tr = torchvision.transforms.Compose([
        SquarePad(),
        torchvision.transforms.Resize(target_image_size),
        torchvision.transforms.ToTensor(),
        normalize,
    ])
    testim_torch = tr(PIL_image)
    return testim_torch

def convert_to_tensor_numpy(numpy_array):
    # image transformations
    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    target_image_size = (224, 224)
    tr = torchvision.transforms.Compose([
        SquarePadNumpy(),
        torchvision.transforms.Resize(target_image_size),
        torchvision.transforms.ToTensor(),
        normalize,
    ])
    testim_torch = tr(numpy_array)
    return testim_torch


def get_prediction(model, tensor_list):
    orientation_values = []

    if (len(tensor_list) == 0):
        return orientation_values

    input_tensor = torch.stack(tensor_list)
    input_tensor = input_tensor.cuda()
    pred = model(input_tensor)

    for i in range(len(tensor_list)):
        y = pred.detach().cpu().numpy()[i]

        orientation = np.argmax(y) * 5
        orientation_values.append(orientation)

    return orientation_values

@jit
def get_prediction_single(model, tensor):
    input_tensor = tensor.unsqueeze(0)
    input_tensor = input_tensor.cuda()
    pred = model(input_tensor)
    y = pred.detach().cpu().numpy()
    orientation = np.argmax(y) * 5

    return orientation

import os
import pickle
import torch
import scipy.io as sio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from crnn import CRNN


class FixHeightResize(object):
    """
    Scale images to fixed height
    """

    def __init__(self, height=32, minwidth=100):
        self.height = height
        self.minwidth = minwidth

    # img is an instance of PIL.Image
    def __call__(self, img):
        w, h = img.size
        width = max(int(w * self.height / h), self.minwidth)
        return img.resize((width, self.height), Image.ANTIALIAS)


class IIIT5k(Dataset):
    """
    IIIT-5K dataset，(torch.utils.data.Dataset)

    Args:
        root (string): Root directory of dataset
        training (bool, optional): If True, train the model, otherwise test it (default: True)
        fix_width (bool, optional): Scale images to fixed size (default: True)
    """

    def __init__(self, root, training=True, fix_width=True):
        super(IIIT5k, self).__init__()
        data_str = 'traindata' if training else 'testdata'
        data = sio.loadmat(os.path.join(root, data_str+'.mat'))[data_str][0]
        self.img, self.label = zip(*[(x[0][0], x[1][0]) for x in data])

        # image resize + grayscale + transform to tensor
        transform = [transforms.Resize((32, 100), Image.BILINEAR)
                     if fix_width else FixHeightResize(32)]
        transform.extend([transforms.Grayscale(), transforms.ToTensor()])
        transform = transforms.Compose(transform)

        # load images
        self.img = [transform(Image.open(root+'/'+img)) for img in self.img]

    def __len__(self, ):
        return len(self.img)

    def __getitem__(self, idx):
        return self.img[idx], self.label[idx]


def load_data(root, training=True, fix_width=True):
    """
    load IIIT-5K dataset

    Args:
        root (string): Root directory of dataset
        training (bool, optional): If True, train the model, otherwise test it (default: True)
        fix_width (bool, optional): Scale images to fixed size (default: True)

    Return:
        Training set or test set
    """

    if training:
        batch_size = 128 if fix_width else 1
        filename = os.path.join(
            root, 'train'+('_fix_width' if fix_width else '')+'.pkl')
        if os.path.exists(filename):
            dataset = pickle.load(open(filename, 'rb'))
        else:
            print('==== Loading data.. ====')
            dataset = IIIT5k(root, training=True, fix_width=fix_width)
            pickle.dump(dataset, open(filename, 'wb'))
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, num_workers=4)
    else:
        batch_size = 128 if fix_width else 1
        filename = os.path.join(
            root, 'test'+('_fix_width' if fix_width else '')+'.pkl')
        if os.path.exists(filename):
            dataset = pickle.load(open(filename, 'rb'))
        else:
            print('==== Loading data.. ====')
            dataset = IIIT5k(root, training=False, fix_width=fix_width)
            pickle.dump(dataset, open(filename, 'wb'))
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4)
    return dataloader


class LabelTransformer(object):
    """
    encoder and decoder

    Args:
        letters (str): Letters contained in the data
    """

    def __init__(self, letters):
        self.encode_map = {letter: idx+1 for idx, letter in enumerate(letters)}
        self.decode_map = ' ' + letters

    def encode(self, text):
        if isinstance(text, str):
            length = [len(text)]
            result = [self.encode_map[letter] for letter in text]
        else:
            length = []
            result = []
            for word in text:
                length.append(len(word))
                result.extend([self.encode_map[letter] for letter in word])
        return torch.IntTensor(result), torch.IntTensor(length)

    def decode(self, text_code):
        result = []
        for code in text_code:
            word = []
            for i in range(len(code)):
                if code[i] != 0 and (i == 0 or code[i] != code[i-1]):
                    word.append(self.decode_map[code[i]])
            result.append(''.join(word))
        return result

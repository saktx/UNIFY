import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

def tinyimagenet_dataloaders(train_batch_size=64, test_batch_size=100, data_dir='datasets/tiny-imagenet-200'):

    #############################################################
    train_transform = transforms.Compose([transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    data_dir = r"C:\RESEARCH_EXPERIMENTS\PROJECTS\UNIFY\datasets\tiny-imagenet-200/"
    
    
    class Tinyimagenet_train(Dataset):
        def __init__(self, root, transform):
            self.root, self.transform = root, transform
            self.label_map = {}
            with open(os.path.join(root, '../wnids.txt')) as file:
                for i, line in enumerate(file.readlines()):
                    line = line[:-1]
                    self.label_map[line] = i
            self.images, self.labels = [], []
            for label in self.label_map.keys():
                for image in os.listdir(os.path.join(root, label, 'images')):
                    self.images.append(os.path.join(label, 'images', image))
                    self.labels.append(label)
        def __len__(self):
            return len(self.images)
        def __getitem__(self, item):
            image, label = self.images[item], self.labels[item]
            image = Image.open(os.path.join(self.root, image))
            image = image.convert('RGB')
            image = self.transform(image)
            label = self.label_map[label]
            return image, label

    class Tinyimagenet_val(Dataset):
        def __init__(self, root, transform):
            self.root, self.transform = root, transform
            self.images, self.labels = [], []
            with open(os.path.join(root, 'val_annotations.txt')) as file:
                for line in file.readlines():
                    image, label = line.split('	')[:2]
                    self.images.append(image)
                    self.labels.append(label)
            self.label_map = {}
            with open(os.path.join(root, '../wnids.txt')) as file:
                for i, line in enumerate(file.readlines()):
                    line = line[:-1]
                    self.label_map[line] = i
        def __len__(self):
            return len(self.images)
        def __getitem__(self, item):
            image, label = self.images[item], self.labels[item]
            image = Image.open(os.path.join(self.root, 'images', image))
            image = image.convert('RGB')
            image = self.transform(image)
            label = self.label_map[label]
            return image, label
    
    tinyimagenet_train = Tinyimagenet_train(os.path.join(data_dir, 'train'), transform=train_transform)
    tinyimagenet_val = Tinyimagenet_val(os.path.join(data_dir, 'val'), transform=test_transform)
    train_loader = DataLoader(tinyimagenet_train, batch_size=train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(tinyimagenet_val, batch_size=test_batch_size, shuffle=False, drop_last=True)
    #############################################################
    return train_loader, val_loader     # test_loader
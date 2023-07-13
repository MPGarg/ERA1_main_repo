from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch_lr_finder import LRFinder

class cifar_ds10(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

def tl_ts_mod(transform_train,transform_valid,batch_size=512):
    #for determining LR
    trainset_lr = cifar_ds10(root='./data', train=True, download=True, transform=transform_valid)
    trainloader_lr = torch.utils.data.DataLoader(trainset_lr, batch_size=batch_size, shuffle=True, num_workers=2)
    #for main training
    trainset = cifar_ds10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = cifar_ds10(root='./data', train=False, download=True, transform=transform_valid)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainset,trainloader,testset,testloader,trainset_lr,trainloader_lr
  
def set_albumen_params(mean, std):
    num_holes= 1
    cutout_prob= 0.5
    max_height = 8
    max_width = 8

    transform_train = A.Compose(
        [
        A.PadIfNeeded(min_height=36, min_width=36),
        A.RandomCrop(width=32, height=32,always_apply=False),
        A.HorizontalFlip(p=0.2),
        A.CoarseDropout(max_holes=num_holes,min_holes = 1, max_height=max_height, max_width=max_width, 
        p=cutout_prob,fill_value=tuple([x * 255.0 for x in mean]),
        min_height=max_height, min_width=max_width, mask_fill_value = None),
        A.Normalize(mean = mean, std = std,p=1.0, max_pixel_value=255, always_apply = True),
        ToTensorV2()
        ])
    
    transform_valid = A.Compose(
        [
        A.Normalize(
                mean=mean,
                std=std,
                p=1.0,
                max_pixel_value=255,
            ),
        ToTensorV2()
        ])
    return transform_train, transform_valid 

def load_data():
    transform = transforms.Compose(
      [transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)
    return trainloader, trainset  

def show_sample(dataset):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
    dataiter = iter(dataset)

    index = 0
    fig = plt.figure(figsize=(20,10))
    for i in range(10):
        images, labels = next(dataiter)
        actual = classes[labels]
        image = images.squeeze().to('cpu').numpy()
        ax = fig.add_subplot(2, 5, index+1)
        index = index + 1
        ax.axis('off')
        ax.set_title(f'\n Label : {actual}',fontsize=10) 
        ax.imshow(np.transpose(image, (1, 2, 0))) 
        images, labels = next(dataiter) 
    
def process_dataset(batch_size=512,visualize = ''):
    trl, trs = load_data()
    
    mean = list(np.round(trs.data.mean(axis=(0,1,2))/255., 4))
    std = list(np.round(trs.data.std(axis=(0,1,2))/255.,4))
        
    transform_train, transform_valid = set_albumen_params(mean, std)
    trainset_mod, trainloader_mod, testset_mod, testloader_mod,trainset_lr,trainloader_lr = tl_ts_mod(transform_train,transform_valid,batch_size=batch_size)

    if visualize == 'X':
        show_sample(trs)

    return trainset_mod, trainloader_mod, testset_mod, testloader_mod , mean, std ,trainset_lr,trainloader_lr


def save_model(model, epoch, optimizer, path):
    """Save torch model in .pt format

    Args:
        model (instace): torch instance of model to be saved
        epoch (int): epoch num
        optimizer (instance): torch optimizer
        path (str): model saving path
    """
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)

def plot_acc_loss(train_acc,train_losses,test_acc,test_losses):
    fig, axs = plt.subplots(1,2,figsize=(15,5))

    axs[0].plot(train_losses, label='Training Losses')
    axs[0].plot(test_losses, label='Test Losses')
    axs[0].legend(loc='upper right')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title("Loss")

    axs[1].plot(train_acc, label='Training Accuracy')
    axs[1].plot(test_acc, label='Test Accuracy')
    axs[1].legend(loc='lower right')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title("Accuracy")

    plt.show()    

def display_incorrect_pred(mismatch, n=20 ):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    display_images = mismatch[:n]
    index = 0
    fig = plt.figure(figsize=(20,20))
    for img in display_images:
        image = img[0].squeeze().to('cpu').numpy()
        pred = classes[img[1]]
        actual = classes[img[2]]
        ax = fig.add_subplot(4, 5, index+1)
        ax.axis('off')
        ax.set_title(f'\n Predicted Label : {pred} \n Actual Label : {actual}',fontsize=10) 
        ax.imshow(np.transpose(image, (1, 2, 0))) 
        #ax.imshow(image, cmap='gray_r')
        index = index + 1
    plt.show()

def find_lr(net, optimizer, criterion, train_loader):
    """Find learning rate for using One Cyclic LRFinder
    Args:
        net (instace): torch instace of defined model
        optimizer (instance): optimizer to be used
        criterion (instance): criterion to be used for calculating loss
        train_loader (instance): torch dataloader instace for trainig set
    """
    lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=10, num_iter=100, step_mode="exp")
    lr_finder.plot()
    lr_finder.reset()


     

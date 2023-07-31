import torch 
import torchvision
import numpy as np
import pandas as pd
import os
from torchvision import transforms, datasets
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from operator import itemgetter
from PIL import Image
from sklearn.model_selection import train_test_split
# from torch.utils.data import Dataset, TensorDataset

DATA_PATH ='/p/compressionleakage'

utk_data_path = "/p/compressionleakage/age_gender.csv"

def get_dataloaders_celeba(batch_size, num_workers=0,
                           train_transforms=None,
                           test_transforms=None,
                           get_attr = lambda attr: attr[31],
                           download=True):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()
    

    train_dataset = datasets.CelebA(root='.',
                                    split='train',
                                    transform=train_transforms,
                                    target_type='attr',
                                    target_transform=get_attr,
                                    download=download)

    valid_dataset = datasets.CelebA(root='.',
                                    split='valid',
                                    target_type='attr',
                                    target_transform=get_attr,
                                    transform=test_transforms)

    test_dataset = datasets.CelebA(root='.',
                                   split='test',
                                   target_type='attr',
                                   target_transform=get_attr,
                                   transform=test_transforms)


    # print(test_dataset.attr_names.index('Male'))

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    return train_loader, valid_loader, test_loader



def get_data(dataset_name, to_drop = None, clip_percentage = None, train_batch_size = 100, val_batch_size = 100, get_class = None, get_loader = True):
    np.random.seed(1000)
    if dataset_name == 'celeba':
        path = os.path.join(DATA_PATH, dataset_name)
        BATCH_SIZE = 256
        # NUM_EPOCHS = 25
        # LEARNING_RATE = 0.001
        # NUM_WORKERS = 4

        custom_transforms = transforms.Compose([
            transforms.CenterCrop((160, 160)),
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        get_smile = lambda attr: attr[31]
        get_male = lambda attr: attr[20]

        train_loader, val_loader, test_loader = get_dataloaders_celeba(
            batch_size=BATCH_SIZE,
            train_transforms=custom_transforms,
            test_transforms=custom_transforms,
            get_attr=get_smile,
            download=False,
            num_workers=4)


    if dataset_name == 'UTK':
        label = 'ethnicity'
        pd00 = pd.read_csv(utk_data_path)
        age_bins = [0, 10, 15, 20, 25, 30, 40, 50, 60, 120]
        age_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        pd00['age_bins'] = pd.cut(pd00.age, bins=age_bins, labels=age_labels)
        X = pd00.pixels.apply(lambda x: np.array(x.split(" "), dtype=float))
        X = np.stack(X)
        X = X / 255.0
        X = X.astype('float32').reshape(X.shape[0], 1, 48, 48)
        y = pd00[label].values
        # np.random.seed(options['seed'])  # random seed of partition data into train/test
        x_train, x_test, y_train, y_test  = train_test_split(X, y,  test_size=0.2)
        
        train_tensor = TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train))
        train_loader = DataLoader(dataset=train_tensor, batch_size=train_batch_size, shuffle=True)
        x_test, y_test = torch.FloatTensor(x_test).cuda(), torch.LongTensor(y_test).cuda()

        test_tensor = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        # create a DataLoader for the test set
        test_loader = DataLoader(test_tensor, batch_size=train_batch_size, shuffle=False)
        
    elif dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        # Normalize the test set same as training set without augmentation
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # trainset= torchvision.datasets.CIFAR10(download=False, train=True, root="./data/", transform = transform_train)
        # testset = torchvision.datasets.CIFAR10(download=False, train=False, root="./data/", transform = transform_test)
        
        trainset = torchvision.datasets.CIFAR10(root= DATA_PATH, train=True,
                                            download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root= DATA_PATH, train=False,
                                           download=True, transform=transform_test)

        print(len(testset.data), len(testset.targets))
        train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=20, prefetch_factor = 5, drop_last=True)
        val_loader = DataLoader(testset, batch_size=val_batch_size, shuffle=False, num_workers=20, prefetch_factor = 5)


    if to_drop is not None and clip_percentage is not None:
        if dataset_name == 'CIFAR10':
            drop_class = int(to_drop)
            
            indx_train = [i for i, e in enumerate(trainset.targets) if e == drop_class]
            # np.isin
            n_samples = int(clip_percentage * len(indx_train))
            indx_train = indx_train[:n_samples]
            trainset.data = np.delete(trainset.data, indx_train, axis = 0)
            trainset.targets = np.delete(trainset.targets, indx_train)

            train_loader = DataLoader(trainset, shuffle=True, batch_size=train_batch_size, num_workers=20, prefetch_factor = 5, drop_last=True)



        # indx_test = [i for i, e in enumerate(testset.targets) if e == drop_class]
        # n_samples = int(clip_percentage * len(indx_test))
        # indx_test = indx_test[:n_samples]
        # testset.data = np.delete(testset.data, indx_test, axis = 0)
        # testset.targets = np.delete(testset.targets, indx_test)

        # all_indices = list(range(len(trainset)))
        # # Get the indices of all samples in class 0
        # class_drop_indices = [i for i, (image, label) in enumerate(trainset) if label == drop_class]
        # n_samples = int(clip_percentage * len(class_drop_indices))
        # class_drop_indices = class_drop_indices[:n_samples]
        # # Get the indices of all samples in all classes except class 0
        # other_class_indices = list(set(all_indices) - set(class_drop_indices))
        # # Create a sampler that samples from all classes except class 0
        # train_sampler = SubsetRandomSampler(other_class_indices)

        # all_indices = list(range(len(testset)))
        # Get the indices of all samples in class 0
        # class_drop_indices = [i for i, (image, label) in enumerate(testset) if label == drop_class]
        # n_samples = int(clip_percentage * len(class_drop_indices))
        # class_drop_indices = class_drop_indices[:n_samples]
        # # Get the indices of all samples in all classes except class 0
        # other_class_indices = list(set(all_indices) - set(class_drop_indices))
        # # Create a sampler that samples from all classes except class 0
        # test_sampler = SubsetRandomSampler(other_class_indices)

        
        # print(len(testset.data), len(testset.targets))

        
    if get_class is not None:
        indx_train = [i for i, e in enumerate(trainset.targets) if e == get_class]
        trainset.data = list(itemgetter(*indx_train)(trainset.data))
        trainset.targets = list(itemgetter(*indx_train)(trainset.targets))

        indx_test = [i for i, e in enumerate(testset.targets) if e == get_class]
        testset.data = list(itemgetter(*indx_test)(testset.data))
        testset.targets = list(itemgetter(*indx_test)(testset.targets))        

    # print(len(testset.data), len(testset.targets)) 

    # if n_subsets is not None:
    
    # val_loader = DataLoader(testset, batch_size=val_batch_size, shuffle=False)
        

    # train_loader, val_loader = trainset, testset
    if get_loader:
        return train_loader, val_loader, test_loader
    else:
        return trainset, testset

            
if __name__ == '__main__':
    train, val, test = get_data('celeba')

    # print(len(train.dataset), len(test.dataset))
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import scipy.misc
from scipy import io


class TrainDataset(Dataset):
    def __init__(self, dataset_dir, label_dir):
        super(Dataset, self).__init__()
        self.data = scipy.io.loadmat(dataset_dir)["PacketTrainData"]
        self.label = scipy.io.loadmat(label_dir)["TrainLabels"].flatten()
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        image = (torch.from_numpy(self.data[index])).unsqueeze(0).float()
        label = torch.from_numpy(np.asarray(self.label[index]))
        return image, label

    def __len__(self):
        return len(self.data)


class ValDataset(Dataset):
    def __init__(
        self, dataset_dir, label_dir
    ):  # Send any other arguments if it is required
        super(Dataset, self).__init__()
        self.data = scipy.io.loadmat(dataset_dir)["PacketValData"]
        print("Val: ", self.data.shape)
        self.label = scipy.io.loadmat(label_dir)["ValLabels"].flatten()
        print("Val: ", self.label.shape)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        image = (torch.from_numpy(self.data[index])).unsqueeze(0).float()
        label = torch.from_numpy(np.asarray(self.label[index]))
        return image, label

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):
    def __init__(self, dataset_dir, label_dir):
        super(Dataset, self).__init__()
        self.data = scipy.io.loadmat(dataset_dir)["PacketTestData"]
        print("test: ", self.data.shape)
        self.label = scipy.io.loadmat(label_dir)["TestLabels"].flatten()
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        image = (torch.from_numpy(self.data[index])).unsqueeze(0).float()
        label = torch.from_numpy(np.asarray(self.label[index]))
        return image, label

    def __len__(self):
        return len(self.data)


def val_data(batch_size=32, shuffle=True, drop_last=True, num_workers=0):
    data_path = "./rfml/data/rf1024_data/valData.mat"
    label_path = "./rfml/data/rf1024_data/valLabels.mat"
    dataset = ValDataset(data_path, label_path)
    dataloader_val = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader_val


def train_data(
        data_path="./rfml/data/rf1024_data/trainData.mat",
        label_path="./rfml/data/rf1024_data/trainLabels.mat",
        batch_size=32, shuffle=True, drop_last=True, num_workers=0):
    data_path = data_path
    label_path = label_path
    dataset = TrainDataset(data_path, label_path)
    dataloader_train = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader_train


def test_data(
    batch_size=1, shuffle=True, drop_last=True, num_workers=0
):
    data_path = "./rf1024_data/testData.mat"
    label_path = "./rf1024_data/testLabels.mat"
    dataset = TestDataset(data_path, label_path)
    dataloader_train = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader_train

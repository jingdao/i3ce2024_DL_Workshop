import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import h5py

class_names = ['balcony', 'beam', 'column', 'door', 'fence', 'floor', 'roof', 'stairs', 'wall', 'window']

def loadFromH5(filename):
    f = h5py.File(filename, 'r')
    points = f['data'][:]
    for i in range(len(points)):
        points[i,:,:] =  np.array(points[i,:,:])
    labels = np.array(f['label'][:])
    return points, labels

class ClassificationDataset(data.Dataset):
    def __init__(self, filepath, N=2048):
        self.N = N
        self.points, self.labels = loadFromH5(filepath)
        print('Created dataset from %s with %d samples' % (filepath, len(self.labels))) 

    def __getitem__(self, index):
        pc = self.points[index].astype(np.float32)
        resample_idx = np.random.choice(len(pc), self.N, replace=len(pc)<self.N)
        pc = torch.from_numpy(pc[resample_idx].T)
        cls = self.labels[int(index)]
        return pc, cls

    def __len__(self):
        return len(self.labels)

if __name__=='__main__':
    train_dataset = ClassificationDataset(filepath='data/modelnet10_train.h5', N=2048)
    test_dataset = ClassificationDataset(filepath='data/modelnet10_test.h5', N=2048)

    pc, cls = train_dataset[0]
    print(pc, cls)

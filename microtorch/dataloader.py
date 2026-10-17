import random

class DataLoader:
    def __init__(self,data,batch_size=32,shuffle= True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0,len(indices),self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch = [self.data[i] for i in batch_indices]
            inputs = [item[0] for item in batch]
            targets = [item[1] for item in batch]
            yield inputs,targets
    def __len__(self):
        return (len(self.data) + self.batch_size-1) // self.batch_size
    
import numpy as np
from torchnet.dataset import ListDataset, ConcatDataset
import torch
from tqdm import tqdm
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
batch_size=128
train_matches = 'm50_50000_50000_0.txt'
test_matches = "m50_1000_1000_0.txt"
train_datapath = "./train/notredame/data.npy"
test_datapath = "./train/yosemite/data.npy"


def get_train_iter():
    return get_iter(train_datapath, train_matches)


def get_test_iter():
    return get_iter(test_datapath, test_matches, False)


def get_testtrain_iter():
    return get_iter(train_datapath, test_matches, False)


def get_iter(datapath, matches, train=True):
    data = np.load(datapath, allow_pickle=True).item()
    patches = data['patches']
    info = data['info']
    match_data = data['match_data']

    def get_iterator(dataset, batch_size, nthread):
        def get_list_dataset(pair_type):
            ele_list = dataset[pair_type][:len(dataset[pair_type]) // batch_size * batch_size]

            def load(idx):
                o = {'input': np.stack((dataset['patches'][v].astype(np.float32)
                                        - dataset['mean'][v]) / 256.0 for v in idx),
                     'target': 1 if pair_type == 'matches' else -1}
                o['input'] = torch.from_numpy(o['input'])
                o['target'] = torch.LongTensor([o['target']])
                o['input'] = o['input'].float()
                o['target'] = o['target'].float()
                o['input'] = o['input'].cuda()
                o['target'] = o['target'].cuda()
                return o

            ele_list = list(map(load, ele_list))
            ds = ListDataset(elem_list=ele_list)
            # ds = ds.transform({'input': torch.from_numpy, 'target': lambda x: torch.LongTensor([x])})
            # ds = ds.transform({'input': torch.from_numpy, 'target': lambda x: torch.Cudaha([x])})

            return ds.batch(policy='include-last', batchsize=batch_size // 2)

        concat = ConcatDataset([get_list_dataset('matches'),
                                get_list_dataset('nonmatches')])

        return concat.parallel(batch_size=2, shuffle=train, num_workers=nthread)

    def load_provider():
        print('Loading test data')

        p = data

        for i, t in enumerate(['matches', 'nonmatches']):
            p[t] = p['match_data'][matches][i]

        return p

    test_iter = get_iterator(load_provider(), batch_size, 0)
    return test_iter


if __name__ == "__main__":

    test_iter = get_train_iter()


    def cast(t):
        return t.cuda() if torch.cuda.is_available() else t


    def create_variables(sample):
        inputs = Variable(cast(sample['input'].float().view(-1, 2, 64, 64)))
        targets = Variable(cast(sample['target'].float().view(-1)))
        return inputs, targets


    for sample in tqdm(test_iter, dynamic_ncols=True):
        inputs, targets = create_variables(sample)
        # inputs=inputs.cpu().numpy()
        # targets=targets.cpu().numpy()
        # print(targets)
        # for two_batch in inputs:
        #     batch1,batch2=two_batch
        #     plt.subplot(121)
        #     plt.imshow(batch1,cmap='gray')
        #     plt.subplot(122)
        #     plt.imshow(batch2,cmap='gray')
        #     plt.show()

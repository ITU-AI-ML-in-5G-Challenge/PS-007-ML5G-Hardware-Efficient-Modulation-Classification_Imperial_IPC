from torch.utils.data import Dataset
import h5py
import numpy as np
import torch


class radioml_18_dataset(Dataset):
    def __init__(self, dataset_path):
        super(radioml_18_dataset, self).__init__()
        self.open_flag = False
        self.dataset_path = dataset_path
        # do not touch this seed to ensure the prescribed train/test split!
        np.random.seed(2018)
        train_indices = []
        test_indices = []
        for mod in range(0, 24):  # all modulations (0 to 23)
            for snr_idx in range(0, 26):  # all SNRs (0 to 25 = -20dB to +30dB)
                # 'X' holds frames strictly ordered by modulation and SNR
                start_idx = 26 * 4096 * mod + 4096 * snr_idx
                indices_subclass = list(range(start_idx, start_idx + 4096))
                # 90%/10% training/test split, applied evenly for each mod-SNR pair
                split = int(np.ceil(0.1 * 4096))
                np.random.shuffle(indices_subclass)
                train_indices_subclass = indices_subclass[split:]
                test_indices_subclass = indices_subclass[:split]

                # you could train on a subset of the data, e.g. based on the SNR
                # here we use all available training samples
                if snr_idx >= 0:
                    train_indices.extend(train_indices_subclass)
                test_indices.extend(test_indices_subclass)
        self.train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        self.test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    def __getitem__(self, idx):
        # This part of the code is necessary to allow multiple workers, i.e., num_workers>1 in the dataloader settings
        # Otherwise there are some issues with hdf5 multi-access.
        if not self.open_flag:
            h5_file = h5py.File(self.dataset_path, 'r')
            self.data = h5_file['X']
            self.mod = np.argmax(h5_file['Y'], axis=1)  # comes in one-hot encoding
            self.snr = h5_file['Z'][:, 0]
            self.len = self.data.shape[0]
            self.mod_classes = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                                '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', '256QAM',
                                'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
            self.snr_classes = np.arange(-20., 32., 2)  # -20dB to 30dB
            self.open_flag = True
        # transpose frame into Pytorch channels-first format (NCL = -1,2,1024)
        return self.data[idx].transpose(), self.mod[idx], self.snr[idx]

    def __len__(self):
        return self.len

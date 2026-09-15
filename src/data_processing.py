import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class LaneChangeDataset(Dataset):
    def __init__(self, data_folder, seq_length=10):
        self.data_folder = data_folder
        self.seq_length = seq_length
        self.processed_data = self._preprocess_data()

    def _preprocess_data(self):
        processed_data = []
        input_columns = [
            'carCenterXft', 'carCenterYft', 'speedY', 'speed', 'laneId', 'pressure',
            'leadCarCenterXft', 'leadCarCenterYft', 'leadSpeed',
            'leftLeadCarCenterXft', 'leftLeadCarCenterYft', 'leftLeadSpeed',
            'leftFollowCarCenterXft', 'leftFollowCarCenterYft', 'leftFollowSpeed'
        ]
        target_columns = [
            'speedY', 'speed'
        ]

        all_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]

        all_data = []
        for filename in tqdm(all_files, desc="Loading files into memory"):
            file_path = os.path.join(self.data_folder, filename)
            event_data = pd.read_csv(file_path)
            all_data.append(event_data)

        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data.fillna(-1, inplace=True)

        input_scaler = MinMaxScaler()
        input_data = combined_data[input_columns].values
        input_data = input_scaler.fit_transform(input_data)

        target_scaler = MinMaxScaler()
        target_data = combined_data[target_columns].values
        target_data = target_scaler.fit_transform(target_data)

        total_length = len(combined_data)
        for i in range(total_length - self.seq_length):
            input_seq = input_data[i:i + self.seq_length]
            target_seq = target_data[i:i + self.seq_length]

            x_mark_enc = np.zeros(self.seq_length)
            x_mark_dec = np.zeros(10)

            processed_data.append((input_seq, target_seq, x_mark_enc, x_mark_dec))

        return processed_data

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        inputs, targets, x_mark_enc, x_mark_dec = self.processed_data[idx]
        inputs = torch.tensor(inputs, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.float32)
        x_mark_enc = torch.tensor(x_mark_enc, dtype=torch.float32)
        x_mark_dec = torch.tensor(x_mark_dec, dtype=torch.float32)
        return inputs, targets, x_mark_enc, x_mark_dec


def load_and_split_data(data_folder, seq_length=10, batch_size=32):
    dataset = LaneChangeDataset(data_folder, seq_length)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

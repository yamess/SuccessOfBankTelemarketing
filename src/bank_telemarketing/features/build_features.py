from torch.utils.data.dataset import Dataset
import numpy as np
import torch


class CustomDataset(Dataset):
    def __init__(self, emb_cols, x, y):
        super(CustomDataset, self).__init__()
        self.cat = emb_cols
        _emb = x.loc[:, emb_cols]
        self.emb_data = np.stack(
            [c.values for _, c in _emb.items()], axis=1
        ).astype(np.int64)

        # Continuous data
        other_data = x.drop(emb_cols, axis=1)
        self.cont_data = np.stack(
            [c.values for _, c in other_data.items()], axis=1
        ).astype(np.float32)
        self.y = y.values.astype(np.int32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        x_cont = self.cont_data[item]
        x_emb = self.emb_data[item]
        y = np.asarray(self.y[item])

        out = {
            "x_cont": torch.from_numpy(x_cont),
            "x_emb": torch.from_numpy(x_emb),
            "y": torch.tensor(y, dtype=torch.long)
        }
        return out

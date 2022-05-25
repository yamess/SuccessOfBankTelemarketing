import logging
from typing import List
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)


class CreateTensorDataset(Dataset):
    def __init__(self, emb_cols: List[str], x_data, y_data):
        super(CreateTensorDataset, self).__init__()
        self.emb_cols = emb_cols

        # Embedding data
        emb_ = x_data.loc[:, self.emb_cols]
        self.emb_data = np.stack(
            [c.values for _, c in emb_.items()], axis=1
        ).astype(np.int64)

        # Continuous data
        other_cols = emb_cols
        other_data = x_data.drop(other_cols, axis=1)
        self.cont_data = np.stack(
            [c.values for _, c in other_data.items()], axis=1
        ).astype(np.float32)
        self.y = y_data.values.astype(np.int32)

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


class CategoricalEmbeddingSizes:
    """
    Class to select the columns needs to be embedded and find the embeddings sizes of those columns.
    Columns selected are those with more than two categories
    """

    def __init__(self):
        self._cat_cols = None
        self._emb_cols = None
        self._emb_dims = None
        self._emb_info = None

    @property
    def emb_dims(self):
        return self._emb_dims

    @property
    def emb_cols(self):
        return self._emb_cols

    @property
    def emb_info(self):
        return list(zip(self._emb_cols, self._emb_dims))

    def get_emb_cols(self, data):
        emb_cols = [
            c for c in self._cat_cols if len(data[c].unique()) > 2
        ]
        return emb_cols

    def get_cat_emb_dims(self, data, cat_cols: List):
        self._cat_cols = cat_cols
        self._emb_cols = self.get_emb_cols(data=data)
        cat_dims = [len(data[col].unique()) + 1 for col in self._emb_cols]
        self._emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
        return self._emb_dims, self._emb_cols


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, embedding_weights):
        self.embedding_weights_info = embedding_weights

        self.emb_cols = []
        # self.emb_sizes = emb_sizes
        self.embeddings = {}
        self.set_embeddings()
        # self.embeddings.eval()
        # self.emb_pack = list(zip(emb_cols, emb_sizes, self.embeddings))

    def set_embeddings(self):
        for k, v in enumerate(self.embedding_weights_info):
            _emb = nn.Embedding.from_pretrained(v[-1]).eval()
            _col = v[0]
            self.embeddings[_col] = _emb
            self.emb_cols.append(_col)

    def transform(self, data):
        _tmp = data.copy()
        try:
            for col in self.embeddings.keys():
                x = torch.tensor(data[col])
                x = self.embeddings[col](x).numpy()
                _size = self.embeddings[col].weight.size()[1]  # Get embedding output size
                cols = [f"{col}_{k}" for k in range(_size)]
                _tmp[cols] = x
            _tmp.drop(self.emb_cols, axis=1, inplace=True)
            return _tmp

            # for values in self.emb_pack:
            #     col = values[0]
            #     size = values[1]
            #     emb = values[2]
            #     x = torch.tensor(data[col]).to("cuda")
            #     x = emb(x).detach().cpu().numpy()
            #     cols = [f"{col}_{k}" for k in range(size[1])]
            #     _tmp[cols] = x
            # _tmp.drop(self.emb_cols, axis=1, inplace=True)
            # return _tmp
        except Exception as er:
            logger.error(er)
            raise Exception

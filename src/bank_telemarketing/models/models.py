import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, hidden, output_size, continuous_size, embedding_sizes, dropout):
        super(Classifier, self).__init__()

        self.emb_dims = embedding_sizes
        self.cont_dims = continuous_size
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(self.cont_dims)

        # Embedding layers for categorical columns
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(cat, size) for cat, size in self.emb_dims]
        )
        n_emb = sum(e.embedding_dim for e in self.embedding_layers)

        # Linear layers
        self.linear = nn.Sequential(
            nn.Linear(in_features=n_emb + continuous_size, out_features=hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(in_features=hidden, out_features=int(hidden/2)),
            nn.ReLU(),
            nn.BatchNorm1d(int(hidden/2)),
            nn.Linear(in_features=int(hidden/2), out_features=output_size)
        )

        # Initialize the layers weight
        self.embedding_layers.apply(self.init_layers)
        self.linear.apply(self.init_layers)

    @staticmethod
    def init_layers(m):
        if type(m) == nn.Linear or type(m) == nn.Embedding:
            nn.init.kaiming_normal_(m.weight)

    def forward(self, x_cont, x_cat):
        embeddings = [
            self.dropout(f(x_cat[:, i])) for i, f in enumerate(self.embedding_layers)
        ]
        embeddings = torch.cat(embeddings, 1)

        x_cont = self.bn(x_cont)
        x = torch.cat((embeddings, x_cont), 1)
        x = self.linear(x)
        return x

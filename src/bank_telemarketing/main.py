import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch import optim
from torch.nn import BCEWithLogitsLoss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from bank_telemarketing.config import config
from bank_telemarketing.features.build_features import CustomDataset
from bank_telemarketing.models.models import Classifier
from bank_telemarketing.preprocessing.embedding import CategoricalEmbeddingSizes
from bank_telemarketing.preprocessing.preprocess import MultiLabelEncoder, CustomScaler
from bank_telemarketing.train.engine import engine

if __name__ == "__main__":
    data = pd.read_csv("../../data/clean/clean_bank_full.csv")

    x_emb_cols = ["job", "marital", "education", "default", "housing", "loan", "month", "day_of_week", "poutcome"]
    x_bin_col = ["contact", "has_been_contacted"]
    y_col = "subscribed"
    x_numerical = ["age", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                   "euribor3m", "nr.employed"]
    cats = x_emb_cols + x_bin_col
    all_cols = x_emb_cols + x_bin_col + x_numerical

    data[cats] = data[cats].astype("category")

    x_pipe = Pipeline(
        steps=
        [
            ("label_encoder", MultiLabelEncoder(cols=cats)),
            ("scaler", CustomScaler(cols=x_numerical))
        ]
    )
    y_mapping = {"no": 0, "yes": 1}

    y = data.subscribed
    X = data.drop(["subscribed", "duration"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=56)

    # Train data
    y_train = y_train.replace(y_mapping).reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    X_train = x_pipe.fit_transform(X_train)
    train_dataset = CustomDataset(
        emb_cols=x_emb_cols,
        x=X_train,
        y=y_train
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config["TRAIN_BS"]
    )

    # Validation data
    y_test = y_test.replace(y_mapping).reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    X_test = x_pipe.transform(X_test)
    test_dataset = CustomDataset(
        emb_cols=x_emb_cols,
        x=X_test,
        y=y_test
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=config["VALID_BS"]
    )

    # Get embedding size
    emb = CategoricalEmbeddingSizes().get_cat_emb_dims(data=X_train, cat_cols=cats)[0]

    model = Classifier(
        hidden=168,
        output_size=1,
        continuous_size=11,
        embedding_sizes=emb,
        dropout=0.5
    )
    model.to(config["DEVICE"])
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.05)
    pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    criterion.to(config["DEVICE"])

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        patience=10,
    )

    engine(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        eval_dataloader=test_dataloader,
        config=config,
        scheduler=scheduler,
        pos_weight=pos_weight,
        checkpoint=None
    )

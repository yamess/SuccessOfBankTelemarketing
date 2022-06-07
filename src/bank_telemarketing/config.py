import torch

config = {
    "SEED": 64,
    "TRAIN_DATA_PATH": "data/train_data.csv",
    "CHECKPOINT_DIR": "model_registry",
    "LOGS_PATH": "logs/metrics_logs.csv",
    "MODEL_PATH": "models/model.pt",
    "N_EPOCHS": 100,
    "TRAIN_BS": 64,
    "VALID_BS": 1000,
    "CUT_POINT": 0.5,
    "LR": 0.01,
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

import csv
import os
import time
from copy import deepcopy
from typing import Optional

import torch

from bank_telemarketing.train.train_model import train, evaluate


def engine(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        criterion,
        config,
        pos_weight,
        checkpoint_dir: str,
        logs_dir: str,
        checkpoint: Optional[dict] = None,
        scheduler=None,
):
    engine_start_time = time.time()

    if checkpoint:
        best_valid_loss = checkpoint["best_valid_loss"]
        best_f1 = checkpoint["best_f1"]
        best_precision = checkpoint["best_precision"]
        epoch_at_best = checkpoint["epoch_at_best"]
    else:
        checkpoint = {}
        best_valid_loss = 1e10
        best_precision = 0.0
        best_f1 = 0.0
        epoch_at_best = 0

    # checkpoint = {}
    # best_valid_loss = 1e10
    # best_precision = 0.0
    # best_f1 = 0.0
    # epoch_at_best = 0

    print("======================= Training Started ============================")

    for e in range(1 + epoch_at_best, config["N_EPOCHS"] + epoch_at_best):
        e_start_time = time.time()

        metrics_train = train(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=config["DEVICE"],
            cut_point=config["CUT_POINT"],
            pos_weight=pos_weight,
        )
        metrics_valid = evaluate(
            model=model,
            dataloader=eval_dataloader,
            criterion=criterion,
            device=config["DEVICE"],
            cut_point=config["CUT_POINT"],
            pos_weight=pos_weight,
        )

        if scheduler:
            scheduler.step(metrics_valid["loss"])

        e_end_time = time.time()
        e_elapsed_time = e_end_time - e_start_time

        display_msg = (
            f"Epoch: {e: <{4}} | Elapsed Time: {e_elapsed_time: 3.2f} s | Train Loss: {metrics_train['loss']: .4f} | "
            f"Valid Loss: {metrics_valid['loss']: .4f} | Train F1: {metrics_train['f1']: .4f} | "
            f"Valid F1: {metrics_valid['f1']: .4f} | Train Precision: {metrics_train['precision']: .4f} | "
            f"Valid Precision: {metrics_valid['precision']: .4f} | "
        )

        if metrics_valid["precision"] > best_precision:
            best_valid_loss = metrics_valid["loss"]
            best_f1 = metrics_valid["f1"]
            best_precision = metrics_valid["precision"]
            best_state_dict = deepcopy(model.state_dict())

            display_msg += " + "

            checkpoint["epoch_at_best"] = e
            checkpoint["best_valid_loss"] = best_valid_loss
            checkpoint['best_f1'] = best_f1
            checkpoint["best_precision"] = best_precision
            checkpoint["best_state_dict"] = best_state_dict

            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{str(best_precision).replace('.', '')}.pt")
            torch.save(checkpoint, checkpoint_path)

        logs_path = os.path.join(logs_dir, "metrics.csv")
        with open(logs_path, "a") as f:
            csv_writer = csv.DictWriter(
                f, fieldnames=["epoch", "loss", "f1", "precision"]
            )
            info = {
                "epoch": e,
                "loss": metrics_valid["loss"],
                "f1": metrics_valid["f1"],
                "precision": metrics_valid["precision"],
            }
            csv_writer.writerow(info)

        print(display_msg)

    engine_end_time = time.time()
    total_time = engine_end_time - engine_start_time
    print(f"Total Time elapsed: {total_time: .4f}")
    print("======================== End of Training ===================")
    print(" *********************** SUMMARY FOR VALIDATION ***********************")
    print(f"  Best Model loss: {checkpoint['best_valid_loss']}")
    print(f"  Best Model F1 Score: {checkpoint['best_f1']}")
    print(f"  Best Model Precision: {checkpoint['best_precision']}")
    print(" *********************************************************************")

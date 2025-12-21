import os
import pandas as pd

import torch
from torch.utils.data import DataLoader

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from scripts.args import parse_args

from scripts.train_funcs import (
    ForCausalLMLoss_fixed,
    HTRDataset,
    run_experiment,
)

def main():
    args = parse_args()
    print(args)

    SLICE_DATASET_DIR = args.slice_dataset_dir
    EXPERIMENTS_DIR = args.experiments_dir
    BATCH_SIZE = args.batch_size
    EVAL_BATCH_SIZE = args.eval_batch_size
    WORKERS = 0
    DEVICE = torch.DEVICE("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    processor = TrOCRProcessor.from_pretrained("Kansallisarkisto_cyrillic-htr-model/processor", use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained("Kansallisarkisto_cyrillic-htr-model")
    model = model.to(DEVICE)

    model.loss_function
    model.loss_function = ForCausalLMLoss_fixed

    dataset_df = pd.read_csv('samples.csv')
    print('Объём датасета:', len(dataset_df))

    train_df = dataset_df[dataset_df['sample'] == 'train'].reset_index(drop=True)
    val_df = dataset_df[dataset_df['sample'] == 'val'].reset_index(drop=True)
    test_df = dataset_df[dataset_df['sample'] == 'test'].reset_index(drop=True)

    train = HTRDataset(root_dir=SLICE_DATASET_DIR, df=train_df, processor=processor)
    val = HTRDataset(root_dir=SLICE_DATASET_DIR, df=val_df, processor=processor)
    test = HTRDataset(root_dir=SLICE_DATASET_DIR, df=test_df, processor=processor)
    debug = HTRDataset(root_dir=SLICE_DATASET_DIR, df=train_df.head(10), processor=processor)

    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(DEVICE.type=="cuda"), num_workers=WORKERS)
    val_dataloader = DataLoader(val, batch_size=EVAL_BATCH_SIZE, pin_memory=(DEVICE.type=="cuda"), num_workers=WORKERS)
    test_dataloader = DataLoader(test, batch_size=EVAL_BATCH_SIZE, pin_memory=(DEVICE.type=="cuda"), num_workers=WORKERS)
    debug_dataloader = DataLoader(debug, batch_size=EVAL_BATCH_SIZE, pin_memory=(DEVICE.type=="cuda"), num_workers=WORKERS)

    run_experiment(
        model=model,
        processor=processor,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        debug_loader=debug_dataloader,
        lr=args.learning_rate,
        epochs=args.epochs,
        freeze_encoder=args.freeze_encoder,
        freeze_decoder=args.freeze_decoder,
        device=DEVICE,
        debug=False,
    )

if __name__ == '__main__':
    main()
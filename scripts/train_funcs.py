import gc
import pandas as pd
import datetime
import os
from tqdm import tqdm
import editdistance
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import get_linear_schedule_with_warmup
from transformers.loss.loss_utils import fixed_cross_entropy


class HTRDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        # self.cache = [None for _ in range(len(df))]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        # if self.cache[idx]:
        #     return self.cache[idx]
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(os.path.join(self.root_dir, file_name)).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length
        ).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {
            "pixel_values": pixel_values.squeeze(),
            "labels": torch.tensor(labels),
            "texts": text,
        }
        return encoding


def ForCausalLMLoss_fixed(
    logits,
    labels,
    vocab_size,
    num_items_in_batch = None,
    ignore_index = -100,
    shift_labels = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        # labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels.contiguous() # FIX OLD LOSS

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


def compute_cer(preds, labels):
    total_dist = 0
    total_len = 0

    for p, l in zip(preds, labels):
        total_dist += editdistance.eval(p, l)
        total_len += len(l)

    return total_dist, total_len


def train_epoch(
    model, 
    loader, 
    optimizer, 
    scheduler, 
    device, 
    scaler=None
):
    model.train()
    train_cer = 0.0
    train_loss = 0.0
    tokens_cnt = 0.0
    for batch in tqdm(loader):
        optimizer.zero_grad(set_to_none=True)
        if scaler:
            with torch.autocast("cuda"):
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device, non_blocking=True),
                    labels=batch["labels"].to(device, non_blocking=True)
                )
                loss = outputs.loss
        
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
        else:
            outputs = model(
                pixel_values=batch["pixel_values"].to(device, non_blocking=True),
                labels=batch["labels"].to(device, non_blocking=True)
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
        batch_tokens_cnt = sum([
            len(batch['labels'][idx][batch['labels'][idx] != -100]) - 1
            for idx in range(batch['labels'].size(0))
        ])
        train_loss += loss.item() * batch_tokens_cnt
        tokens_cnt += batch_tokens_cnt
    train_loss = train_loss / tokens_cnt

    print(f"{'Train':10} Loss: {round(train_loss, 5):15}")#CER: {train_cer:20}")

    return train_loss, train_cer


def eval_epoch(
    model, 
    processor, 
    loader, 
    device, 
):
    model.eval()
    total_dist = 0.0
    total_len = 0.0
    total_loss = 0.0
    tokens_cnt = 0.0
    with torch.no_grad():
        for batch in tqdm(loader):
            # run batch generation
            for k, v in batch.items():
                if k not in ['texts']:
                    batch[k] = v.to(device)

            batch_tokens_cnt = (batch["labels"] != -100).sum().item()
            total_loss += model(
                pixel_values=batch["pixel_values"].to(device, non_blocking=True),
                labels=batch["labels"].to(device, non_blocking=True)
            ).loss.item() * batch_tokens_cnt
            tokens_cnt += batch_tokens_cnt

            with torch.autocast("cuda"):
                outputs = model.generate(
                    batch["pixel_values"],
                )

            pred_str = processor.batch_decode(outputs, skip_special_tokens=True)

            # labels = batch["labels"].clone()
            # labels[labels == -100] = processor.tokenizer.pad_token_id
            # label_str = processor.batch_decode(labels, skip_special_tokens=True)
            label_str = batch["texts"]
            
            dist, length = compute_cer(pred_str, label_str)
            total_dist += dist
            total_len += length

    valid_cer = total_dist / total_len
    valid_loss = total_loss / tokens_cnt

    print(f"{'Valid':10}Loss: {round(valid_loss, 5):15} CER: {round(valid_cer, 5):15}")

    return valid_loss, valid_cer


def run_experiment(
    model,
    processor,
    experiments_dir,
    train_loader,
    val_loader,
    lr=5e-5,
    epochs=30,
    freeze_encoder=True,
    freeze_decoder=True,
    device=None,
    debug=False,
    debug_loader=None
):
    if debug:
        train_loader = debug_loader
        val_loader = debug_loader

    if freeze_encoder:
        print('encoder locked')
        for param in model.encoder.parameters():
            param.requires_grad = False
    if freeze_decoder:
        print('decoder locked')
        for param in model.decoder.parameters():
            param.requires_grad = False

    dt_now = str(datetime.datetime.now())[:-7].replace(' ', '_').replace(':', '-')
    experiment_dir = os.path.join(experiments_dir, dt_now)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs('experiments_hist', exist_ok=True)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    scaler = None # torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # логирование метрик
    history = {
        "epoch": [],
        "train_loss": [],
        # "train_metric": [],
        "val_loss": [],
        "val_metric": [],
        "lr": [],
    }

    best_cer = 1
    for epoch in range(1, epochs + 1):
        
        train_loss, _ = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
        )
        
        val_loss, val_metric = eval_epoch(
            model=model,
            processor=processor,
            loader=val_loader,
            device=device,
        )
        
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        # history["train_metric"].append(train_metric)
        history["val_metric"].append(val_metric)
        history["epoch"].append(epoch)
        history["lr"].append(current_lr)
        

        print(f"Epoch {epoch:02d} | "
              f"Train loss {train_loss:.4f} | " # CER {train_metric:.4f}
              f"Val loss {val_loss:.4f} CER {val_metric:.4f} | "
              f"lr {current_lr:.2e}"
        )
        print()
        # model_save_path = os.path.join(experiment_dir, f"model_epoch_{epoch}.pt")
        
        if val_metric < best_cer and not debug:
            best_model_save_path = os.path.join(experiment_dir, f"best_model.pt")
            best_cer = val_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scaler_state_dict': scaler.state_dict(),
                'loss': train_loss,
            }, best_model_save_path)
            
        if not debug:
            history_df = pd.DataFrame(history)
            history_df.sort_values(by='epoch').to_excel(f'./experiments_hist/history_{dt_now}.xlsx', index=False)
        
        torch.cuda.empty_cache()
        gc.collect()

    return history, model

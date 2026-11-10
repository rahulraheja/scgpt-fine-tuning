import copy
import logging
import time
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from typing import Dict

from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerGenerator
from scgpt.utils import map_raw_id_to_vocab_id


def train_one_epoch(model, train_loader, optimizer, scaler, config,
                    gene_ids, n_genes, epoch, logger: logging.Logger) -> float:
    model.train()
    total_loss = 0.0
    device = config.device
    num_batches = len(train_loader)

    for batch_idx, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data = batch_data.to(device)

        x = batch_data.x
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        pert_flags = x[:, 1].long().view(batch_size, n_genes)
        target_gene_values = batch_data.y

        if config.include_zero_gene == "all":
            input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
        else:
            input_gene_ids = (
                ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
            )

        if len(input_gene_ids) > config.max_seq_len:
            input_gene_ids = torch.randperm(
                len(input_gene_ids), device=device
            )[: config.max_seq_len]

        input_values = ori_gene_values[:, input_gene_ids]
        input_pert_flags = pert_flags[:, input_gene_ids]
        target_values = target_gene_values[:, input_gene_ids]

        mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
        src_key_padding_mask = torch.zeros_like(
            input_values, dtype=torch.bool, device=device
        )

        with torch.amp.autocast(device_type=device.type, enabled=config.amp):
            output_dict = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
                src_key_padding_mask=src_key_padding_mask,
                CLS=config.CLS,
                CCE=config.CCE,
                MVC=config.MVC,
                ECS=config.ECS,
            )
            output_values = output_dict["mlm_output"]
            masked_positions = torch.ones_like(input_values, dtype=torch.bool)
            loss = masked_mse_loss(output_values, target_values, masked_positions)

        model.zero_grad()
        if config.amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        if batch_idx % config.log_interval == 0 and batch_idx > 0:
            logger.info(
                f"| epoch {epoch:3d} | {batch_idx:3d}/{num_batches:3d} batches "
                f"| loss {total_loss / (batch_idx + 1):.4f}"
            )

    return total_loss / num_batches


def eval_perturb(loader: DataLoader, model: TransformerGenerator,
                 device, config, gene_ids) -> Dict:
    model.eval()
    pert_cat, pred, truth, pred_de, truth_de = [], [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        pert_cat.extend(batch.pert)
        with torch.no_grad():
            p = model.pred_perturb(
                batch,
                include_zero_gene=config.include_zero_gene,
                gene_ids=gene_ids,
                amp=config.amp,
            )
            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())
            for i, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[i, de_idx])
                truth_de.append(t[i, de_idx])

    return {
        "pert_cat": np.array(pert_cat),
        "pred": torch.stack(pred).numpy().astype(np.float32),
        "truth": torch.stack(truth).numpy().astype(np.float32),
        "pred_de": torch.stack(pred_de).numpy().astype(np.float32),
        "truth_de": torch.stack(truth_de).numpy().astype(np.float32),
    }


def run_training(model, pert_data, config, gene_ids, n_genes,
                 logger: logging.Logger, save_dir):
    device = config.device
    model = model.to(device)
    logger.info(f"Training on device: {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config.schedule_interval, gamma=config.schedule_gamma
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

    best_val_loss = np.inf
    best_model = copy.deepcopy(model)
    patience = 0

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model,
            pert_data.dataloader["train_loader"],
            optimizer,
            scaler,
            config,
            gene_ids,
            n_genes,
            epoch,
            logger,
        )

        val_res = eval_perturb(
            pert_data.dataloader["val_loader"], model, device, config, gene_ids
        )
        # Use mean MSE over val set as early-stopping signal (simpler than pearson mid-training)
        val_mse = np.mean((val_res["pred"] - val_res["truth"]) ** 2)

        elapsed = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch:3d} | train_loss {train_loss:.4f} "
            f"| val_mse {val_mse:.4f} | {elapsed:.0f}s"
        )

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), save_dir / "best_model.pt")
            logger.info(f"  -> New best val_mse {best_val_loss:.4f}, checkpoint saved")
            patience = 0
        else:
            patience += 1
            logger.info(f"  -> No improvement ({patience}/{config.early_stop})")
            if patience >= config.early_stop:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        scheduler.step()

    return best_model

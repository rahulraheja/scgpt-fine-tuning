from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import torch


@dataclass
class Config:
    # Paths
    data_dir: str = "./data"
    pretrained_dir: str = "./pretrained/scGPT_human"
    save_dir: str = "./outputs/runs"

    # Data
    data_name: str = "adamson"
    split: str = "simulation"

    # Tokenization
    pad_token: str = "<pad>"
    special_tokens: Tuple[str, ...] = ("<pad>", "<cls>", "<eoc>")
    pad_value: int = 0
    pert_pad_id: int = 2
    include_zero_gene: str = "all"
    # 512 fits comfortably on CPU; increase to 1536 on GPU
    max_seq_len: int = 512

    # Task heads (all off for perturbation-only task)
    CLS: bool = False
    CCE: bool = False
    MVC: bool = False
    ECS: bool = False

    # Architecture — must match pretrained checkpoint args.json
    embsize: int = 512
    d_hid: int = 512
    nlayers: int = 12
    nhead: int = 8
    n_layers_cls: int = 3
    dropout: float = 0.0

    # Mac: False (no flash-attn without CUDA). Set True on NVIDIA GPU after installing flash-attn.
    use_fast_transformer: bool = False

    # Training
    lr: float = 1e-4
    batch_size: int = 64
    eval_batch_size: int = 64
    epochs: int = 15
    schedule_interval: int = 1
    schedule_gamma: float = 0.9
    early_stop: int = 10
    log_interval: int = 100

    # Mac/CPU: False. Set True on NVIDIA GPU with CUDA.
    amp: bool = False

    # Whether to force CPU even when MPS is available.
    # MPS cannot handle the large attention buffers needed for seq_len=512+,
    # so we default to CPU on Mac. Set False only after verifying MPS compatibility.
    force_cpu: bool = True

    # Only load these pretrained layer prefixes; perturbation head trains from scratch
    load_param_prefixs: Tuple[str, ...] = (
        "encoder",
        "value_encoder",
        "transformer_encoder",
    )

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (not self.force_cpu
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()):
            return torch.device("mps")
        return torch.device("cpu")

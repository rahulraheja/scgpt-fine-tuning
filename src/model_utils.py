import numpy as np
import torch
from pathlib import Path

from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab


def load_vocab_and_gene_ids(pretrained_dir: str, genes: list, special_tokens: tuple):
    """Load scGPT vocab and map dataset genes to vocab token IDs."""
    vocab = GeneVocab.from_file(Path(pretrained_dir) / "vocab.json")

    for tok in special_tokens:
        if tok not in vocab:
            vocab.append_token(tok)

    gene_ids = np.array(
        [vocab[g] if g in vocab else vocab["<pad>"] for g in genes],
        dtype=int,
    )
    n_in_vocab = sum(1 for g in genes if g in vocab)
    print(f"Genes: {len(genes)} total, {n_in_vocab} in scGPT vocab, "
          f"{len(genes) - n_in_vocab} mapped to <pad>")
    return vocab, gene_ids


def build_model(config, vocab) -> TransformerGenerator:
    model = TransformerGenerator(
        len(vocab),
        config.embsize,
        config.nhead,
        config.d_hid,
        config.nlayers,
        nlayers_cls=config.n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=config.dropout,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        pert_pad_id=config.pert_pad_id,
        do_mvc=config.MVC,
        use_fast_transformer=config.use_fast_transformer,
    )
    return model


def load_pretrained_weights(model: TransformerGenerator,
                            pretrained_dir: str,
                            load_param_prefixs: tuple) -> int:
    """Load matching prefix layers from checkpoint. Returns count of loaded tensors."""
    model_dict = model.state_dict()
    pretrained = torch.load(
        Path(pretrained_dir) / "best_model.pt", map_location="cpu"
    )

    to_load = {
        k: v for k, v in pretrained.items()
        if any(k.startswith(p) for p in load_param_prefixs)
        and k in model_dict
        and v.shape == model_dict[k].shape
    }

    model_dict.update(to_load)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(to_load)} pretrained parameter tensors "
          f"(out of {len(pretrained)} in checkpoint)")
    return len(to_load)

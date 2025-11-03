from collections.abc import Callable

import torch


def load_text(path: str) -> str:
    with open(path) as f:
        return f.read()


def build_vocab(text: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, stoi, itos


def encode(string: str, char_to_id: dict[str, int]) -> list[int]:
    return [char_to_id[ch] for ch in string]


def decode(token_ids: list[int], id_to_char: dict[int, str]) -> str:
    return "".join([id_to_char[i] for i in token_ids])


def prepare_datasets(text: str, train_ratio: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    # NOTE: We should encode using vocab; caller should pass encoded data instead.
    # Left here for compatibility if text is already in characters.
    n = int(train_ratio * len(data))
    return data[:n], data[n:]


def prepare_datasets_encoded(
    encoded_ids: list[int], train_ratio: float = 0.9
) -> tuple[torch.Tensor, torch.Tensor]:
    data = torch.tensor(encoded_ids, dtype=torch.long)
    n = int(train_ratio * len(data))
    return data[:n], data[n:]


def make_batcher(
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    *,
    batch_size: int,
    context_length: int,
    device: str,
) -> Callable[[str], tuple[torch.Tensor, torch.Tensor]]:
    def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - context_length, (batch_size,))
        x = torch.stack([data[i : i + context_length] for i in ix])
        y = torch.stack([data[i + 1 : i + context_length + 1] for i in ix])
        return x.to(device), y.to(device)

    return get_batch

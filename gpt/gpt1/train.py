from collections.abc import Callable

from model import BigramLanguageModel
import torch


@torch.no_grad()
def estimate_loss(
    model: BigramLanguageModel,
    get_batch: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
    eval_iters: int,
) -> dict[str, torch.Tensor]:
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(
    model: BigramLanguageModel,
    optimizer: torch.optim.Optimizer,
    get_batch: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
    train_iters: int,
    eval_iters: int,
    log_every: int | None = None,
) -> None:
    if log_every is None:
        log_every = max(1, train_iters // 10)
    for steps in range(train_iters):
        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if steps % log_every == 0:
            losses = estimate_loss(model, get_batch, eval_iters)
            print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

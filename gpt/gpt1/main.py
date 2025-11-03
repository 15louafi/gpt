from config import *
from data_utils import (
    build_vocab,
    decode,
    encode,
    load_text,
    make_batcher,
    prepare_datasets_encoded,
)
from model import BigramLanguageModel
import torch
from train import train as run_train

text = load_text(INPUT_PATH)
chars, char_to_id, id_to_char = build_vocab(text)
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)

encoded = encode(text, char_to_id)
train_data, val_data = prepare_datasets_encoded(encoded, train_ratio=0.9)

batcher = make_batcher(
    train_data, val_data, batch_size=BATCH_SIZE, context_length=CONTEXT_LENGTH, device=device
)

model = BigramLanguageModel(vocab_size)
model.to(device)
print(model)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
run_train(model, optimizer, batcher, TRAIN_ITERS, EVAL_ITERS)


idx = torch.zeros((1, 1), dtype=torch.long, device=device)
out_ids = model.generate(idx, max_new_tokens=200)[0].tolist()
print(decode(out_ids, id_to_char=id_to_char))

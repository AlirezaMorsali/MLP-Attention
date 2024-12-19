import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from model import GPTLanguageModel
from utils import device, estimate_loss, get_batch, load_data

if __name__ == "__main__":
    # Hyperparameters
    experiment_name = "Final"
    result_path = f"results/{experiment_name}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    batch_size = 64
    block_size = 256
    max_iters = 1000
    eval_interval = 50
    learning_rate = 0.0003
    eval_iters = 200
    n_embd = 384
    n_head = 3
    n_layer = 3
    n_hidden_layers = 1
    dropout = 0.2
    hidden_size = block_size

    # Load data
    text, chars, vocab_size, stoi, itos, data = load_data("input.txt")
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Initialize models
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        mlp_attention=False,
    ).to(device)

    mlp_attention_model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        mlp_attention=True,
    ).to(device)

    original_params = sum(p.numel() for p in model.parameters()) / 1e6
    mlp_attention_params = (
        sum(p.numel() for p in mlp_attention_model.parameters()) / 1e6
    )

    print(f"Original Model: {original_params} M parameters")
    print(f"MLP Attention Model: {mlp_attention_params} M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    mlp_attention_optimizer = torch.optim.AdamW(
        mlp_attention_model.parameters(), lr=learning_rate
    )

    train_loss = []
    val_loss = []
    mlp_attention_train_loss = []
    mlp_attention_val_loss = []
    x_val = []

    for iter in range(max_iters):
        # Evaluate loss periodically
        if iter % eval_interval == 0 or iter == max_iters - 1:
            x_val.append(iter)
            losses = estimate_loss(
                model, train_data, val_data, eval_iters, batch_size, block_size
            )
            train_loss.append(losses["train"])
            val_loss.append(losses["val"])
            print(
                f"Original model: step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

            losses = estimate_loss(
                mlp_attention_model,
                train_data,
                val_data,
                eval_iters,
                batch_size,
                block_size,
            )
            mlp_attention_train_loss.append(losses["train"])
            mlp_attention_val_loss.append(losses["val"])
            print(
                f"MLP Attention model: step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # Get batch
        xb, yb = get_batch("train", train_data, val_data, batch_size, block_size)

        # Original model update
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # MLP Attention model update
        mlp_logits, mlp_loss = mlp_attention_model(xb, yb)
        mlp_attention_optimizer.zero_grad(set_to_none=True)
        mlp_loss.backward()
        mlp_attention_optimizer.step()

    # Save hyperparameters and results
    hyper_params = {
        "batch_size": batch_size,
        "block_size": block_size,
        "max_iters": max_iters,
        "eval_interval": eval_interval,
        "learning_rate": learning_rate,
        "eval_iters": eval_iters,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
        "n_hidden_layers": n_hidden_layers,
        "hidden_size": block_size,
        "original_params_million": original_params,
        "mlp_attention_params_million": mlp_attention_params,
        "train_loss": [val.tolist() for val in train_loss],
        "val_loss": [val.tolist() for val in val_loss],
        "mlp_attention_train_loss": [val.tolist() for val in mlp_attention_train_loss],
        "mlp_attention_val_loss": [val.tolist() for val in mlp_attention_val_loss],
        "epochs": x_val,
    }

    with open(f"{result_path}/hyper_params.json", "w") as outfile:
        json.dump(hyper_params, outfile, indent=4)

    with open(f"{result_path}/losses.npy", "wb") as f:
        np.save(f, np.array(train_loss))
        np.save(f, np.array(val_loss))
        np.save(f, np.array(mlp_attention_train_loss))
        np.save(f, np.array(mlp_attention_val_loss))

    plt.plot(x_val[1:], train_loss[1:], label="Training Loss")
    plt.plot(x_val[1:], val_loss[1:], label="Validation Loss")
    plt.plot(x_val[1:], mlp_attention_train_loss[1:], label="Modded Training Loss")
    plt.plot(x_val[1:], mlp_attention_val_loss[1:], label="Modded Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(f"{result_path}/losses.png")
    plt.show()

    # Optionally save the trained model weights
    torch.save(model.state_dict(), f"{result_path}/original_model.pt")
    torch.save(
        mlp_attention_model.state_dict(), f"{result_path}/mlp_attention_model.pt"
    )

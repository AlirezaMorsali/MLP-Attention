
"""
This code draws strong inspiration and borrows heavily from the implementation available at https://github.com/facebookresearch/xformers/blob/main/xformers/benchmarks/benchmark.py.
"""

import argparse
import torch
import json
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from contextlib import suppress
from torch.autograd.profiler import record_function
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from typing import Any, Dict, List, Optional
from model import GPTLanguageModel


_use_cuda = torch.cuda.is_available()

# To download the tinyshakespeare run the following line in terminal
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]  # encoder: take a string, output a list of integers


def decode(l):
    return "".join(
        [itos[i] for i in l]
    )  # decoder: take a list of integers, output a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split, block_size, batch_size, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def _get_trace_handler(name: str):
    def trace_handler(prof):
        prof.export_chrome_trace(f"profile_{name}.json")
        prof.export_stacks(f"stacks_{name}.txt", "self_cuda_time_total")

    return trace_handler


def _train_for_several_steps(
    block: GPTLanguageModel,
    num_steps: int,
    batch_size: int,
    sequence_length: int,
    embed_dim: int,
    autocast: bool,
    device: torch.device,
    lr: float = 0.0003,
    norm_type: Optional[float] = None,
    profile: bool = False,
    att_name: str = "",
    backward: bool = True,
) -> Dict[str, float]:
    # use SGD with momentum instead of Adam, since Adam is scale invariant
    # and this makes it bad for tests
    optim = torch.optim.SGD(block.parameters(), lr=lr, momentum=0.9)

    if _use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.time()

    # Optional profiler, requires a context and some setup
    profiler = (
        torch.profiler.profile(  # type: ignore
            activities=[
                torch.profiler.ProfilerActivity.CPU,  # type: ignore
                torch.profiler.ProfilerActivity.CUDA,  # type: ignore
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),  # type: ignore
            on_trace_ready=_get_trace_handler(
                f"{att_name}_batch_{batch_size}_seq_{sequence_length}_embed_dim_{embed_dim}"
            ),
            profile_memory=True,
            with_stack=True,
        )
        if profile
        else suppress()
    )

    # Actual vanilla training loop
    # - nonsensical data, but remove that from the compute time
    # inputs = torch.rand(batch_size, sequence_length).to(device)
    # inputs = torch.randint(low=1, high=vocab_size, size=(batch_size, sequence_length), dtype=torch.int).to(device)
    # xb, yb = get_batch("train")
    xb, yb = get_batch("train", sequence_length, batch_size, device)



    with profiler as p:  # type: ignore
        for _ in range(num_steps):
            optim.zero_grad()

            with torch.cuda.amp.autocast(enabled=autocast):
                with record_function("attention_forward"):
                    _, loss = block(xb, yb)
                    # print(f'shape: {output.shape}')

                # with record_function("loss"):
                #     loss = F.mse_loss(
                #         inputs.unsqueeze(-1).repeat(1, 1, output.shape[-1]),
                #         output,
                #         reduction="sum",
                #     )
            if (backward):
                with record_function("backward"):
                    loss.backward()

                if norm_type is not None:
                    clip_norm = 0.3
                    torch.nn.utils.clip_grad_norm_(block.parameters(), clip_norm, norm_type)
                optim.step()

            if p:
                p.step()

    if _use_cuda:
        torch.cuda.synchronize()
        max_memory = torch.cuda.max_memory_allocated() / 2**20
    else:
        max_memory = -1
    run_time = time.time() - start_time

    return {"run_time": run_time, "max_memory": round(max_memory, 1)}



def benchmark_model(num_warmup: int, num_steps: int, **kwargs) -> Dict[str, float]:
    # Run warm-up first
    warm_up_args = {**kwargs}
    warm_up_args["profile"] = False
    _train_for_several_steps(num_steps=num_warmup, **warm_up_args)

    return _train_for_several_steps(num_steps=num_steps, **kwargs)

def test_gpt_block(
    attention_name: str,
    heads: int,
    autocast: bool,
    batch_size: int,
    sequence_length: int,
    embed_dim: int,
    n_layer:int,
    dropout: float,
    num_steps: int,
    num_warmup: int,
    device: torch.device,
    profile: bool,
) -> Dict[str, float]:



    # # mlp_attention_model = GPTLanguageModel(mlp_attention=True)
    if attention_name == 'mlp':
      
      mlp_attention_model = GPTLanguageModel(n_embd=embed_dim, n_head=heads, n_layer=n_layer, block_size=sequence_length, hidden_size=sequence_length,  vocab_size=vocab_size, mlp_attention=True, dropout=dropout, device=device)
      block = mlp_attention_model.to(device)
    else:
      model = GPTLanguageModel(n_embd=embed_dim, n_head=heads, n_layer=n_layer, block_size=sequence_length, hidden_size=sequence_length,  vocab_size=vocab_size, mlp_attention=False, dropout=dropout, device=device)
      block = model.to(device)
    
    model_params = sum(p.numel() for p in block.parameters()) / 1e6
    # print the number of parameters in the model
    print(f"{attention_name} Model: {model_params} M parameters")
    
    print(
        "Testing:",
        block,
        batch_size,
        sequence_length,
        embed_dim,
        autocast,
        device,
        attention_name,
    )



    return benchmark_model(
        num_steps=num_steps,
        num_warmup=num_warmup,
        block=block,
        batch_size=batch_size,
        sequence_length=sequence_length,
        embed_dim=embed_dim,
        autocast=autocast,
        device=device,
        profile=profile,
        att_name=attention_name,
    )









def plot(args, results: List[Dict[str, Any]]):
    df = pd.DataFrame(results)
    HEADS = args.heads[-1]
    AMP = args.pytorch_amp[-1]
    EMB = args.embedding_dim[-1]
    BATCH_SIZE = args.batch_size[-1]

    df_filtered = df[
       (df["heads"] == HEADS)
        & (df["autocast"] == AMP)
        & (df["embed_dim"] == EMB)
        & (df["batch_size"] == BATCH_SIZE)
    ]

    df_filtered.sort_values(
        by=["sequence_length", "max_memory"], ascending=[False, True], inplace=True
    )
    sns.barplot(
        x="sequence_length",
        y="max_memory",
        hue="attention_name",
        data=df_filtered,
        palette="Set2",
    )
    plt.xlabel("Sequence length")
    plt.ylabel("Max memory being used")
    plt.title("Memory use")
    plt.savefig("memory_vs_attention.png")
    plt.clf()

    df_filtered.sort_values(
        by=["sequence_length", "run_time"], ascending=[False, True], inplace=True
    )
    sns.barplot(
        x="sequence_length",
        y="run_time",
        hue="attention_name",
        data=df_filtered,
        palette="Set2",
    )
    plt.xlabel("Sequence length")
    plt.ylabel("Average epoch time")
    plt.title("Runtime")
    plt.savefig("runtime_vs_attention.png")





if __name__ == "__main__":
    # Get the user requests
    parser = argparse.ArgumentParser(
        "Benchmark different attention mechanisms on various sequence lengths"
    )
    parser.add_argument(
        "-a", "--attentions", nargs="+", default=['mlp', 'dot']
    )
    parser.add_argument("-emb", "--embedding_dim", nargs="+", default=[64], type=int)
    parser.add_argument(
        "-sl", "--seq_length", nargs="+", default=[16, 256], type=int
    )
    parser.add_argument("-bs", "--batch_size", nargs="+", default=[4, 8], type=int)
    parser.add_argument("-heads", "--heads", nargs="+", default=[3, 4], type=int)
    parser.add_argument("-nl", "--n_layer", nargs="+", default=[3], type=int)

    parser.add_argument("-fp16", "--pytorch_amp", nargs="+", default=[True], type=bool)
    parser.add_argument("-plot", "--plot", action="store_true", default=True)
    parser.add_argument(
        "-profile",
        "--profile",
        help="Pofile the runtime and memory",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    # Setup the test configs
    constants = {
        "device": torch.device("cuda") if _use_cuda else torch.device("cpu"),
        "num_warmup": 5,
        "num_steps": 10,
        "dropout": 0.2,
        "profile": args.profile,
    }

    param_grid = {
        "autocast": args.pytorch_amp,
        "heads": args.heads,
        "attention_name": args.attentions,
        "sequence_length": args.seq_length,
        "embed_dim": args.embedding_dim,
        "batch_size": args.batch_size,
        "n_layer":args.n_layer,
    }

    print(
        "Testing the following parameters: \n",
        json.dumps(param_grid, sort_keys=True, indent=4),
    )


    grid = ParameterGrid(param_grid)

    grid_outputs = []

    for params in tqdm(grid, total=len(grid)):
        outputs = test_gpt_block(**constants, **params)  # type: ignore
        results = {**outputs, **params}
        grid_outputs.append(results)

    print(json.dumps(grid_outputs, sort_keys=True, indent=4))

    # Optional plots
    if args.plot:
        plot(args, grid_outputs)

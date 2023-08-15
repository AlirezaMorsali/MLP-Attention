"""
This code draws strong inspiration and borrows heavily from the implementation available at https://github.com/facebookresearch/xformers/blob/main/xformers/benchmarks/benchmark_core.py.
"""

import itertools
import argparse
import torch
from torch.utils import benchmark
import config
from tqdm import tqdm
import json

from model import Head, MLPHead, MultiHeadAttention, Block, GPTLanguageModel

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


def get_batch(split, batch_size, block_size, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y



def bench_head(input_shapes, n_embd, head_size, dropout, min_run_time, device):
  results = []
  for mlp, shape in itertools.product([False, True], input_shapes):
      x = torch.rand(shape, device=device)
      hidden_size = block_size = shape[1]

      head = Head(n_embd, head_size, hidden_size, block_size, dropout) if mlp else MLPHead(n_embd, block_size, head_size, hidden_size, dropout)
      head.to(device)
      
      result = benchmark.Timer(
          stmt="head(x)",
          globals={"head": head, "x": x},
          label="Attention",
          sub_label=f"Head {'mlp' if mlp else 'dot'}",
          description=f"Input Shape: {shape}",
      ).blocked_autorange(min_run_time=min_run_time)
      
      results.append(result)

  compare = benchmark.Compare(results)
  compare.print()

  return compare

def bench_multihead(input_shapes, n_embd, n_head, head_size, dropout, min_run_time, device):

  results = []
  for mlp, shape in itertools.product([False, True], input_shapes):
      x = torch.rand(shape, device=device)
      hidden_size = block_size = shape[1]

      multiheadattention = MultiHeadAttention(n_embd, n_head, block_size, head_size, hidden_size, mlp_attention=mlp, dropout=dropout)
      multiheadattention.to(device)

      result = benchmark.Timer(
          stmt="multiheadattention(x)",
          globals={"multiheadattention": multiheadattention, "x": x},
          label="MultiHeadAttention",
          sub_label=f"Head {'mlp' if mlp else 'dot'}",
          description=f"Input Shape: {shape}",
      ).blocked_autorange(min_run_time=min_run_time)
      
      results.append(result)

  compare = benchmark.Compare(results)
  compare.print()

  return compare



def bench_block(input_shapes, n_embd, n_head, dropout, min_run_time, device):
    
  results = []
  for mlp, shape in itertools.product([False, True], input_shapes):
      x = torch.rand(shape, device=device)
      hidden_size = block_size = shape[1]

      block = Block(n_embd, n_head, block_size, hidden_size, mlp_attention=mlp, dropout=dropout)
      block.to(device)

      result = benchmark.Timer(
          stmt="block(x)",
          globals={"block": block, "x": x},
          label="Block",
          sub_label=f"Head {'mlp' if mlp else 'dot'}",
          description=f"Input Shape: {shape}",
      ).blocked_autorange(min_run_time=min_run_time)
      
      results.append(result)

  compare = benchmark.Compare(results)
  compare.print()

  return compare


def bench_model(input_shapes, n_embd, n_head, n_layer, vocab_size, dropout, min_run_time, device):
  results = []
  for mlp, shape in itertools.product([False, True], input_shapes):

      batch_size = shape[0]
      hidden_size = block_size = shape[1]
      # x = torch.rand(shape, device=device)
      x, _ = get_batch('train', batch_size, block_size, device)

      gpt_m = GPTLanguageModel(n_embd, n_head, n_layer, block_size, hidden_size,  device, vocab_size=vocab_size, mlp_attention=mlp, dropout=dropout)
      gpt_model = gpt_m.to(device)


      result = benchmark.Timer(
          stmt="gpt_model(x)",
          globals={"gpt_model": gpt_model, "x": x},
          label="gpt_model",
          sub_label=f"Head {'mlp' if mlp else 'dot'}",
          description=f"Input Shape: {shape}",
      ).blocked_autorange(min_run_time=min_run_time)
      
      results.append(result)


  # Compare and print benchmark results
  compare = benchmark.Compare(results)
  compare.print()

  return compare



def main():
    parser = argparse.ArgumentParser(description="Run benchmark functions.")
    parser.add_argument("--head", action="store_true", help="Run the head benchmark")
    parser.add_argument("--multihead", action="store_true", help="Run the multihead benchmark")
    parser.add_argument("--block", action="store_true", help="Run the block benchmark")
    parser.add_argument("--model", action="store_true", help="Run the model benchmark")
    parser.add_argument("--min_run_time", type=float, default=1, help="Minimum runtime")
    parser.add_argument("--n_head", default=3, type=int, help="number of heads")
    parser.add_argument("--seq_length", default=256, nargs='+', type=int, help="sequence length")
    parser.add_argument("--n_embd", default=384, type=int, help="embedding size")
    parser.add_argument("--n_layer", default=3, type=int, help="number of layers")
    parser.add_argument("--batch_size", nargs='+', type=int,  default=[16, 32, 64], help="List of batch sizes for benchmarking")
    parser.add_argument("--dropout", default=0.2, type=float, help='drop out value')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    head_size = args.n_embd//args.n_head
    n_head = args.n_head
    seq_length = args.seq_length
    n_embd = args.n_embd
    dropout = args.dropout
    min_run_time = args.min_run_time
    batch_sizes = args.batch_size
    n_layer = args.n_layer
    

    param_grid = {
        "n_embd": n_embd,
        "seq_length": seq_length,
        "n_head": n_head,
        "min_run_time": min_run_time,
        "batch_sizes": batch_sizes,
        "device": 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(
        "Testing the following parameters: \n",
        json.dumps(param_grid, sort_keys=True, indent=4),
    )

    # Create a list of all possible combinations of the input shapes
    input_shapes = list(itertools.product(batch_sizes, seq_length, [n_embd]))


    if args.head:
        bench_head(input_shapes, n_embd, head_size, dropout, min_run_time, device)
    if args.multihead:
        bench_multihead(input_shapes, n_embd, n_head, head_size, dropout, min_run_time, device)
    if args.block:
        bench_block(input_shapes, n_embd, n_head, dropout, min_run_time, device)
    if args.model:
        bench_model(input_shapes, n_embd, n_head, n_layer, vocab_size, dropout, min_run_time, device)



if __name__ == "__main__":
    main()


import torch

batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 50
learning_rate = 0.0003
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 3
n_layer = 3
n_hidden_layers = 1
dropout = 0.2
hidden_size = block_size

import torch
from model import GPTLanguageModel
from utils import decode, device, load_data

if __name__ == "__main__":
    # Load data and vocab
    text, chars, vocab_size, stoi, itos, data = load_data("input.txt")
    block_size = 256
    n_embd = 384
    n_head = 3
    n_layer = 3
    dropout = 0.2

    # Load the trained model
    model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        mlp_attention=False,
    ).to(device)

    # Load the weights from training
    model.load_state_dict(
        torch.load("results/Final/original_model.pt", map_location=device)
    )
    model.eval()

    # Provide a starting sequence (prompt)
    context = torch.tensor([[stoi[" "]]], dtype=torch.long, device=device)

    # Generate text
    output = model.generate(context, max_new_tokens=1000)
    print(decode(output[0].tolist(), itos))

from importlib.metadata import version

import tiktoken
import torch

from utils.workshop.architecture.gpt_model import GPTModel
from utils.workshop.architecture import supplementary
from utils.workshop.architecture.supplementary import create_dataloader_v1
from utils.workshop.pretraining.supplementary import calc_loss_loader, calc_loss_batch, evaluate_model, generate_and_print_sample, plot_losses
from utils.workshop.weightloading.gpt_download import download_and_load_gpt2

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.0,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}

GPT_CONFIG_124M_256 = {
    "vocab_size": 50257,   # Vocabulary size
    "context_length": 256, # Shortened context length (orig: 1024)
    "emb_dim": 768,        # Embedding dimension
    "n_heads": 12,         # Number of attention heads
    "n_layers": 12,        # Number of layers
    "drop_rate": 0.1,      # Dropout rate
    "qkv_bias": False      # Query-key-value bias
}

TRAIN_START_TEXT = "It's his ridiculous modesty"

def check_setup():
    print("torch version:", version("torch"))
    print("tiktoken version:", version("tiktoken"))
    pass


def load_data() -> str:
    with open("./data/raw/the-verdict.txt", "r") as _f:
        data = _f.read()
        print(f"text length: {len(data)}")
        print(f"starting texts: {data[:20]}")
    return data


def example_byte_pair_tokenizer(raw_text: str):
    # dataloader = supplementary.create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    dataloader = supplementary.create_dataloader_v1(raw_text)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)


def generate_llm_model(model_config: dict) -> GPTModel:
    model:GPTModel = GPTModel(model_config)
    model.eval()  # Disable dropout during inference

    # # Sample showing tokenization and dimensions
    # tokenizer = tiktoken.get_encoding("gpt2")
    #
    # batch = []
    #
    # txt1 = "Every effort moves you"
    # txt2 = "Every day holds a"
    #
    # batch.append(torch.tensor(tokenizer.encode(txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))
    # batch = torch.stack(batch, dim=0)
    # print(batch)
    # torch.manual_seed(123)
    #
    # out = model(batch)
    # print("Input batch:\n", batch)
    # print("\nOutput shape:", out.shape)
    # print(out)
    # #####################

    return model


def generate_text_simple(model: GPTModel, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def example_generate_text(model: GPTModel, model_config: dict):
    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    out_ids = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=model_config["context_length"]
    )

    print("Output:", out_ids)
    print("Output length:", len(out_ids[0]))

    decoded_text = tokenizer.decode(out_ids.squeeze(0).tolist())
    print(decoded_text)
    
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


def pretrain_model(text_data: str, model_config: dict, model_path: str):
    model: GPTModel = generate_llm_model(model_config=model_config)
    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=model_config["context_length"],
        stride=model_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=model_config["context_length"],
        stride=model_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # An optional check that the data was loaded correctly:
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    # Another optional check that the token sizes are in the expected ballpark:
    train_tokens = 0
    for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()

    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()

    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)
    print("All tokens:", train_tokens + val_tokens)

    if not torch.cuda.is_available():
        print("CUDA is not available")
        # raise RuntimeWarning("CUDA is not available")
        pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes

    # INITIAL LOSS
    torch.manual_seed(123)  # For reproducibility due to the shuffling in the data loader

    with torch.no_grad():  # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    # Want loss to go towards zero
    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
    ########

    # TRAINING WITH LOSS MONITORED
    torch.manual_seed(123)
    model = GPTModel(model_config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10
    tokenizer = tiktoken.get_encoding("gpt2")
    train_losses, val_losses, tokens_seen, trained_model = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=TRAIN_START_TEXT, tokenizer=tokenizer
    )

    print("save the model")
    torch.save(model.state_dict(), model_path)

    print("plot the losses")
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen, model


def check_pretrain_model_output(model_path: str, model_config):
    model = GPTModel(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    start_context = "It's his ridiculous modesty"
    tokenizer = tiktoken.get_encoding("gpt2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer).to(device),
        max_new_tokens=10,
        context_size=model_config["context_length"]
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


def weight_loading():
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")

    print(f"settings: {settings}")
    print(f"params: {params}")


def run_app():
    check_setup()
    raw_data = load_data()

    ## Sample tokenizer
    ## Normally use a pretrained tokenizer for llm
    ## Also normally use BytePair encoding
    # from utils.simple_tokenizer import ExampleSimpleTokenizer
    # tokens = ExampleSimpleTokenizer(text=raw_data)


    # BytePair tokenizer example
    example_byte_pair_tokenizer(raw_text=raw_data)

    # Raw model example
    model: GPTModel = generate_llm_model(model_config=GPT_CONFIG_124M)
    example_generate_text(model=model, model_config=GPT_CONFIG_124M)


    # Pre-train model
    model_path="./model.pth"
    # pretrain_model(text_data=raw_data, model_config=GPT_CONFIG_124M_256, model_path=model_path)
    # Sample test trained model
    check_pretrain_model_output(model_path=model_path, model_config=GPT_CONFIG_124M_256)


    # Weight loading on the model
    weight_loading()

    return None


if __name__ == '__main__':
    run_app()

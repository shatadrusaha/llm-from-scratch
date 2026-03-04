# Import necessary libraries.
import torch.nn as nn
import torch
import numpy as np
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


"""                                 GPT Model Implementation.                                   """

# Implementation of multi-head attention.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.dim_head = (
            d_out // num_heads
        )  # Reduces the projection dim to match the desired output dim.

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(
            d_out, d_out
        )  # Use a Linear layer to combine head outputs.
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # # Compute query, key, and value vectors.
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Implicitly split the matrix by adding a num_heads dimension.
        # Then unroll the last dim: (b, num_tokens, d_out) --> (b, num_tokens, num_heads, head_dim).
        keys = keys.view(b, num_tokens, self.num_heads, self.dim_head)
        values = values.view(b, num_tokens, self.num_heads, self.dim_head)
        queries = queries.view(b, num_tokens, self.num_heads, self.dim_head)

        # Transpose from shape (b, num_tokens, num_heads, dim_head) to (b, num_heads, num_tokens, dim_head).
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores.
        attn_scores = queries @ keys.transpose(
            2, 3
        )  # Compute dot product for each head.
        mask_bool = self.mask.bool()[
            :num_tokens, :num_tokens
        ]  # Mask truncated to the number of tokens.

        attn_scores.masked_fill_(
            mask_bool, -torch.inf
        )  # Use the mask to fill attention scores.

        # Normalise attention scores.
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Apply dropout to attention weights.
        attn_weights = self.dropout(attn_weights)

        # Compute context vectors.
        context_vecs = (attn_weights @ values).transpose(
            1, 2
        )  # Tensor shape: (b, num_tokens, n_heads, dim_head).
        context_vecs = context_vecs.contiguous().view(
            b, num_tokens, self.d_out
        )  # Combine heads, where self.d_out = self.num_heads * self.dim_head
        context_vecs = self.out_proj(context_vecs)  # Add an optional linear projection.

        return context_vecs


# Define a GELU activation function class.
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        _gelu = (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )
        return _gelu


# Define a feed forward neural network module.
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(
                cfg["emb_dim"], 4 * cfg["emb_dim"]
            ),  # Expand the embedding dimension.
            GELU(),  # Apply GELU activation function.
            nn.Linear(
                4 * cfg["emb_dim"], cfg["emb_dim"]
            ),  # Reduce the embedding dimension back to the original size.
        )

    def forward(self, x):
        return self.layers(x)


# Define layer normalization class.
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


# Transformer block class.
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg=cfg)
        self.norm1 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.norm2 = LayerNorm(emb_dim=cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(p=cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block.
        shortcut = x

        x = self.norm1(x)
        x = self.att(x)  # shape: (batch_size, num_tokens, emb_dim)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back.

        # Shortcut connection for feed forward block.
        shortcut = x

        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back.

        return x


# GPT model class.
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(
            num_embeddings=cfg["vocab_size"], embedding_dim=cfg["emb_dim"]
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=cfg["context_length"], embedding_dim=cfg["emb_dim"]
        )
        self.drop_emb = nn.Dropout(p=cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(emb_dim=cfg["emb_dim"])
        self.out_head = nn.Linear(
            in_features=cfg["emb_dim"], out_features=cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )  # Device setting for model training, CPU or GPU.
        x = tok_embeds + pos_embeds  # shape: (batch_size, seq_len/num_tokens, emb_dim)
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits


"""                                 GPT Model Implementation.                                   """

# Function to assign trainable PyTorch parameters (weights) from a tensor/array.
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


# Function to load GPT-2 weights into the GPT model.
# Set the model’s positional and token embedding weights to those specified in params.
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    # Iterate over each transformer block and set the weights and biases.
    for i in range(len(params["blocks"])):
        # Divide the attention weights into three equal parts (query, key, and value).
        q_w, k_w, v_w = np.split(
            ary=params["blocks"][i]["attn"]["c_attn"]["w"],
            indices_or_sections=3,
            axis=-1,
        )
        gpt.trf_blocks[i].att.W_query.weight = assign(
            left=gpt.trf_blocks[i].att.W_query.weight, right=q_w.T
        )
        gpt.trf_blocks[i].att.W_key.weight = assign(
            left=gpt.trf_blocks[i].att.W_key.weight, right=k_w.T
        )
        gpt.trf_blocks[i].att.W_value.weight = assign(
            left=gpt.trf_blocks[i].att.W_value.weight, right=v_w.T
        )

        # Divide the bias weights into three equal parts (query, key, and value).
        q_b, k_b, v_b = np.split(
            ary=params["blocks"][i]["attn"]["c_attn"]["b"],
            indices_or_sections=3,
            axis=-1,
        )
        gpt.trf_blocks[i].att.W_query.bias = assign(
            left=gpt.trf_blocks[i].att.W_query.bias, right=q_b
        )
        gpt.trf_blocks[i].att.W_key.bias = assign(
            left=gpt.trf_blocks[i].att.W_key.bias, right=k_b
        )
        gpt.trf_blocks[i].att.W_value.bias = assign(
            left=gpt.trf_blocks[i].att.W_value.bias, right=v_b
        )

        # Set the output projection weights and biases.
        gpt.trf_blocks[i].att.out_proj.weight = assign(
            left=gpt.trf_blocks[i].att.out_proj.weight,
            right=params["blocks"][i]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[i].att.out_proj.bias = assign(
            left=gpt.trf_blocks[i].att.out_proj.bias,
            right=params["blocks"][i]["attn"]["c_proj"]["b"],
        )

        # Set the feed-forward network weights and biases.
        gpt.trf_blocks[i].ff.layers[0].weight = assign(
            left=gpt.trf_blocks[i].ff.layers[0].weight,
            right=params["blocks"][i]["mlp"]["c_fc"]["w"].T,
        )  # fully-connected layer (layer 1)
        gpt.trf_blocks[i].ff.layers[0].bias = assign(
            left=gpt.trf_blocks[i].ff.layers[0].bias,
            right=params["blocks"][i]["mlp"]["c_fc"]["b"],
        )  # fully-connected layer (layer 1)
        gpt.trf_blocks[i].ff.layers[2].weight = assign(
            left=gpt.trf_blocks[i].ff.layers[2].weight,
            right=params["blocks"][i]["mlp"]["c_proj"]["w"].T,
        )  # projection layer (layer 2)
        gpt.trf_blocks[i].ff.layers[2].bias = assign(
            left=gpt.trf_blocks[i].ff.layers[2].bias,
            right=params["blocks"][i]["mlp"]["c_proj"]["b"],
        )  # projection layer (layer 2)

        # Set the layer normalization scale and shift parameters.
        gpt.trf_blocks[i].norm1.scale = assign(
            left=gpt.trf_blocks[i].norm1.scale, right=params["blocks"][i]["ln_1"]["g"]
        )  # layer norm 1
        gpt.trf_blocks[i].norm1.shift = assign(
            left=gpt.trf_blocks[i].norm1.shift, right=params["blocks"][i]["ln_1"]["b"]
        )  # layer norm 1
        gpt.trf_blocks[i].norm2.scale = assign(
            left=gpt.trf_blocks[i].norm2.scale, right=params["blocks"][i]["ln_2"]["g"]
        )  # layer norm 2
        gpt.trf_blocks[i].norm2.shift = assign(
            left=gpt.trf_blocks[i].norm2.shift, right=params["blocks"][i]["ln_2"]["b"]
        )  # layer norm 2

    # Set the final layer normalization scale and shift parameters.
    gpt.final_norm.scale = assign(left=gpt.final_norm.scale, right=params["g"])
    gpt.final_norm.shift = assign(left=gpt.final_norm.shift, right=params["b"])

    # Final, linear output, layer weights.
    # The original GPT-2 model by OpenAI used 'weight tying' (reusing the token embedding weights in the output layer to reduce the total number of parameters).
    gpt.out_head.weight = assign(left=gpt.out_head.weight, right=params["wte"])


# Function (modified) for text generation with more diversity, temperature scaling and top-k sampling.
def generate(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):
    # The for loop is the same as before: gets logits and only focuses on the last time step.
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # Filter logits with top_k sampling.
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        # Apply temperature scaling and sample the next token.
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        # Carry out greedy next token selection as before, when temperature scaling is disabled.
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Stop generating early, if end-of-sequence token is encountered.
        if idx_next == eos_id:
            break

        # Append sampled index to the running sequence.
        idx = torch.cat((idx, idx_next), dim=1)  # shape (batch, n_tokens+1)
    return idx


tokenizer = tiktoken.get_encoding("gpt2")


# Helper functions to convert text to token IDs and vice versa.
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text=text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(
        0
    )  # .unsqueeze(0) adds the batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # .squeeze(0) removes the batch dimension
    return tokenizer.decode(tokens=flat.tolist())


# Function to calculate loss for a batch of input and target tokens.
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        input=logits.flatten(0, 1), target=target_batch.flatten()
    )
    return loss


# Function to calculate average loss over a data loader.
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        # Iterate over all batches, if no fixed 'num_batches' is specified.
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader, if 'num_batches' exceeds the number of batches in the data loader.
        # if num_batches > len(data_loader):
        #     num_batches = len(data_loader)
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()  # Sum loss for each batch.
        else:
            break
    return total_loss / num_batches  # Average the loss over all batches.


# Main function for pretraining an LLM.
def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    # Initialize lists to track losses and tokens seen.
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Start the main training loop.
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode.

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from the previous batch iteration.
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            loss.backward()  # Calculate loss gradients.
            optimizer.step()  # Update model weights using loss gradients.
            tokens_seen += (
                input_batch.numel()
            )  # Count the number of tokens processed in this batch.
            global_step += 1

            # Optional evaluation step.
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Epoch {epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Prints a sample text, after each epoch.
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


# Helper function to evaluate the model on training and validation datasets.
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # Set the model to evaluation mode.

    # Disable gradient tracking for loss calculation.
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # Set the model back to training mode.
    return train_loss, val_loss


# Function to generate text using the GPT model.
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is a (batch, n_tokens) array of indices in the current context.
    for _ in range(max_new_tokens):
        # Crop current context, if it exceeds the supported context size.
        # e.g., if LLM supports only 5 tokens and the context size is 10, only the last 5 tokens are used as context.
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)  # shape (batch, n_tokens, vocab_size)

        # Focus only on the last time step.
        logits = logits[:, -1, :]  # shape (batch, vocab_size)

        # Apply softmax to convert logits to probabilities.
        probas = torch.softmax(logits, dim=-1)  # shape (batch, vocab_size)

        # Greedy decoding, get the index of the token with the highest probability.
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # shape (batch, 1)

        # Append sampled index to the running sequence.
        idx = torch.cat((idx, idx_next), dim=1)  # shape (batch, n_tokens+1)
    return idx


# Helper function to generate and print a sample text from the model.
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()

    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format.
    print("\n")
    model.train()


"""                                 Training a LLM.                                   """
# Function to plot training and validation losses over epochs and tokens seen.
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

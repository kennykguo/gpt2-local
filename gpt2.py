from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math



class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # For scaling down the residual connections
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size
                                                                                                       ))
        
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh * hs = C = 768 channels in the Transformer
        qkv = self.c_attn(x) # (n_embd, 3 * n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # Make # of heads into a batch_dimension
        # Treats B and nh as batches
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)
        att = att.masked_fill(self.bias[:,:,:T, :T] == 0, float ('-inf')) # (B, nh, T, T)
        att = F.softmax(att, dim = -1) # (B, nh, T, T)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side -> (B, T, nh, hs) -> (B, T, nh * hs)
        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU -> Gausian error linear units -> like ReLU but no flat tail at exactly zero
        self.gelu = nn.GELU(approximate = "tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # For scaling down the residual connections
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) 
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # Addition is a branch in the gradients
        # Gradients flow through blocks unchanged
        # Pre-normalization version
        # Attention is a communication, reduce, pooling, weighted sum function
        # MLP collection happens individually - attention is the reduce, MLP is the map
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257 # 50,000 BPE merges, 256 byte tokens, 1 end of text tokens
    n_layer: int = 12
    n_head: int  = 12
    n_embd: int = 768


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # All are randomly initialized
        # nn.ModuleDict allows you to index into model layers like a dictionary
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # Index like a list
            h = nn.ModuleList([Block(config) for _ in range (config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # Weight sharing scheme (old value is deleted by torch)
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # Scaling down residual connections so they have a standard deviation of 1 -> 1:20:00
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # Forward the token and posisition embeddings (exactly like a dictionary)
        # Need to be careful to initialize on the correct device
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)

        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)

        x = tok_emb + pos_emb

        # Forward the blocks of the transformer using list comprehension
        for block in self.transformer.h:
            x = block(x)
        
        # Forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)

        loss = None
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # ( B * T, vocab_size) and (B * T)
        return logits, loss

    # Defines a method bound to a class, but not the instance of the class
    # @classmethod
    # def from_pretrained(cls, model_type):
    #     """Loads pretrained GPT-2 model weights from huggingface"""
    #     assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    #     from transformers import GPT2LMHeadModel
    #     print("loading weights from pretrained gpt: %s" % model_type)

    #     # n_layer, n_head and n_embd are determined from model_type
    #     config_args = {
    #         'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
    #         'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
    #         'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
    #         'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    #     }[model_type]
    #     config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
    #     config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
    #     # Create a from-scratch initialized minGPT model
    #     config = GPTConfig(**config_args)
    #     model = GPT(config)
    #     sd = model.state_dict()
    #     sd_keys = sd.keys()
    #     sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

    #     # init a huggingface/transformers model
    #     model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    #     sd_hf = model_hf.state_dict()

    #     # copy while ensuring all of the parameters are aligned and match in names and shapes
    #     sd_keys_hf = sd_hf.keys()
    #     sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    #     sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    #     transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
    #     # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
    #     # this means that we have to transpose these weights when we import them
    #     assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    #     for k in sd_keys_hf:
    #         if any(k.endswith(w) for w in transposed):
    #             # special treatment for the Conv1D weights we need to transpose
    #             assert sd_hf[k].shape[::-1] == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k].t())
    #         else:
    #             # vanilla copy over the other parameters
    #             assert sd_hf[k].shape == sd[k].shape
    #             with torch.no_grad():
    #                 sd[k].copy_(sd_hf[k])

    #     return model
    
### --------------------------------------------------------------------------------------------------------------------------------

class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
    
        with open("input.txt", 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2') 
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)

        self.current_position += B * T
        # Always need B * T + 1 tokens after
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x ,y
### ------------------------------------------------------------------------------------------------------------------------------------

import tiktoken
import code
import time

# Initialize device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# Initalize our data
train_loader = DataLoaderLite(B=2, T = 1024)

# Tells torch what kernel precision to run -> "highest", "high", etc
torch.set_float32_matmul_precision('high') 

# Create model
model = GPT(GPTConfig())
model = model.to(device)
model = torch.compile(model)

# First and second moment, momentum and RMSProp
optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)

for i in range(50):

    t0 = time.time()

    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    # Start with zero gradient, loss.backward() does a +=, so you must set them to zero
    optimizer.zero_grad()

    # with torch.autocast(device_type = device, dtype = torch.bfloat16):
    logits, loss = model(x,y)
    
    # code.interact(local = locals())
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    print(f"step {i}, loss {loss.item()}, dt: {dt:.2f}ms")





import sys
sys.exit(0)

num_return_sequences = 5
max_length = 30

# Prefix tokens
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("I am going to rob your house")
tokens = torch.tensor(tokens, dtype = torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to('cpu')

# Generating tokens (can put into Jupyter notebook aswell)
while x.size(1) < max_length: # (B, T) -> (5, 8)
    with torch.no_grad():
        # (B, T, vocab_size)
        # Logits at last position
        logits = model(x) # (B, vocab_size)
        logits = logits[:, -1, :]
        # Softmax
        probs = F.softmax(logits, dim = -1)
        # top-k sampling (5, 50) -> keep top 50 probabilities, otherwise clamp to zero
        # Never sample random tokens
        topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim = 1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
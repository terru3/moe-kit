## Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## Model
class MLP(nn.Module):
    def __init__(self, n_embd, n_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(n_ff, n_embd),
            
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head # Dimension of each head's key, query, and value
        
        self.drop = nn.Dropout(p=dropout)

        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)

    def split_heads(self, x):
        B, S, D = x.size()
        # split dimension into n_head * head_dim, then transpose the sequence length w/ n_head
        # output: [B, n_head, S, head_dim]
        return x.view(B, S, self.n_head, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        B, _, S, head_dim = x.size() # _ is n_head which we will merge
        # output: [B, S, n_embd]
        return x.transpose(1, 2).contiguous().view(B, S, self.n_embd)

    def scaled_dot_product(self, q, k, v, dropout, mask=None):
        # q,k,v are [B, n_head, S, head_dim]
        # wei = [B, n_head, S, S]
        wei = q @ k.transpose(-2,-1) / np.sqrt(self.head_dim)
        # mask is [B, 1, S, S]
        if mask is not None:
          wei = wei.masked_fill(mask, float('-inf'))
        wei = dropout(F.softmax(wei, dim=-1))
        out = wei @ v
        return out

    def forward(self, x, mask=None):
        # x: (B, S, n_embd)
        # Step 1 and 2: Project full query, key, value, then split via reshaping
        q = self.split_heads(self.query(x))
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))

        # Step 3: Compute scaled dot-product attention with causal mask
        attn = self.scaled_dot_product(q, k, v, self.drop, mask)

        # Step 4 and 5: Concatenate attention scores, return projected output matrix
        out = self.out(self.combine_heads(attn)) # (B, S, n_embd)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_ff, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttention(n_embd, n_head, dropout)
        self.mlp = MLP(n_embd, n_ff, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # residual connection (stream)
        # pre layer norm
        x = x + self.drop(self.sa(self.ln1(x), mask))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

class PositionalEncoding(nn.Module):
  """
  Formula taken from the original Transformer paper:
  PE(pos, 2i (even)) = sin(pos/(10000^{2i/d_model}))
  PE(pos, 2i+1 (odd)) = cos(pos/(10000^{2i/d_model}))

  See reference for more details:
  https://kikaben.com/transformers-positional-encoding/
  """
  def __init__(self, d_model, max_len):
      # just set d_model = n_embd and max_len = seq_len
      super().__init__()

      position = torch.arange(max_len).unsqueeze(1) # [max_len, 1]
      divisor = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)) # [d_model / 2, half for each of sin and cos]
      pe = torch.zeros(max_len, d_model)
      pe[:, 0::2] = torch.sin(position * divisor)
      pe[:, 1::2] = torch.cos(position * divisor)
      self.register_buffer('pe', pe) # result: self.pe = [max_len, d_model], mapping each token index to a vector of length d_model as desired

  def forward(self, x):
      # x = torch.arange(seq_length) has shape [seq_length], so x.size(0) extracts it, then we index self.pe for the first seq_length mappings
      # note we do not add the positional embeddings to x itself yet, we simply return them
      # output = (seq_length, d_model=n_embd)
      return self.pe[:x.size(0)]

class VanillaTransformer(nn.Module):
    def __init__(self, vocab_size, seq_length,
                 n_embd, n_head, n_ff, n_layer,
                 device, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = PositionalEncoding(n_embd, seq_length)

        self.blocks = nn.Sequential(*[Block(n_embd,
                                            n_head,
                                            n_ff,
                                            dropout) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.drop = nn.Dropout(dropout)
        self.seq_length = seq_length
        self.device = device
        self.init_params()

    # weight initialization (Xavier uniform)
    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # Remark: Xavier normal is not supported at this time.

    def get_causal_mask(self,  x):
        """
        Generates causal mask for decoding
        """
        B, S = x.shape # x = (batch_size x seq_len)
        attn_shape = (B, 1, S, S)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8') # k = 1 shifts the diagonal, so that the main diagonal gets 0's
        return (torch.from_numpy(subsequent_mask) == 0).to(self.device)
        # True along main diagonal + below, False elsewhere

    def forward(self, x):
        
        x = x.to(torch.int64)
        B, S = x.shape

        # get mask
        mask = self.get_causal_mask(x).to(self.device)
        # mask = (B x 1 x S x S)

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(S))
        x = self.drop(tok_emb + pos_emb)
        # (B, S, n_embd)
        for block in self.blocks:
            x = block(x, ~mask) # (B, S, n_embd)
        # negate mask to fill originally False values with -inf later
        logits = self.lm_head(x) # (B, S, vocab_size)

        return logits


    def generate(self, input_ids, method='multinomial',
                 max_new_tokens=1000, temp=None,
                 num_beams=None, p_nucleus=None, k=None):

        # input_ids begins as (B, S)
        self.eval()

        for _ in range(max_new_tokens):
            if method in ['multinomial', 'temperature', 'greedy', 'nucleus', 'top-k']:
                # i) Truncate to the most recent `max length` tokens
                text_cond = input_ids[:, -self.seq_length:]
                # ii) Retrieve predictions
                with torch.no_grad():
                    logits = self(text_cond)
                # model output: (B, S, vocab_size)
                # iii) Find last token logits of each
                logits = logits[:, -1, :] # (B, vocab_size)

                # if temperature sampling, divide logits by temp before applying softmax
                if method == 'temperature':
                    logits = logits / temp

                # iv) Take softmax along each
                probs = F.softmax(logits, dim=-1)

                # v) Sample next token depending on method
                if method == 'greedy':
                    next_idx = probs.argmax(dim=-1).unsqueeze(-1)

                elif method in ['multinomial', 'temperature', 'nucleus', 'top-k']:
                    if method == 'nucleus':
                        assert p_nucleus is not None and (0 < p_nucleus) and (p_nucleus <= 1)

                        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                        prob_cumsum = sorted_probs.cumsum(dim=-1)
                        idx_remove = prob_cumsum > p_nucleus
                        # shift one right to ensure the first token is above the threshold
                        idx_remove[..., 1:] = idx_remove[..., :-1].clone()
                        idx_remove[..., 0] = False
                        # retrieve original indices by reverse-sorting
                        remove_mask = idx_remove.gather(dim=-1,
                                          index=sorted_idx.argsort(dim=-1))
                        # ^ specifically, we do this by first argsorting the indices which were returned from argsort
                        # you can show that this returns indices that when used to subset a sorted array, returns the original array in unsorted order
                        # https://stackoverflow.com/questions/52127723/pytorch-better-way-to-get-back-original-tensor-order-after-torch-sort
                        probs[remove_mask] = 0

                    if method == 'top-k':
                        remove_mask = probs < torch.topk(probs, k).values[..., -1, None] # topk returns (B, 1), leaving only the
                        # kth largest probs (i.e. the cutoff value for each). Then mask is same size as probs (B, vocab_size)
                        probs[remove_mask] = 0

                    # Sample probabilistically via scores
                    next_idx = torch.multinomial(probs, num_samples=1) # (B, 1)

                # vi) Autoregressively append to input_text
                input_ids = torch.cat((input_ids, next_idx), dim=-1)

                # now input_text = (B, S + 1)
        
        return input_ids


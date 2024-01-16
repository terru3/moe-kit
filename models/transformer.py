### TODOs:
# more documentation
# More features (see below) such as (smaller) weight initialization

## Imports
import copy
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.backends.cuda import sdp_kernel, SDPBackend

backend_map = {
    SDPBackend.MATH: {
        "enable_math": True,
        "enable_flash": False,
        "enable_mem_efficient": False,
    },
    SDPBackend.FLASH_ATTENTION: {
        "enable_math": False,
        "enable_flash": True,
        "enable_mem_efficient": False,
    },
    SDPBackend.EFFICIENT_ATTENTION: {
        "enable_math": False,
        "enable_flash": False,
        "enable_mem_efficient": True,
    },
}

## util functions
def softmax_off_by_one(x, dim):
    e_x = torch.exp(x - torch.amax(x, dim=dim, keepdim=True)[0])
    return e_x / (1 + e_x.sum(dim=dim, keepdim=True))


class GEGLU(nn.Module):
    """
    https://arxiv.org/abs/2002.05202
    """

    def forward(self, x):
        assert x.shape[-1] % 2 == 0

        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class SwiGLU(nn.Module):
    """
    https://arxiv.org/abs/2002.05202
    """

    def forward(self, x):
        assert x.shape[-1] % 2 == 0

        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


## activation mappings. defined inside to make use of model params
act_fn_dict = {
    "GELU": nn.GELU(),
    "GEGLU": GEGLU(),  # halves dim to n_ff/2
    "SwiGLU": SwiGLU(),
}  # halves dim to n_ff/2
# SwiGLU seems faster than GELU (tho potentially due to fewer params), and potentially better too, from preliminary testing

## Model
class MLP(nn.Module):
    def __init__(self, n_embd, n_ff, activation, dropout=0.1):
        super().__init__()

        act_fn = act_fn_dict[activation]
        # if GEGLU or SwiGLU, halves hidden dim after chunking
        # note this decreases num model params
        hidden_dim_out = n_ff if activation == "GELU" else n_ff // 2

        self.net = nn.Sequential(
            nn.Linear(n_embd, n_ff),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim_out, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_kv_head,
        device,
        use_rotary_embd,
        scale,
        softmax_off_by_one,
        dropout=0.1,
    ):
        super().__init__()

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.softmax_off_by_one = softmax_off_by_one
        self.drop = nn.Dropout(p=dropout)

        self.n_kv_head = n_kv_head
        self.n_repeat = self.n_head // self.n_kv_head

        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.value = nn.Linear(n_embd, n_kv_head * self.head_dim, bias=False)
        self.out = nn.Linear(n_embd, n_embd, bias=False)

        self.device = device

        self.rotary_embd = (
            RotaryPositionalEmbedding(self.head_dim, scale=scale)
            if use_rotary_embd
            else None
        )
        # note in GQA, RoPE still uses regular head_dim

    def split_heads(self, x, n_head):
        # note n_head may differ in the case of GQA (q = n_head, k/v = n_kv_head)
        B, S, D = x.size()
        # split dimension into n_head * head_dim, then transpose the sequence length w/ n_head
        # output: [B, n_head, S, head_dim]
        return x.view(B, S, n_head, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        B, _, S, head_dim = x.size()  # _ is n_head which we will merge
        # output: [B, S, n_embd]
        return x.transpose(1, 2).contiguous().view(B, S, self.n_embd)

    ## Note: Did not re-write scaled dot product to support GQA——GQA is directly compatible with F.scaled_dot_product_attention
    # def scaled_dot_product(self, q, k, v, dropout, mask=None):
    #     # q,k,v are [B, n_head, S, head_dim]
    #     # wei = [B, n_head, S, S]
    #     wei = q @ k.transpose(-2, -1) / np.sqrt(self.head_dim) # use regular head_dim for scaling
    #     # mask is [B, 1, S, S]
    #     if mask is not None:
    #         wei = wei.masked_fill(mask, float("-inf"))

    #     if self.softmax_off_by_one:
    #         wei = dropout(softmax_off_by_one(wei, dim=-1))
    #     else:
    #         wei = dropout(F.softmax(wei, dim=-1))
    #     out = wei @ v
    #     return out

    def forward(self, x, mask=None):
        # x: (B, S, n_embd)
        # Step 1 and 2: Project query, key, value, then split via reshaping
        q = self.split_heads(self.query(x), self.n_head)
        k = self.split_heads(self.key(x), self.n_kv_head)
        v = self.split_heads(self.value(x), self.n_kv_head)

        if self.rotary_embd:
            q, k = self.rotary_embd(q=q, k=k)

        ## GQA
        k, v = repeat_kv(k, v, self.n_repeat)
        assert (
            k.shape[1] == self.n_head and v.shape[1] == self.n_head
        ), "key and value n_head do not match query n_head"
        # now q, k, v are [B, n_head, S, head_dim)

        # Step 3: Compute scaled dot-product attention with causal mask
        # with torch's flash attention, our mask argument is not actually used
        with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
            try:
                attn = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=self.drop.p if self.device.type == "cuda" else 0,
                    is_causal=True,
                )
            # Both fused kernels do not support non-null attn_mask; let it generate its own (Dec 2023)
            # CPU: Both fused kernels do not support non-zero dropout. (Dec 2023)
            except RuntimeError:
                print("FlashAttention is not supported. See warnings for reasons.")

        # attn = self.scaled_dot_product(q, k, v, self.drop, mask)

        # Step 4 and 5: Concatenate attention scores, return projected output matrix
        out = self.out(self.combine_heads(attn))  # (B, S, n_embd)
        return out


# helper function for GQA
def repeat_kv(k, v, n_repeat):
    k = torch.repeat_interleave(k, repeats=n_repeat, dim=1)
    v = torch.repeat_interleave(v, repeats=n_repeat, dim=1)
    return k, v


class SwitchFeedForward(nn.Module):
    """
    Switch FeedForward Layer.
    TODO
    Inputs:
        -
    Returns: Tuple of length 4
        -Layer output
        -Token count per expert (for auxiliary loss)
        -Sum of token probs per expert (for auxiliary loss)
        -Token count dropped (for logging)
    """

    def __init__(
        self,
        d_model,
        n_ff,
        use_amp,
        capacity_factor,
        drop_tokens: bool,
        n_experts: int,
        expert: MLP,
        activation,
        noise=0.1,
        dropout=0.1,
    ):
        super().__init__()

        self.use_amp = use_amp
        self.capacity_factor = capacity_factor
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens
        self.noise = noise

        self.experts = nn.ModuleList(
            [
                copy.deepcopy(expert(d_model, n_ff, activation, dropout))
                for _ in range(n_experts)
            ]
        )

        # Routing layer
        self.switch = nn.Linear(d_model, n_experts)

    def forward(self, x):

        x = x.float()  # cast to float32 for stability
        B, S, n_embd = x.shape

        # apply multiplicative jitter
        if self.noise > 0:
            x *= torch.empty_like(x).uniform_(1.0 - self.noise, 1.0 + self.noise)

        x = rearrange(x, "b s d -> (b s) d")
        probs = F.softmax(self.switch(x), dim=-1)  # (b*s) x n_experts

        # convert to half precision
        if self.use_amp:
            probs = probs.half()
            x = x.half()

        max_prob, route_idx = torch.max(probs, dim=-1)

        # compute expert capacity
        # (num tokens * CF) / n_experts
        capacity = int(x.shape[0] * self.capacity_factor / self.n_experts)

        # obtain token idx for each expert
        # list of len (n_expert) of tensors indicating token idx going to that expert
        token_indices = [
            torch.eq(route_idx, i).nonzero() for i in range(self.n_experts)
        ]

        # num tokens of each expert
        # new_tensor ensures same dtype and device
        expert_token_counts = x.new_tensor(
            [len(token_indices[i]) for i in range(self.n_experts)]
        )

        # check capacity and drop tokens
        dropped = []
        if self.drop_tokens:
            for i in range(self.n_experts):
                if expert_token_counts[i] > capacity:
                    # no shuffle——drop earlier tokens
                    dropped.append(token_indices[i][capacity:])
                    token_indices[i] = token_indices[i][:capacity]

        # feed tokens to relevant experts
        out = torch.zeros_like(x)
        expert_out = [
            self.experts[i](x[token_indices[i], :]) for i in range(self.n_experts)
        ]

        for i in range(self.n_experts):
            out[token_indices[i], :] = expert_out[i]
        if dropped:
            # concat dropped tokens, skip experts
            dropped = torch.cat(dropped)
            out[dropped, :] = x[dropped, :]

        # scale values by gating probabilities
        # unsqueeze max_prob for broadcasting
        out * rearrange(max_prob, "num_tokens -> num_tokens ()")

        # separate batch_size and seq_len
        # do not use SEQ_LEN or BATCH_SIZE. if inference, may have batch_size=1 and/or smaller seq len, for example
        out = rearrange(out, "(b s) d -> b s d", s=S)

        return out, expert_token_counts, probs.sum(0), len(dropped)


class Block(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_kv_head,
        n_ff,
        device,
        norm_first,
        use_rotary_embd,
        use_amp,
        switch,
        capacity_factor,
        drop_tokens,
        n_experts,
        expert,
        softmax_off_by_one,
        activation,
        noise,
        mlp_dropout,
        expert_dropout,
        scale,
    ):
        super().__init__()
        self.sa = MultiHeadAttention(
            n_embd,
            n_head,
            n_kv_head,
            device,
            use_rotary_embd,
            scale,
            softmax_off_by_one,
            mlp_dropout,
        )
        if switch:
            self.ff = SwitchFeedForward(
                n_embd,
                n_ff,
                use_amp,
                capacity_factor,
                drop_tokens,
                n_experts,
                expert=MLP,
                activation=activation,
                noise=noise,
                dropout=mlp_dropout,
            )  # no change to dropout here
        else:
            self.ff = MLP(n_embd, n_ff, activation=activation, dropout=mlp_dropout)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.norm_first = norm_first
        self.mlp_drop = nn.Dropout(p=mlp_dropout)
        self.expert_drop = nn.Dropout(p=expert_dropout)
        self.switch = switch

    def forward(self, x, mask):
        # residual connection (stream)

        # pre layer norm
        if self.norm_first:
            x = x + self.mlp_drop(self.sa(self.ln1(x), mask))
            if self.switch:
                out, expert_token_counts, prob_sum, n_dropped = self.ff(self.ln2(x))
                x = x + self.expert_drop(out)  # expert dropout
                return x, expert_token_counts, prob_sum, n_dropped
            else:
                x = x + self.mlp_drop(self.ff(self.ln2(x)))
        else:
            x = self.ln1(x + self.mlp_drop(self.sa(x, mask)))
            if self.switch:
                out, expert_token_counts, prob_sum, n_dropped = self.ff(x)
                x = self.ln1(x + self.expert_drop(out))  # expert dropout
                return x, expert_token_counts, prob_sum, n_dropped
            else:
                x = self.ln2(x + self.mlp_drop(self.ff(x)))

        return x


class PositionalEncoding(nn.Module):
    """
    Formula taken from the original Transformer paper:
    PE(pos, 2i (even)) = sin(pos/(10000^{2i/d_model}))
    PE(pos, 2i+1 (odd)) = cos(pos/(10000^{2i/d_model}))

    https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model, max_len):
        # just set d_model = n_embd and max_len = seq_len
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        divisor = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
        )  # [d_model / 2, half for each of sin and cos]
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * divisor)
        pe[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer(
            "pe", pe
        )  # result: self.pe = [max_len, d_model], mapping each token index to a vector of length d_model as desired

    def forward(self, x):
        # x = torch.arange(seq_length) has shape [seq_length], so x.size(0) extracts it, then we index self.pe for the first seq_length mappings
        # note we do not add the positional embeddings to x itself yet, we simply return them
        # output = (seq_length, d_model=n_embd)
        return self.pe[: x.size(0)]


class RotaryPositionalEmbedding(nn.Module):
    """
    Applies relative positional embeddings to the queries and keys prior to performing scaled dot-product attention. Embeddings
    are calculated via process in which each position of Q and K receives a unique rotation.

    scale (float): Scales frequency of RoPE. Trick to interpolate embeddings to extend context length
    """

    def __init__(self, dim, base=10000, scale=1):
        # dim != n_embd. dim = key head_dim
        super().__init__()

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2)
        self.register_buffer("inv_freq", inv_freq)
        self.scale = scale
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, q, k):
        B, n_head, S, head_dim = k.shape

        t = torch.arange(S, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        t *= self.scale
        freqs = torch.einsum(
            "i,j->ij", t, self.inv_freq
        )  # outer product: (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1).to(k.device)  # (seq_len, dim)
        self.cos_cached = emb.cos()[
            None, None, :, :
        ]  # both (1, 1, seq_len, dim), prepare for broadcasting across q and k
        self.sin_cached = emb.sin()[None, None, :, :]

        return apply_rotary_pos_emb(q, k, self.cos_cached, self.sin_cached)


# apply rotary pos emb helpers
def rotate_half(x):
    """
    Splits x in half and applies rotation (e.g. [3, 1, 2, 0] -> [-2, 0, 3, 1]).
    """
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k = (B, n_head, S, head_dim) = result shapes
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class Transformer(nn.Module):
    """
    TODO
    Inputs:
        -vocab_size (int)
        -seq_length (int)
        -n_embd (int)
        -n_head (int):
        -n_ff (int):
        -n_layer (int):
        -device:
        -n_kv_head (int):
        -norm_first (bool=True):
        -use_rotary_embd (bool=True):
        -softmax_off_by_one (bool):
        -use_amp (bool=False):
        -amp_dtype:
        -switch (bool=False): Indicates whether to insert Switch MoE layers.
        -switch_first (bool): Indicates whether to use a Switch layer in the first block.
        -every_n_switch (int): Frequency to insert Switch layers.
        -capacity_factor (float):
        -drop_tokens (bool):
        -n_experts (int):
        -expert:
        -activation (str):
        -noise (float):
        -mlp_dropout (float):
        -expert_dropout (float):
    Returns:
        -

    Switch Transformer: https://arxiv.org/abs/2101.03961
    """

    def __init__(
        self,
        vocab_size,
        seq_length,
        n_embd,
        n_head,
        n_ff,
        n_layer,
        device,
        n_kv_head=None,
        norm_first=True,
        use_rotary_embd=True,
        softmax_off_by_one=False,
        use_amp=False,
        amp_dtype=torch.bfloat16,
        switch=False,
        switch_first=None,
        every_n_switch=None,
        capacity_factor=None,
        drop_tokens=None,
        n_experts=None,
        expert=None,
        activation="GELU",
        noise=0.1,
        mlp_dropout=0.1,
        expert_dropout=0.4,
        scale=1,
    ):
        super().__init__()

        if switch:
            assert (
                isinstance(switch_first, bool)
                and isinstance(every_n_switch, int)
                and isinstance(capacity_factor, (int, float))
                and isinstance(drop_tokens, bool)
                and isinstance(n_experts, int)
                and expert is not None
            ), "For a switch transformer, you must provide a boolean `switch_first`, integer `every_n_switch`, numeric `capacity_factor`, boolean `drop_tokens`, \
                    integer `n_experts` and a MLP class `expert` to serve as the experts."

        assert (
            isinstance(mlp_dropout, (int, float))
            and isinstance(expert_dropout, (int, float))
            and 0 <= mlp_dropout <= 1
            and 0 <= expert_dropout <= 1
        ), "`mlp_dropout` and `expert_dropout` must be numeric values between 0 and 1 (inclusive)."

        if not n_kv_head:
            n_kv_head = n_head
        assert (
            n_head % n_kv_head == 0
        ), "n_kv_head must be divisible by, and at most equal to n_head"

        assert (
            isinstance(n_embd, int)
            and isinstance(n_head, int)
            and isinstance(n_ff, int)
            and isinstance(n_layer, int)
            and isinstance(n_kv_head, int)
            and n_embd > 0
            and n_head > 0
            and n_ff > 0
            and n_layer > 0
            and n_kv_head > 0
        ), "n_embd/n_head/n_ff/n_layer/n_kv_head must be positive integers."

        self.token_embedding = nn.Embedding(vocab_size, n_embd)

        #### toggle between rotary or classic sinusoidal embedding
        self.position_embedding = (
            PositionalEncoding(n_embd, seq_length) if not use_rotary_embd else None
        )
        if scale != 1 and not use_rotary_embd:
            warnings.warn(
                "`scale` provided, but `use_rotary_embd=False`. `scale` will have no effect.",
                stacklevel=2,
            )

        ### Alternate blocks with switch = True/False
        switch_args = np.full((n_layer,), False)
        if switch:
            switch_args[0] = switch_first
            if switch_first:
                switch_args[(every_n_switch)::every_n_switch] = True
            else:
                switch_args[(every_n_switch - 1) :: every_n_switch] = True
                if every_n_switch == 1:
                    switch_args[0] = False
                    warnings.warn(
                        "switch_first=False, but every_n_switch=1. This sets the first layer to a regular MLP and all other layers to Switch layers, and may not be the intended behaviour.",
                        stacklevel=2,
                    )

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    n_head,
                    n_kv_head,
                    n_ff,
                    device,
                    norm_first,
                    use_rotary_embd,
                    use_amp,
                    switch_args[i],
                    capacity_factor,
                    drop_tokens,
                    n_experts,
                    expert,
                    softmax_off_by_one,
                    activation,
                    noise,
                    mlp_dropout,
                    expert_dropout,
                    scale,
                )
                for i in range(n_layer)
            ]
        )

        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.drop = nn.Dropout(mlp_dropout)
        self.switch = switch
        self.seq_length = seq_length
        self.device = device
        self.use_amp = use_amp

        # for printing
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.is_gqa = n_kv_head != n_head

        self.init_params()

    # weight initialization (Xavier uniform)
    def init_params(self, default_initialization=False):
        if not default_initialization:
            for name, p in self.named_parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # Remark: Xavier normal is not supported at this time.

    def get_causal_mask(self, x):
        """
        Generates causal mask for decoding
        """
        B, S = x.shape  # x = (batch_size x seq_len)
        attn_shape = (B, 1, S, S)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype(
            "uint8"
        )  # k = 1 shifts the diagonal, so that the main diagonal gets 0's
        return (torch.from_numpy(subsequent_mask) == 0).to(self.device)
        # True along main diagonal + below, False elsewhere

    def forward(self, x):

        x = x.to(torch.int64)
        B, S = x.shape

        # get mask
        mask = self.get_causal_mask(x).to(self.device)
        # mask = (B x 1 x S x S)

        tok_emb = self.token_embedding(x)
        if self.position_embedding:
            pos_emb = self.position_embedding(torch.arange(S))
            x = self.drop(tok_emb + pos_emb)
        else:
            x = self.drop(tok_emb)
        # (B, S, n_embd)

        expert_token_counts, prob_sum, n_dropped = [], [], []
        for block in self.blocks:
            if block.switch:
                x, counts_i, prob_sum_i, n_dropped_i = block(x, ~mask)
                expert_token_counts.append(counts_i)
                prob_sum.append(prob_sum_i)
                n_dropped.append(n_dropped_i)
            else:
                x = block(x, ~mask)  # (B, S, n_embd)
        # negate mask to fill originally False values with -inf later
        logits = self.lm_head(x)  # (B, S, vocab_size)

        if self.switch:
            return (
                logits,
                torch.stack(expert_token_counts),
                torch.stack(prob_sum),
                n_dropped,
            )
        return logits

    def generate(
        self,
        input_ids,
        method="multinomial",
        max_new_tokens=1000,
        temp=None,
        num_beams=None,
        p_nucleus=None,
        k=None,
    ):

        # input_ids begins as (B, S)
        self.eval()

        for _ in range(max_new_tokens):
            if method in ["multinomial", "temperature", "greedy", "nucleus", "top-k"]:
                # i) Truncate to the most recent `max length` tokens
                text_cond = input_ids[:, -self.seq_length :]
                # ii) Retrieve predictions
                with torch.no_grad():
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.use_amp,
                    ):
                        if self.switch:
                            logits, _, _, _ = self(text_cond)
                        else:
                            logits = self(text_cond)
                # model output: (B, S, vocab_size)
                # iii) Find last token logits of each
                logits = logits[:, -1, :]  # (B, vocab_size)

                # if temperature sampling, divide logits by temp before applying softmax
                if method == "temperature":
                    logits = logits / temp

                # iv) Take softmax along each
                probs = F.softmax(logits, dim=-1)

                # v) Sample next token depending on method
                if method == "greedy":
                    next_idx = probs.argmax(dim=-1).unsqueeze(-1)

                elif method in ["multinomial", "temperature", "nucleus", "top-k"]:
                    if method == "nucleus":
                        assert (
                            p_nucleus is not None
                            and (0 < p_nucleus)
                            and (p_nucleus <= 1)
                        )

                        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                        prob_cumsum = sorted_probs.cumsum(dim=-1)
                        idx_remove = prob_cumsum > p_nucleus
                        # shift one right to ensure the first token is above the threshold
                        idx_remove[..., 1:] = idx_remove[..., :-1].clone()
                        idx_remove[..., 0] = False
                        # retrieve original indices by reverse-sorting
                        remove_mask = idx_remove.gather(
                            dim=-1, index=sorted_idx.argsort(dim=-1)
                        )
                        # ^ specifically, we do this by first argsorting the indices which were returned from argsort
                        # you can show that this returns indices that when used to subset a sorted array, returns the original array in unsorted order
                        # https://stackoverflow.com/questions/52127723/pytorch-better-way-to-get-back-original-tensor-order-after-torch-sort
                        probs[remove_mask] = 0

                    if method == "top-k":
                        remove_mask = (
                            probs < torch.topk(probs, k).values[..., -1, None]
                        )  # topk returns (B, 1), leaving only the
                        # kth largest probs (i.e. the cutoff value for each). Then mask is same size as probs (B, vocab_size)
                        probs[remove_mask] = 0

                    # Sample probabilistically via scores
                    next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)

                # vi) Autoregressively append to input_text
                input_ids = torch.cat((input_ids, next_idx), dim=-1)

                # now input_text = (B, S + 1)

        return input_ids

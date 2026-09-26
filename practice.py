import torch
torch.manual_seed(0)
Tokens_per_block= 16
D= 512
num_blocks=8


torch.manual_seed(0)
seq_len= 40

keys_a= torch.randn(seq_len, D)
values_a= torch.randn(seq_len, D)

keys_b= torch.randn(seq_len, D)
values_b= torch.randn(seq_len, D)

query_a= torch.randn(D)
query_b= torch.randn(D)

cache = init_cache(8)
def init_cache(num_blocks)
cache = {}
cache["keys"]= torch.zeros(num_blocks, Tokens_per_block, D)
cache["value"]= torch.zeros(num_blocks, Tokens_per_block, D)

cache["block_tables"] = {}
cache["lengths"] = {}
cache["free_blocks"] = list(range(num_blocks))
for i in range(seq_len):
    append_tokens(
        cache,
        "seq_a",
        keys_a[i:i+1],
        values_a[i:i+1]
    )

    append_tokens(
        cache,
        "seq_b",
        keys_b[i:i+1],
        values_b[i:i+1]
    )
    append_tokens(
        cache,
        "seq_a",
        keys_a[0:1],
        values_a[0:1]
    )

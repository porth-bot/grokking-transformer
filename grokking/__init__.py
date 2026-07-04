"""A from-scratch transformer and training harness for studying grokking.

The core (attention, LayerNorm, blocks) is written out rather than imported
so every tensor operation is inspectable; ``torch`` supplies autograd,
``nn.Linear``/``nn.Embedding`` parameter containers, and the AdamW optimizer.
"""

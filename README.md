# NeuroScript

A compiled DSL for neural architecture design. Declarative syntax, compile-time shape
inference, readable PyTorch output.

```neuroscript
neuron GPT2Small(vocab_size=50257, d_model=768, num_heads=12, d_ff=3072, num_layers=12):
    in: [*, seq]
    out: [*, seq, vocab_size]
    context:
        embed  = Embedding(vocab_size, d_model)
        blocks = unroll(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff)
        ln_f   = LayerNorm(d_model)
        head   = Linear(d_model, vocab_size)
    graph:
        in -> embed -> blocks -> ln_f -> head -> out
```

`unroll(num_layers)` expands at compile time — 12 independent `TransformerBlock` instances,
each with their own weights, chained sequentially. One declaration. Zero boilerplate.

**[Try it in the playground →](https://severeon.github.io/playground)**
&nbsp;·&nbsp;
**[Read the docs →](https://severeon.github.io/docs/intro)**

---

## The idea

PyTorch is expressive but architectures drown in boilerplate. Dimension mismatches surface
at runtime, deep in a training run. Sharing a model means shipping a Python file with
implicit dependencies and no shape contracts.

NeuroScript compiles architecture definitions to idiomatic `nn.Module` code — shape-checked
at compile time, readable output you can audit and modify.

**Neurons all the way down.** Everything is a neuron. Neurons compose into neurons.
The abstraction is uniform from a single `Linear` to a full transformer stack.

---

## Language features

### Residual connections

```neuroscript
neuron PreNormBlock(dim, heads, d_ff):
    in: [*, seq, dim]
    out: [*, seq, dim]
    graph:
        in -> (main, skip)
        main ->
            LayerNorm(dim)
            MultiHeadSelfAttention(dim, heads)
            FFN(dim, d_ff)
            processed
        (processed, skip) -> Add() -> out
```

### Shape-conditional routing

```neuroscript
neuron AdaptivePool(dim):
    in: [*, seq, dim]
    out: [*, dim]
    context:
        @lazy attn_pool = AttentionPooling(dim)
        mean_pool = GlobalAvgPool()
    graph:
        in -> match:
            [*, 1, dim]:  attn_pool -> out   # single token — full attention
            [*, _, dim]:  mean_pool -> out   # long sequence — mean pool
```

The compiler eliminates unreachable arms at compile time. `@lazy` means the heavy
path is never instantiated unless it's actually reachable.

### Shared stacks with `@static`

```neuroscript
neuron UniversalTransformer(dim, heads, d_ff, depth=6):
    in: [*, seq, dim]
    out: [*, seq, dim]
    context:
        # @static = all iterations share the same weights (Universal Transformer style)
        layers = unroll(depth):
            @static layer = TransformerBlock(dim, heads, d_ff)
    graph:
        in -> layers -> out
```

### Higher-order neurons

Pass neuron *types* as parameters. Build generic architectures that work with any block:

```neuroscript
neuron Stack(block: Neuron, d_model, num_heads, d_ff, count=6):
    in: [*, seq, d_model]
    out: [*, seq, d_model]
    context:
        blocks = unroll(count):
            layer = block(d_model, num_heads, d_ff)
    graph:
        in -> blocks -> out

# Use with any compatible block type
Stack(TransformerBlock,   512, 8, 2048, count=6)
Stack(ConformerBlock,     512, 8, 2048, count=4)
Stack(TransformerDecoderBlock, 512, 8, 2048, count=6)
```

### Contract dispatch

When a higher-order neuron needs different wiring for different block shapes:

```neuroscript
neuron SmartStack(block: Neuron, d_model, count=6):
    in: [*, seq, d_model]
    out: [*, seq, d_model]
    context:
        blocks = unroll(count):
            layer = block(d_model)
    graph:
        in ->
            match(block):
                in [*, seq, d_model] -> out [*, seq, d_model]:
                    blocks -> out          # sequence-preserving block
                in [*, d_model] -> out [*, d_model]:
                    blocks -> out          # token-wise block
```

Resolved entirely at compile time. No runtime dispatch, no overhead.

### Variadic ports

```neuroscript
neuron InceptionBlock(dim):
    in: [*, dim]
    out: [*, dim * 4]
    graph:
        in -> (a, b, c, d)
        a -> Conv2d(dim, dim, 1)                         -> pa
        b -> Conv2d(dim, dim, 1) -> Conv2d(dim, dim, 3)  -> pb
        c -> Conv2d(dim, dim, 1) -> Conv2d(dim, dim, 5)  -> pc
        d -> MaxPool(3, stride=1, padding=1)              -> pd
        (pa, pb, pc, pd) -> Concat(1) -> out
```

`Concat` takes a tuple of any length — same neuron, any number of branches.

### Compile-time shape errors

```neuroscript
neuron Broken:
    in: [batch, 128]
    graph:
        in -> Linear(128, 64) -> Linear(32, 10) -> out
        #                              ^
        # error: shape mismatch — expected [*, 32], got [*, 64]
        # caught at compile time, not during training
```

---

## Primitive library

| Category | Neurons |
|---|---|
| **Core** | `Linear` `Bias` `Scale` `MatMul` `Einsum` |
| **Activations** | `GELU` `ReLU` `SiLU` `Tanh` `Sigmoid` `Softmax` `Mish` `PReLU` `ELU` |
| **Normalization** | `LayerNorm` `RMSNorm` `GroupNorm` `BatchNorm` `InstanceNorm` |
| **Regularization** | `Dropout` `DropPath` `DropConnect` |
| **Convolutions** | `Conv1d` `Conv2d` `Conv3d` `DepthwiseConv` `SeparableConv` `TransposedConv` |
| **Pooling** | `MaxPool` `AvgPool` `GlobalAvgPool` `AdaptiveAvgPool` `AdaptiveMaxPool` `GlobalMaxPool` |
| **Embeddings** | `Embedding` `PositionalEncoding` `LearnedPositionalEmbedding` `RotaryEmbedding` |
| **Attention** | `MultiHeadSelfAttention` `ScaledDotProductAttention` |
| **Structural** | `Fork` `Fork3` `Add` `Multiply` `Concat` `Reshape` `Transpose` `Flatten` `Split` `Slice` `Pad` `Identity` |

---

## Axon package manager

Neurons are distributed as signed packages. Cargo-inspired, git-native.

```bash
# Initialize a package
neuroscript init attention-blocks --author "You <you@example.com>"

# Add a dependency
neuroscript add transformer-blocks \
  --git https://github.com/org/transformer-blocks.git --tag v0.3.0

# Fetch and lock
neuroscript fetch

# Sign and publish
neuroscript keygen my-package
neuroscript publish
```

Every package gets SHA-256 checksums per source file and an Ed25519 signature over the
package checksum. Checksums are verified automatically on fetch.

Use imported neurons like any other:

```neuroscript
use transformer-blocks,src/*

neuron MyModel(dim, vocab):
    in: [*, seq]
    out: [*, seq, vocab]
    graph:
        in ->
            Embedding(vocab, dim)
            PreNormBlock(dim, 8, dim * 4)
            PreNormBlock(dim, 8, dim * 4)
            LayerNorm(dim)
            Linear(dim, vocab)
            out
```

---

## Installation

```bash
cargo install neuroscript
pip install neuroscript-runtime
```

```bash
# Compile to PyTorch
neuroscript compile my_model.ns -o model.py

# Self-contained output (no neuroscript_runtime dependency)
neuroscript compile my_model.ns --bundle -o model.py

# Validate shapes without generating code
neuroscript validate my_model.ns
```

---

## Repositories

| | |
|---|---|
| [neuroscript-rs](https://github.com/neuroscript-lang/neuroscript-rs) | Rust compiler · shape inference · codegen |
| [neuroscript-std](https://github.com/neuroscript-lang/neuroscript-std) | Standard library as Axon packages |
| [neuroscript-runtime](https://github.com/neuroscript-lang/neuroscript-runtime) | Python/PyTorch runtime |
| [neuroscript-docs](https://github.com/neuroscript-lang/neuroscript-docs) | Docs site and WASM playground |

---

## Support development

NeuroScript is a solo project. Sponsoring funds development time directly.

**[Sponsor on GitHub →](https://github.com/sponsors/severeon)**

> Goal: $500/month toward a stable v1.0 release.

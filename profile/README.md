# NeuroScript

> ### ⚠️ **Alpha software.** NeuroScript is a solo project
> *however*, I do have 18+ years of experience. 
>
> #### It moves fast, breaks occasionally, and gets hyperfocused on interesting problems. Expect rough edges. Contributions and bug reports very welcome.

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

**[Try it in the playground →](https://neuroscript-lang.com/playground)**
&nbsp;·&nbsp;
**[Read the docs →](https://neuroscript-lang.com/docs/intro)**

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

## Research context

NeuroScript addresses several well-documented shortcomings in contemporary neural network
research and engineering. The papers below expose these problems; the corresponding
NeuroScript features are noted alongside each.

### Reproducibility and transparency

> *"The reproducibility of scientific findings are an important hallmark of quality and
> integrity in research … unfortunately, many publications fall short of this mark."*

- **Tanksley, Hier & Wunsch II** (2022). [Reproducing Neural Network Research Findings via Reverse Engineering](https://eujournal.org/index.php/esj/article/view/15135). *European Scientific Journal, ESJ* 18(4), 61. &mdash; Documents how insufficient detail in neural network publications makes independent reproduction difficult or impossible, even for landmark results like AlphaGo Zero.

- **Semmelrock et al.** (2025). [Reproducibility in Machine-Learning-Based Research: Overview, Barriers and Drivers](https://onlinelibrary.wiley.com/doi/10.1002/aaai.70002). *AI Magazine*. &mdash; Identifies five pillars of ML reproducibility (code versioning, data access, data versioning, experiment logging, pipeline creation) and finds that many published results are not reproducible in principle due to lack of transparency, missing code, and sensitivity of training conditions.

- **Desai et al.** (2025). [What is Reproducibility in AI and ML Research?](https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.70004) *AI Magazine*. &mdash; Introduces a reproducibility framework and argues that computational requirements and system complexity make even well-documented work infeasible to reproduce for most teams.

**How NeuroScript helps:** A `.ns` file is a complete, deterministic architecture specification. The compiler produces identical PyTorch output from the same source every time. Shape contracts, layer connectivity, and weight-sharing semantics are explicit in the source &mdash; there is nothing implicit to reverse-engineer. The Axon package manager adds SHA-256 checksums and Ed25519 signatures so that published architectures can be fetched and verified cryptographically.

### Dimension mismatch and deep learning bugs

> *"The most common bug fix patterns are fixing data dimension and neural network connectivity."*

- **Islam, Pan, Nguyen & Rajan** (2020). [Repairing Deep Neural Networks: Fix Patterns and Challenges](https://dl.acm.org/doi/10.1145/3377811.3380378). *ICSE 2020*, 1135&ndash;1146. &mdash; Studied 970 bug repairs across five DL libraries and found that dimension mismatches and connectivity errors are the most frequent bug category, that fixes risk introducing adversarial vulnerabilities, and that bug localization is a major developer challenge.

- **Zhang et al.** (2020). [An Empirical Study on Program Failures of Deep Learning Jobs](https://hongyujohn.github.io/icse20-main-199.pdf). *ICSE 2020*. &mdash; Analyzed 4,960 real failures from Microsoft's deep learning platform. 13.5% were DL-specific failures caused by inappropriate model parameters/structures and API misunderstanding, surfacing only at runtime.

**How NeuroScript helps:** Every neuron declares explicit `in:` and `out:` shape contracts. The compiler performs full shape inference across the entire graph and rejects dimension mismatches *at compile time*, before any training run begins. There is no runtime surprise &mdash; a program that compiles will not crash due to shape errors.

### Technical debt in ML systems

> *"Machine learning: the high-interest credit card of technical debt."*

- **Sculley et al.** (2015). [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems). *NeurIPS 2015*. &mdash; Google engineers document how ML systems accumulate massive maintenance costs through boundary erosion, entanglement, glue code, and configuration debt. Only a small fraction of real-world ML code is the model itself; the surrounding infrastructure dominates.

**How NeuroScript helps:** NeuroScript separates architecture definition from training infrastructure, optimizer configuration, and data pipelines. A neuron definition is pure structure with no glue code, no implicit global state, and no entangled dependencies. The "neurons all the way down" abstraction enforces clean compositional boundaries at every level. Compiled output is self-contained, auditable PyTorch &mdash; no hidden framework coupling.

### AI trustworthiness and transparency

> *"Trust in AI is shaped by transparency, accuracy, and interpretability."*

- **Liu, Sandjaja & Wunsch** (2024). [AI Trustworthy: Ethical Challenges and Strategies](https://ieeexplore.ieee.org/document/10956105/). *IEEE SOLI 2024*, 1&ndash;6. &mdash; Proposes a framework linking AI transparency to user trust and argues that interpretability and auditability are prerequisites for trustworthy AI deployment across sectors like healthcare and autonomous vehicles.

**How NeuroScript helps:** NeuroScript compiles to readable, idiomatic `nn.Module` code that developers can audit line by line. There is no opaque intermediate representation and no black-box framework magic. The source `.ns` file reads like documentation of the architecture, and the compiled output is standard Python that any PyTorch user can inspect, modify, and reason about.

### Model interoperability and sharing

- **Daoudi et al.** (2025). [Neural Network Interoperability Across Platforms](https://arxiv.org/abs/2511.02610). *arXiv:2511.02610*. &mdash; Documents how migrating neural network implementations across libraries is challenging due to the lack of standardized specification, and proposes metamodel-based approaches to bridge frameworks.

- **Brito da Silva, Elnabarawy & Wunsch II** (2019). [A Survey of Adaptive Resonance Theory Neural Network Models for Engineering Applications](https://arxiv.org/abs/1905.11437). *Neural Networks* 120, 167&ndash;203. &mdash; Surveys 30 years of ART models, noting that useful engineering properties like speed, configurability, and explainability are difficult to replicate across implementations. Highlights that order-dependence and parameter sensitivity complicate reproducible deployment.

**How NeuroScript helps:** A `.ns` file is a framework-independent architecture specification. Today it compiles to PyTorch; the same source can target future backends without changing the architecture definition. Higher-order neurons and contract dispatch allow generic, reusable components that compose without framework-specific glue. The Axon package manager enables sharing architectures as versioned, signed packages rather than ad-hoc Python files.

### ML supply chain security

- **Australian Cyber Security Centre** (2025). [AI and ML Supply Chain Risks and Mitigations](https://www.cyber.gov.au/business-government/secure-design/artificial-intelligence/artificial-intelligence-and-machine-learning-supply-chain-risks-and-mitigations). &mdash; Reports that serialized model files containing malicious content continue to be uploaded to popular sharing platforms undetected, and that ~25% of organizations surveyed had been victims of AI data poisoning.

- **Jiang et al.** (2025). [Supply-Chain Attacks in Machine Learning Frameworks](https://openreview.net/pdf?id=EH5PZW6aCr). *OpenReview*. &mdash; Demonstrates real-world supply chain attacks on ML frameworks, including malicious dependencies that exploit unsafe serialization formats like Python Pickle.

**How NeuroScript helps:** Axon packages are plain-text `.ns` source files &mdash; not serialized binary objects, not Pickle, not executable code. Every package is checksummed (SHA-256 per file) and signed (Ed25519 over the package checksum). The compiler verifies signatures on fetch. Because the source is human-readable and the output is standard Python, there is no vector for hidden executable payloads in the architecture definition itself.

---

## Support development

NeuroScript is a solo project. Sponsoring funds development time directly.

**[Sponsor on GitHub →](https://github.com/sponsors/severeon)**

> Goal: $500/month toward a stable v1.0 release.

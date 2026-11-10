# Pardes107: isolated single encoder family

One384D FP32 graph for queries (128total tokens) and passages (512). Explicit text
recipe3 uses `query: ` and an empty passage prefix; normalization2 is pardes-clean-1.
They are closed enum variants; existing recipes1/2 and normalization1 retain their
bytes and identities. A new family/package/chunking identity prevents old vector reuse.

The exact graph and unchanged tokenizer are distributed as model-pardes-qa491-torahpairs107-fp32-v1.
The config is opt-in. It does not change application defaults or any old family.
The associated vector build keeps the existing full coverage/resolution/retrieval gates.

Attribution: Pardes (https://github.com/efrul/Pardes) student of Dicta's
neodictabert-bilingual-embed, CC BY4.0; original Pardes code AGPL3. See model release
NOTICE/license for original acknowledgements. This repository's new Rust cleaner is
an independently written implementation of the published pardes-clean-1 behavior,
verified against1210 Python outputs; it retains HTML/entity/quote/whitespace semantics
without adding Unicode normalization. New encoder adapters use graph tensors, not
unreviewed remote model code. Model attribution does not imply upstream endorsement.

Validation:669 unit tests passed (one existing ignored, one expensive memory stress
case excluded);27 chunker tests;1210 cleaner differential cases; actual Rust/Python
cleaned inputs and token IDs equal on12 query/passage cases including both token caps.
FP32 ONNX outputs match in those cases. GPU shards still need their own1000-text
parity certificate. No8GB/Intel8 latency claim and no borrowing ensemble TEST scores.

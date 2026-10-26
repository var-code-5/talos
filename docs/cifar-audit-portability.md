# Windows/Linux audit failure investigation

Run `a58b24409529bf46a1fb0ef6` reached its first round with matching dataset,
initial weights, training source, PyTorch 2.13.0+cpu, and NumPy 2.5.2. The Windows
peer used Python 3.14.6; the Linux peer used Python 3.13.7. Both returned a local
5,000-example SGD update, but the strict replay check rejected the updates before
any global aggregation.

Recomputing the saved contributions on Linux gave:

| Submitted contribution | Max weight error | Max momentum error | Relative weight L2 error |
| --- | ---: | ---: | ---: |
| Linux peer, replayed on Linux | 0 | 0 | 0 |
| Windows peer, replayed on Linux | 0.000218009 | 0.005269990 | 0.000345914 |

Windows reported training loss 2.278034887; its Linux replay produced
2.278043651. Both reported the same 12.84% training accuracy for that shard.
Disabling MKLDNN for a native-FP32 Linux replay reduced the maximum weight error
to 0.000031779 but still failed the unchanged tolerance. This evidence is
consistent with optimized floating-point paths diverging through a long sequence
of local SGD updates; matching random seeds is not a cross-platform arithmetic
contract. [PyTorch documents the limits of cross-platform reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

The repair introduces an explicit `fp64-portable` profile: double-precision input
normalization, forward/backward operations and momentum arithmetic, with MKLDNN
disabled. Shared round/checkpoint tensors remain float32 and are promoted on
restore. The existing elementwise audit bounds remain `1e-6 + 1e-5*abs(reference)`.
A separately tested deliberately corrupted update is still rejected.

Local real-data tests cover 5,000-example replays at different CPU thread counts,
a second round with restored momentum, equal-peer training, and host/Firecracker
guest agreement. All 31 local test methods passed. The actual updated Windows
peer still needs the short cross-platform validation run; this document does not
claim that result before it is observed.

The historical 62.48% FP32 performance reference remains valid for its original
profile. It must not be compared against the new FP64 profile. A separate FP64
control is being measured. Numerical-profile mismatches are rejected by the
comparison tool.

Future failures preserve submitted local states under `commits/` and per-round
error measurements under `audits/`. The run summary includes the last audit and
human-readable maximum errors, rather than only an opaque failure message.

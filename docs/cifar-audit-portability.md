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
guest agreement. All 31 local test methods passed. The two peer regression tests also passed
after the DHCP address-refresh repair.

The actual Windows/Linux validation `885b687962267fcf2f412b59` subsequently
completed with `fp64-portable`: two epochs, 10,000 training images, 1,000 test
images, seed 42, and full audits. Windows (Python 3.13.17) and Linux (Python
3.13.7) both used PyTorch 2.13.0+cpu and NumPy 2.5.2. Both peers passed both
rounds with zero weight and momentum differences and identical final models.
Wall time was 25.3615 seconds and held-out accuracy was 29.4%. Mean measured
HTTP RTTs were 34.47 ms from Windows and 82.55 ms from Linux (two samples per
peer; these are sparse observations, not a network latency estimate).

The source measurements are saved locally in
`results/cifar/windows-portable-validation/885b687962267fcf2f412b59/experiment.json`.
This confirms the tested Windows/Linux configuration; it is not a guarantee for
all hardware or software combinations.

The historical 62.48% FP32 performance reference remains valid for its original
profile. It must not be compared against the new FP64 profile. A separate [full FP64 control](../benchmarks/cifar/portable-reference/README.md)
completed with 62.48% accuracy in 959.115 seconds. Numerical-profile mismatches are rejected by the
comparison tool.

Future failures preserve submitted local states under `commits/` and per-round
error measurements under `audits/`. The run summary includes the last audit and
human-readable maximum errors, rather than only an opaque failure message.

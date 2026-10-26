# Observed local control run

Numerical profile: **legacy `fp32-native`**. This is not an FP64 control run.

These are measured artifacts from experiment `2983b5c91394a6e0f7388154`, not
synthetic or physical two-host results. Two equal CPU peers ran on one Linux
laptop with seed 42, full independent auditing, all CIFAR-10 training/test images,
and 10 epochs. Held-out accuracy was 62.48%; both aggregate checkpoints matched.

`initial.json` contains only the public randomly initialized CNN weights. Use it
for the separate-host trial so both placements begin with exactly the same tensor
bytes. `experiment.json` contains configurations and measured per-peer/per-round
metrics. It contains no experiment key, dataset images, or trained model tensors.

See [the validation record](../../../docs/cifar-local-results.md) for timing,
network-volume definitions, hardware/load limitations, and pending native/remote
validation. Comparison command after the friend completes a matching run:

```sh
python -m mtcp.cifar compare benchmarks/cifar/local-reference/experiment.json \
  results/cifar/FRIEND_RUN/experiment.json --output results/cifar/comparison
```

# Observed FP64 local control

Run `3572dc8f3026f03b2e1d87b7` completed 10 epochs on all 50,000 training and
10,000 test images, seed 42, two equal CPU peers on one Linux host, one thread
and a separate pinned CPU per peer, and full independent audits.

The `fp64-portable` profile reached 62.48% accuracy in 959.115 seconds. Both
peers produced the same final model; all 50 rounds passed audits with zero
weight and momentum differences. The recorded mean authenticated HTTP RTTs
were 1.015 and 0.879 ms. Times include training, auditing, evaluation, and waits.
This is one observed run, not a repeated performance estimate.

`experiment.json` preserves the configuration and measured per-round metrics.
`initial.json` is the shared public random initialization. No cluster key is included.
A full separate-host run with matching settings is still required for a placement
comparison. The two-round Windows validation uses a smaller dataset and cannot
provide that performance comparison.

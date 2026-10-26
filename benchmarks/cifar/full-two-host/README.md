# Full two-host CIFAR-10 experiment

Run `c30c66da00ad46ed90695d21` completed 10 epochs with two equal Linux/Windows peers,
50,000 training and 10,000 test images, seed 42, one thread per peer, full audits,
and the `fp64-portable` profile. This is a process-backend experiment.

See [the full measured report](../../../docs/cifar-full-experiment.md) and
[the standalone comparison](../full-placement-comparison/comparison.html).
`experiment.json` retains per-peer and per-round observations;
`trajectory-check.json` records agreement with the single-host control;
`collection-notes.json` records collector connection interruptions;
`provenance.json` identifies source and scope. No cluster key is included.

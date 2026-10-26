# Full CIFAR-10 placement comparison

The same-workload comparison pairs full local portable-FP64 run
`3572dc8f3026f03b2e1d87b7` with full Linux–Windows run
`c30c66da00ad46ed90695d21`. Both use seed 42, 10 epochs, all 50,000 training and
10,000 held-out images, two equal peers, and full independent audits.

Read [the measured report](../../../docs/cifar-full-experiment.md) for definitions,
results, collection interruptions, and limits. `comparison.html` is standalone;
CSV and JSON preserve the comparison metrics. SVG, PDF, and PNG contain the
measured learning and latency curves. One paired observation supplies no
confidence interval and does not isolate network cost from hardware/OS/load.

Regenerate the comparison from the repository root:

```sh
.venv-cifar/bin/python -m mtcp.cifar compare \
  benchmarks/cifar/portable-reference/experiment.json \
  benchmarks/cifar/full-two-host/experiment.json \
  --output benchmarks/cifar/full-placement-comparison
```

Regenerate figures with the plotting version used for this record:

```sh
uv run --no-project --with matplotlib==3.11.2 python \
  benchmarks/cifar/full-placement-comparison/plot.py
```

The plotting dependency is separate from the application/training environment.

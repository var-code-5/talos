import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[3]
files=[root/'benchmarks/cifar/portable-reference/experiment.json',root/'benchmarks/cifar/full-two-host/experiment.json']
runs=[json.loads(p.read_text()) for p in files]
colors=['#235789','#b24c23'];labels=['Two peers, one Linux host','Linux + Windows, two hosts']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(12,8.6),layout='constrained')
for d,color,label in zip(runs,colors,labels):
 rounds=d['peers'][0]['rounds'];x=[r['round']+1 for r in rounds]
 axes[0,0].plot(x,[100*r['test_accuracy'] for r in rounds],color=color,label=label,linewidth=1.8,linestyle='--' if d is runs[1] else '-')
 axes[0,1].plot(x,[r['test_loss'] for r in rounds],color=color,linewidth=1.8,linestyle='--' if d is runs[1] else '-')
 axes[1,0].plot([r['elapsed_seconds'] for r in rounds],[100*r['test_accuracy'] for r in rounds],color=color,linewidth=1.8,label=label)
 for i,p in enumerate(d['peers']):
  axes[1,1].plot([r['round']+1 for r in p['rounds']],[r['rtt_mean_ms'] for r in p['rounds']],color=color,alpha=.8,linewidth=1.2,linestyle='-' if i==0 else '--',label=f"{label}: {p['peer']['name']}")
for ax,title,xlabel,ylabel in [
 (axes[0,0],'A. Accuracy by synchronization round','Synchronization round','Held-out top-1 accuracy (%)'),
 (axes[0,1],'B. Cross-entropy by synchronization round','Synchronization round','Held-out cross-entropy'),
 (axes[1,0],'C. Accuracy by elapsed time','Elapsed time (seconds)','Held-out top-1 accuracy (%)'),
 (axes[1,1],'D. Per-peer authenticated HTTP latency','Synchronization round','HTTP RTT (ms)')]:
 ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.2);ax.set_axisbelow(True)
axes[0,0].legend(fontsize=8,loc='lower right')
axes[1,0].legend(fontsize=8,loc='lower right')
axes[1,1].legend(fontsize=7,loc='upper left')
fig.suptitle('CIFAR-10: full audited training on one versus two physical hosts\n50,000 train / 10,000 test; 10 epochs; seed 42; portable FP64; process backend',fontsize=13)
out=root/'benchmarks/cifar/full-placement-comparison'
for extension in ['svg','pdf','png']:
 fig.savefig(out/f'full-experiment.{extension}',dpi=200,metadata={'Creator':'MTCP CIFAR evidence plotting script'})
plt.close(fig)
print('Saved standalone SVG, PDF, and PNG figures.')

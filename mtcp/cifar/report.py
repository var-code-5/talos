"""Standalone comparison artifact. Reject mismatched experiment workloads."""
import csv
import html
import json
from pathlib import Path
import statistics
from .metrics import distribution

CONFIG_FIELDS=('epochs','train_limit','test_limit','batch_size','sync_samples','threads','lr','momentum','audit','numerics','dataset_id','artifact_digest')


def compare(paths,destination,allocation_comparison=False):
    runs=[json.loads(Path(p).read_text()) for p in paths]
    for run in runs:
        run['config'].setdefault('numerics','fp32-native')
        run['config'].setdefault('allocation','equal')
        run['config'].setdefault('indexing_version',1)
    if not runs:raise ValueError('Supply at least one completed experiment')
    reference=runs[0]['config'];peers=len(reference['members'])
    def hardware(config):
        return [(p['id'],p['host_id'],p['threads'],p['backend'],
                 {k:p['resources'].get(k) for k in ('cpu_description','architecture','os','cpu_affinity','offered_vcpus','offered_memory_mb')}) for p in config['members']]
    initial_digest=runs[0]['peers'][0]['initial_weight_digest']
    torch_base=reference['members'][0]['torch_version'].split('+')[0]
    numpy_version=reference['members'][0]['numpy_version']
    for run in runs:
        if run['status']!='completed' or not run['peer_models_identical']:raise ValueError('Only completed, internally consistent runs can be compared')
        if any(run['config'][key]!=reference[key] for key in CONFIG_FIELDS) or len(run['config']['members'])!=peers:
            raise ValueError('Workload configurations differ; comparison would be confounded')
        if run['config']['indexing_version']!=reference['indexing_version']:
            raise ValueError('Indexing versions differ')
        if allocation_comparison:
            if reference['indexing_version']!=2 or hardware(run['config'])!=hardware(reference):
                raise ValueError('Allocation comparisons require version-2 indexing and matching peer hardware/resources')
            if run['config'].get('planner')!=reference.get('planner'):raise ValueError('Planner settings differ')
        elif run['config']['allocation']!=reference['allocation']:
            raise ValueError('Allocation policies differ; use --allocation-comparison')
        if reference['indexing_version']==2 and run['config'].get('planner')!=reference.get('planner'):
            raise ValueError('Planner settings differ')
        if [p['backend'] for p in run['config']['members']]!=[p['backend'] for p in reference['members']]:
            raise ValueError('Isolation backends differ')
        if any(p['torch_version'].split('+')[0]!=torch_base or p['numpy_version']!=numpy_version for p in run['config']['members']):
            raise ValueError('PyTorch or NumPy versions differ between experiments')
        if any(p['initial_weight_digest']!=initial_digest for p in run['peers']):
            # Different seeds legitimately select different initial models; pair within seed.
            same_seed=next((other for other in runs if other['config']['seed']==run['config']['seed']),run)
            expected=same_seed['peers'][0]['initial_weight_digest']
            if any(p['initial_weight_digest']!=expected for p in run['peers']):raise ValueError('Initial checkpoints differ within the run')
            for other in runs:
                if other['config']['seed']==run['config']['seed'] and any(p['initial_weight_digest']!=expected for p in other['peers']):
                    raise ValueError('Initial checkpoints differ for the same seed; use --initial-checkpoint')
    rows=[]
    for run in runs:
        nodes=run['peers'];total_bytes=sum(p['totals']['network_request_body_bytes']+p['totals']['network_response_body_bytes'] for p in nodes)
        rtt_values=[value for p in nodes for value in p.get('rtt_samples_ms',[])]
        if not rtt_values and len(nodes)==2:
            # With exactly two peers each round's mean is its one observed link RTT.
            rtt_values=[r['rtt_mean_ms'] for p in nodes for r in p['rounds']]
        pooled_rtt=distribution(rtt_values)
        # Directed HTTP exchanges are counted at the initiating peer, so this sum does not double-count them.
        rows.append(dict(run_id=run['config']['run_id'],topology=run['topology'],seed=run['config']['seed'],numerics=run['config']['numerics'],
            allocation=run['config']['allocation'],indexing_version=run['config']['indexing_version'],
            end_to_end_seconds=run.get('end_to_end_seconds',run['wall_seconds']),
            update_wait_peer_seconds=sum(p['totals']['barrier_seconds'] for p in nodes),
            all_sync_peer_seconds=sum(p['totals']['barrier_seconds']+p['totals'].get('completion_wait_seconds',0)+p['totals'].get('plan_agreement_seconds',0) for p in nodes),
            completion_wait_peer_seconds=sum(p['totals'].get('completion_wait_seconds',0) for p in nodes),
            plan_agreement_peer_seconds=sum(p['totals'].get('plan_agreement_seconds',0) for p in nodes),
            calibration_peer_seconds=sum(p['totals'].get('calibration_seconds',0) for p in nodes),
            accuracy_percent=100*run['test_accuracy'],test_loss=run['test_loss'],macro_f1=nodes[0]['final_evaluation']['macro_f1'],
            wall_seconds=run['wall_seconds'],global_samples_per_second=reference['train_limit']*reference['epochs']/run['wall_seconds'],
            rtt_mean_ms=statistics.mean(p['rtt_ms']['mean'] for p in nodes if p['rtt_ms']['mean'] is not None) if len(nodes)>1 else 0,
            rtt_median_ms=pooled_rtt['median'],rtt_p95_ms=pooled_rtt['p95'],
            worst_peer_rtt_p95_ms=max((p['rtt_ms']['p95'] or 0) for p in nodes),network_body_mib=total_bytes/1024**2,
            audit_peer_seconds=sum(p['totals']['audit_seconds'] for p in nodes),
            evaluation_peer_seconds=sum(p['totals']['evaluation_seconds'] for p in nodes),
            compute_peer_seconds=sum(p['totals']['train_seconds'] for p in nodes),
            cpu_peer_seconds=sum(p['totals']['cpu_seconds'] for p in nodes),
            peak_peer_rss_mb=max(r['local_peak_rss_mb'] for p in nodes for r in p['rounds'])))
    root=Path(destination);root.mkdir(parents=True,exist_ok=True)
    with open(root/'comparison.csv','w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    paired=[]
    for seed in sorted({row['seed'] for row in rows}):
        if allocation_comparison:
            equal=[r for r in rows if r['seed']==seed and r['allocation']=='equal']
            aware=[r for r in rows if r['seed']==seed and r['allocation']=='compute-aware']
            if equal and aware:
                paired.append(dict(seed=seed,equal_runs=len(equal),compute_aware_runs=len(aware),
                    end_to_end_speed_ratio=statistics.mean(r['end_to_end_seconds'] for r in equal)/statistics.mean(r['end_to_end_seconds'] for r in aware),
                    accuracy_percentage_point_difference=statistics.mean(r['accuracy_percent'] for r in aware)-statistics.mean(r['accuracy_percent'] for r in equal)))
            continue
        local=[row for row in rows if row['seed']==seed and row['topology']=='single-host-peers']
        remote=[row for row in rows if row['seed']==seed and row['topology']=='separate-host-peers']
        if local and remote:
            paired.append(dict(seed=seed,local_runs=len(local),separate_runs=len(remote),
                observed_wall_speed_ratio=statistics.mean(r['wall_seconds'] for r in local)/statistics.mean(r['wall_seconds'] for r in remote),
                accuracy_percentage_point_difference=statistics.mean(r['accuracy_percent'] for r in remote)-statistics.mean(r['accuracy_percent'] for r in local)))
    (root/'comparison.json').write_text(json.dumps({'runs':rows,'paired_seed_comparisons':paired,'allocation_comparison':allocation_comparison,'configuration':{k:reference[k] for k in CONFIG_FIELDS}},indent=2)+'\n')
    def plot(key,title,ylabel):
        width,height=780,270;left,top,right,bottom=65,25,15,40
        points=[[(r['elapsed_seconds']+run['config'].get('startup_seconds',0),r[key]) for r in run['peers'][0]['rounds']] for run in runs]
        xmax=max(x for series in points for x,y in series) or 1;ymax=max(y for series in points for x,y in series) or 1
        ymax=1 if key=='test_accuracy' else ymax*1.1
        colors=['#2064a5','#1a886b','#ae6651','#8255a0','#d98b16']
        body=''
        for i in range(5):
            y=top+(height-top-bottom)*i/4
            body+=f'<line x1="{left}" y1="{y}" x2="{width-right}" y2="{y}" stroke="#dce4ed"/><text x="{left-8}" y="{y+4}" text-anchor="end" font-size="11" fill="#667">{ymax*(1-i/4):.2f}</text>'
        for i,series in enumerate(points):
            path=' '.join(f'{left+x/xmax*(width-left-right):.2f},{top+(1-y/ymax)*(height-top-bottom):.2f}' for x,y in series)
            body+=f'<polyline points="{path}" stroke="{colors[i%len(colors)]}" stroke-width="2" fill="none"/>'
        body+=f'<text x="{width/2}" y="{height-5}" text-anchor="middle" font-size="12">Elapsed time including startup and calibration (seconds); max {xmax:.1f}</text>'
        legend=' '.join(f'<span style="color:{colors[i%len(colors)]}">{html.escape(run["topology"])} / {html.escape(run['config']['allocation'])} / seed {run["config"]["seed"]}</span>' for i,run in enumerate(runs))
        return f'<section><h2>{title}</h2><p>{ylabel}</p><svg viewBox="0 0 {width} {height}" role="img" aria-label="{title}">{body}</svg><p>{legend}</p></section>'
    headings=list(rows[0]);table='<table><thead><tr>'+''.join(f'<th>{html.escape(k)}</th>' for k in headings)+'</tr></thead><tbody>'
    for row in rows:table+='<tr>'+''.join(f'<td>{html.escape(str(round(row[k],4) if isinstance(row[k],float) else row[k]))}</td>' for k in headings)+'</tr>'
    table+='</tbody></table>'
    class_tables=''
    from .data import CLASSES
    for run in runs:
        quality=run['peers'][0]['final_evaluation']
        class_rows=''.join(f'<tr><td>{name}</td><td>{quality["precision"][i]:.4f}</td><td>{quality["recall"][i]:.4f}</td><td>{quality["f1"][i]:.4f}</td></tr>' for i,name in enumerate(CLASSES))
        environments='\n'.join(f'{p["peer"]["name"]}: {p["environment"]["platform"]}; {p["peer"]["threads"]} CPU thread(s); PyTorch {p["environment"]["torch"]}' for p in run['peers'])
        class_tables+=f'<section><h2>{html.escape(run["topology"])} / {html.escape(run['config']['allocation'])} / seed {run["config"]["seed"]}</h2><pre>{html.escape(environments)}</pre><p>Held-out top-5 accuracy: {quality["test_top5_accuracy"]:.2%}</p><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{class_rows}</tbody></table><details><summary>Confusion matrix (rows true, columns predicted)</summary><pre>{html.escape(json.dumps(quality["confusion_matrix"]))}</pre></details></section>'
    topologies={r['topology'] for r in rows}
    note=('Physical two-host results are pending. These observations currently contain a single topology.'
          if topologies=={'single-host-peers'} else 'Topology is inferred from advertised physical host identities; hardware and OS also affect these observations.')
    page=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>MTCP CIFAR comparison</title><style>body{{font:15px system-ui;background:#f3f6fa;color:#24384e;max-width:1200px;margin:35px auto;padding:0 20px}}section{{background:white;border:1px solid #dce4ed;border-radius:8px;padding:24px;margin:22px 0}}h1{{font-size:32px}}h2{{font-size:20px}}svg{{width:100%;max-height:320px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}section{{overflow-x:auto}}table{{border-collapse:collapse;font-size:12px}}td,th{{border-bottom:1px solid #ddd;padding:12px;text-align:left;white-space:nowrap}}.scroll{{overflow:auto}}p{{line-height:1.6}}span{{margin-right:20px}}</style></head><body><h1>CIFAR-10 peer training comparison</h1><p>{note}</p><p>Same model, data membership, optimizer, synchronization schedule, audit policy, and CPU thread allocation. Seeds are shown explicitly. Network volume counts JSON bodies, not TCP/TLS headers. RTT is authenticated HTTP round-trip latency, not ICMP.</p>{plot('test_accuracy','Held-out accuracy','Fraction correct on the same fixed held-out examples')}{plot('test_loss','Held-out cross-entropy','Lower is better')}<section class="scroll">{table}</section><section><h2>Experiment configuration</h2><pre>{html.escape(json.dumps({k:reference[k] for k in CONFIG_FIELDS},indent=2))}</pre></section></body></html>'''
    extra=plot('rtt_mean_ms','Authenticated HTTP RTT','Per-round observed HTTP latency in milliseconds, including server scheduling')+class_tables
    if paired:
        extra+='<section><h2>Paired seed observations</h2><p>Hardware and OS differences also affect observed ratios; these do not isolate network cost.</p><pre>'+html.escape(json.dumps(paired,indent=2))+'</pre></section>'
    if allocation_comparison:
        detail=[]
        for run in runs:
            detail.append(dict(run_id=run['config']['run_id'],allocation=run['config']['allocation'],
                peers=[dict(name=p['peer']['name'],assignments=[r.get('assigned_samples') for r in p['rounds']],
                    audit_limited_rounds=sum(bool(r.get('audit_limited')) for r in p['rounds'])) for p in run['peers']]))
        extra+='<section><h2>Allocation evidence</h2><p>Unequal local SGD steps change the optimization trajectory. Compare held-out quality and total runtime including calibration. Waiting intervals include polling and HTTP; they are not measured CPU idle time. Completion and plan waits are reported separately from update waits.</p><pre>'+html.escape(json.dumps(detail,indent=2))+'</pre></section>'
    page=page.replace('</body>',extra+'</body>')
    (root/'comparison.html').write_text(page)
    return str(root/'comparison.html')

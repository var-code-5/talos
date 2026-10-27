#!/usr/bin/env python3
"""Run paired physical-peer allocation trials after BOTH peers are updated."""
import argparse
from argparse import Namespace
import json
from pathlib import Path
import statistics

from mtcp.cifar.experiment import start
from mtcp.cifar.metrics import write_json
from mtcp.cifar.report import compare


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint',default='http://127.0.0.1:9101')
    parser.add_argument('--peers',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--train-limit',type=int,default=50000)
    parser.add_argument('--test-limit',type=int,default=10000)
    parser.add_argument('--seeds',default='42,43,44')
    parser.add_argument('--audit',choices=['full','none'],default='full')
    parser.add_argument('--output',default='results/cifar-allocation-paired')
    args=parser.parse_args();root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    seeds=[int(seed) for seed in args.seeds.split(',')]
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Use distinct paired seeds')
    artifacts=[]
    for index,seed in enumerate(seeds):
        initial=root/f'initial-{seed}.json'
        order=('equal','compute-aware') if index%2==0 else ('compute-aware','equal')
        for policy in order:
            trial=Namespace(endpoint=args.endpoint,peers=args.peers,discovery_wait=60,epochs=args.epochs,
                train_limit=args.train_limit,test_limit=args.test_limit,batch_size=32,sync_samples=10000,threads=1,
                seed=seed,lr=.001,momentum=.9,audit=args.audit,numerics='fp64-portable',
                allocation=policy,indexing_version=2,initial_checkpoint=str(initial) if initial.exists() else None,
                no_wait=False,output=str(root/'runs'))
            artifact=start(trial);artifacts.append(artifact)
            result=json.loads(Path(artifact).read_text())
            if not initial.exists():write_json(initial,dict(weights=result['config']['initial_weights'],seed=seed))
            write_json(root/'artifacts.json',artifacts)
    report=compare(artifacts,root/'comparison',allocation_comparison=True)
    summary=json.loads((root/'comparison/comparison.json').read_text())
    pairs=summary['paired_seed_comparisons']
    speed=statistics.median(p['end_to_end_speed_ratio'] for p in pairs)
    accuracy=statistics.mean(p['accuracy_percentage_point_difference'] for p in pairs)
    physical=all(json.loads(Path(p).read_text())['topology']=='separate-host-peers' for p in artifacts)
    full_workload=args.epochs==10 and args.train_limit==50000 and args.test_limit==10000 and seeds==[42,43,44] and args.audit=='full'
    assessment=dict(median_end_to_end_speed_ratio=speed,mean_accuracy_percentage_point_difference=accuracy,
        physical_peer_trials=physical,full_release_trial=full_workload,
        release_target_met=physical and full_workload and speed>=1/.9 and accuracy>=-1,
        report=report,status='experimental')
    write_json(root/'assessment.json',assessment)
    print(json.dumps(assessment,indent=2))


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Controlled 2:1 delay experiment using real allocation/peer execution.

This uses a synthetic deterministic update kernel, NOT a CIFAR accuracy benchmark.
Run from the repository: python -m tools.cifar.validate_allocation
"""
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from mtcp.kernel import digest
from mtcp.cifar.allocation import SETTINGS
from mtcp.cifar.execution import execute
from mtcp.cifar.peer import Peer
from mtcp.cifar.training import ARTIFACT,PARAMETERS,decode,encode


def trial(policy,directory,audit='none',disagree=False,train_limit=19200,epochs=1):
    directory=Path(directory);peers=[]
    members=[dict(id=str(i),name=f'synthetic-{i}',endpoint=f'http://synthetic-{i}',host_id=str(i)) for i in range(2)]
    costs=[dict(train_us=us,setup_us=0,audit_us=us,audit_setup_us=0,evaluation_us=1000) for us in (100,200)]
    config=dict(run_id=f'synthetic{policy.replace("-", "")}{audit}',members=members,epochs=epochs,train_limit=train_limit,
        test_limit=16,sync_samples=9600,batch_size=32,threads=1,seed=42,lr=.001,momentum=.9,audit=audit,numerics='fp64-portable',
        initial_weights=encode(np.zeros(PARAMETERS)),dataset_id='synthetic',artifact_digest=ARTIFACT,
        allocation=policy,indexing_version=2,planner=SETTINGS,startup_seconds=.01,
        calibration=[dict(cost=c,seconds=.01) for c in costs])
    def train(payload,speed):
        before=time.perf_counter();time.sleep(payload['count']*speed/1e6)
        values=decode(payload['weights'])+payload['count']*.000001
        momentum=decode(payload['optimizer_momentum']) if payload['optimizer_momentum'] else np.zeros(PARAMETERS)
        momentum=momentum+payload['count']*.0000001
        elapsed=time.perf_counter()-before
        return dict(weights=encode(values),optimizer_momentum=encode(momentum),count=payload['count'],
            train_seconds=elapsed,wall_seconds=elapsed,cpu_seconds=0.,peak_rss_mb=0.,numerics=payload['numerics'],train_loss=1.,train_accuracy=0.)
    for i in range(2):
        peer=object.__new__(Peer);peer.id=str(i);peer.name=f'synthetic-{i}';peer.key='synthetic';peer.directory=directory/str(i)
        peer.lock=threading.RLock();peer.stop=threading.Event();peer.threads=1;peer.data='synthetic';peer.active=config['run_id']
        peer.local_train=lambda payload,speed=costs[i]['train_us']:train(payload,speed)
        peer.runs={config['run_id']:dict(id=config['run_id'],config=config,config_digest=digest(config),status='ready',
            rounds=[],plans={},completions={},commits={},created_at=time.time(),peer=members[i],error=None)}
        peers.append(peer)
    def call(endpoint,route,key,body=None,timeout=10):
        before=time.perf_counter();peer=next(p for p,m in zip(peers,members) if m['endpoint']==endpoint)
        run=peer.runs[config['run_id']];parts=route.strip('/').split('/')
        with peer.lock:
            if route=='/v1/ping':response={}
            elif len(parts)==3:response=dict(status=run['status'])
            elif parts[3]=='rounds':
                commit=run['commits'].get(parts[4]);response=dict(ready=commit is not None,commit=commit,status=run['status'])
            else:
                record=copy.deepcopy(run[parts[4]].get(parts[5]))
                response=dict(ready=record is not None,record=record,status=run['status'])
            response=copy.deepcopy(response)
        return response,dict(seconds=time.perf_counter()-before,request_body_bytes=0,response_body_bytes=len(json.dumps(response)))
    def evaluate(*args):
        time.sleep(.001)
        return dict(test_accuracy=0.,test_loss=1.,macro_f1=0.,evaluation_seconds=.001)
    from mtcp.cifar.allocation import make_plan
    def planned(*args,**kwargs):
        value=make_plan(*args,**kwargs)
        if disagree and threading.current_thread().name=='synthetic-peer-1':
            value['counts'][0]-=32;value['counts'][1]+=32
        return value
    with patch('mtcp.cifar.execution.call',side_effect=call),patch('mtcp.cifar.execution.evaluate',side_effect=evaluate),patch('mtcp.cifar.execution.make_plan',side_effect=planned):
        threads=[threading.Thread(target=execute,args=(p,config['run_id']),name=f'synthetic-peer-{p.id}') for p in peers]
        for thread in threads:thread.start()
        for thread in threads:thread.join(60)
        if any(t.is_alive() for t in threads):
            for p in peers:p.stop.set()
            for thread in threads:thread.join(1)
            raise RuntimeError('Synthetic experiment did not finish')
    return [p.public(config['run_id']) for p in peers]


def main():
    root=Path('results/cifar-allocation-validation');root.mkdir(parents=True,exist_ok=True)
    results={}
    with tempfile.TemporaryDirectory() as temporary:
        for policy in ('equal','compute-aware'):
            nodes=trial(policy,Path(temporary)/policy)
            if any(p['status']!='completed' for p in nodes):raise RuntimeError('; '.join(f"{p['peer']['name']}: {p.get('error')}" for p in nodes))
            if len({p['final_model_digest'] for p in nodes})!=1:raise RuntimeError('Peer models differ')
            results[policy]=dict(wall_seconds=max(p['wall_seconds'] for p in nodes),
                all_wait_peer_seconds=sum(r['barrier_seconds']+r['completion_wait_seconds']+r['plan_agreement_seconds'] for p in nodes for r in p['rounds']),
                update_wait_peer_seconds=sum(p['totals']['barrier_seconds'] for p in nodes),
                assignments=[r['assigned_samples'] for r in nodes[0]['rounds']])
    equal=results['equal'];aware=results['compute-aware']
    results.update(kind='synthetic delay kernel; not CIFAR accuracy or physical-host evidence',
        speed_ratio=equal['wall_seconds']/aware['wall_seconds'],
        all_wait_reduction=1-aware['all_wait_peer_seconds']/equal['all_wait_peer_seconds'])
    (root/'synthetic.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
    if results['all_wait_reduction']<.5 or aware['wall_seconds']>=equal['wall_seconds']:
        raise SystemExit('Synthetic performance target not met')


if __name__=='__main__':main()

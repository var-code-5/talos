"""Start a symmetric experiment from any peer; collect results without training authority."""
import json
import os
from pathlib import Path
import secrets
import time
import urllib.request

from .network import call
from .metrics import write_json


def public(endpoint,route):
    with urllib.request.urlopen(endpoint.rstrip('/')+route,timeout=5) as response:return json.load(response)


def start(args):
    startup=time.perf_counter()
    key=os.environ.get('MTCP_CLUSTER_KEY','')
    if len(key)<32:raise ValueError('Set MTCP_CLUSTER_KEY to the peers\' shared experiment key')
    deadline=time.monotonic()+args.discovery_wait
    while True:
        try:
            members=public(args.endpoint,'/v1/peers')['peers']
        except OSError:
            members=[]
        if len(members)>=args.peers:break
        if time.monotonic()>deadline:raise TimeoutError(f'Only {len(members)} peers discovered; expected {args.peers}. Start the other app or use --connect.')
        time.sleep(.5)
    if len(members)!=args.peers:raise ValueError('Discovered membership differs from requested peer count; use a separate cluster key for this experiment')
    members=sorted(members,key=lambda p:p['id'])
    proposing_peer=public(args.endpoint,'/v1/resources')
    # Prove endpoint/key identity and current capabilities before agreeing on membership.
    verified=[]
    for member in members:
        value,_=call(member['endpoint'],'/v1/hello',key,{'peer':proposing_peer})
        if value['peer']['id']!=member['id']:raise ValueError('Peer endpoint identity changed')
        verified.append(value['peer'])
    members=sorted(verified,key=lambda p:p['id'])
    allocation=getattr(args,'allocation','equal')
    indexing=getattr(args,'indexing_version',None) or (2 if allocation=='compute-aware' else 1)
    if allocation=='compute-aware' and indexing!=2:raise ValueError('Compute-aware allocation requires indexing version 2')
    if indexing==2 and any(1 not in p.get('allocation_protocols',[]) for p in members):
        raise ValueError('Update and restart every peer to support compute-aware allocation')
    first=members[0]
    numerics=getattr(args,'numerics','fp64-portable')
    for member in members:
        if numerics not in member.get('numerics_supported',['fp32-native']):
            raise ValueError(f"Peer {member['name']} needs a code update and restart to support {numerics}")
    if any(p['threads']!=args.threads for p in members):
        raise ValueError('Every peer must advertise the requested PyTorch thread allocation')
    if any(p['dataset_id']!=first['dataset_id'] or p['artifact_digest']!=first['artifact_digest'] for p in members):
        raise ValueError('Peers must have the same prepared dataset and training source')
    initial_path=getattr(args,'initial_checkpoint',None)
    if initial_path:
        initial=json.loads(Path(initial_path).read_text())['weights']
    else:
        # Agree on exact initial tensor bytes, avoiding CPU/OS RNG-kernel differences.
        value,_=call(args.endpoint,'/v1/initialize',key,{'seed':args.seed})
        initial=value['weights']
    config=dict(run_id=secrets.token_hex(12),members=members,epochs=args.epochs,train_limit=args.train_limit,
        test_limit=args.test_limit,batch_size=args.batch_size,sync_samples=args.sync_samples,
        threads=args.threads,seed=args.seed,lr=args.lr,momentum=args.momentum,audit=args.audit,
        dataset_id=first['dataset_id'],artifact_digest=first['artifact_digest'],initial_weights=initial,numerics=numerics)
    if indexing==2:
        from .allocation import SETTINGS
        from mtcp.kernel import digest
        config.update(allocation=allocation,indexing_version=2,planner=SETTINGS)
        request_digest=digest(config)
        # Independent calibration can run concurrently without serializing the
        # startup cost of one laptop behind another.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(members)) as pool:
            futures=[pool.submit(call,m['endpoint'],'/v1/calibrate',key,{'config':config},timeout=700) for m in members]
            calibration=[future.result()[0] for future in futures]
        config.update(calibration=calibration,calibration_request_digest=request_digest,startup_seconds=time.perf_counter()-startup)
    for member in members:call(member['endpoint'],'/v1/runs',key,config)
    print(f"Started {config['run_id']} on {len(members)} peers ({allocation} allocation); no peer is a permanent coordinator",flush=True)
    if args.no_wait:return config['run_id']
    return collect(config,key,args.output)


def collect(config,key,output):
    run_id=config['run_id'];deadline=time.monotonic()+86400;previous=None
    while time.monotonic()<deadline:
        states=[]
        for member in config['members']:
            value,_=call(member['endpoint'],f'/v1/runs/{run_id}',key,timeout=10);states.append(value)
        progress=tuple((s['status'],len(s['rounds'])) for s in states)
        if progress!=previous:
            print(' | '.join(f"{s['peer']['name']}: {s['status']}, {len(s['rounds'])} rounds" for s in states),flush=True);previous=progress
        if any(s['status']=='failed' for s in states):
            write_json(Path(output)/run_id/'experiment.json',{'config':config,'peers':states,'status':'failed'})
            errors='; '.join(f"{s['peer']['name']}: {s.get('error')}" for s in states if s['status']=='failed')
            raise RuntimeError(f"Run {run_id} failed. {errors} Full peer diagnostics saved in {Path(output)/run_id/'experiment.json'}")
        if all(s['status']=='completed' and s.get('accepting_runs',True) for s in states):break
        time.sleep(1)
    else:raise TimeoutError('Experiment collection timed out; peers continue independently')
    digests={s['final_model_digest'] for s in states}
    result=dict(config=config,peers=states,status='completed',peer_models_identical=len(digests)==1,
                topology=states[0]['topology'],wall_seconds=max(s['wall_seconds'] for s in states),
                test_accuracy=states[0]['final_evaluation']['test_accuracy'],
                test_loss=states[0]['final_evaluation']['test_loss'],
                end_to_end_seconds=max(s.get('end_to_end_seconds',s['wall_seconds']) for s in states))
    location=Path(output)/run_id/'experiment.json';write_json(location,result)
    print(f"Saved {location}; accuracy={result['test_accuracy']:.2%}; peer models identical={result['peer_models_identical']}",flush=True)
    if len(digests)!=1:raise RuntimeError('Peers disagree on the final aggregated model')
    return str(location)

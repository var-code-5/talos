"""Bounded, isolated calibration of the same CIFAR kernel used by a peer."""
import json
import math
import statistics
import subprocess
import sys
import time

from mtcp.kernel import digest
from .allocation import validate_cost


def measured_us(seconds, divisor=1, positive=False):
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds < 0 or divisor <= 0:
        raise ValueError('Invalid timing measurement')
    value = round(seconds*1_000_000/divisor)
    if positive and value <= 0:
        raise ValueError('Execution measurement must be positive')
    return value


def isolated(operation, payload, directory, cpus):
    request=dict(operation=operation,payload=payload,directory=directory,cpus=cpus)
    try:
        completed=subprocess.run([sys.executable,'-m','mtcp.cifar.calibration'],
            input=json.dumps(request),text=True,capture_output=True,timeout=120,check=True)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('CIFAR calibration exceeded its 120 second deadline') from exc
    except subprocess.CalledProcessError as exc:
        raise ValueError(f'CIFAR calibration failed: {exc.stderr[-1000:]}') from exc
    return json.loads(completed.stdout)


def calibrate(peer, request):
    from .peer import validate_config
    from .training import ARTIFACT, decode
    # Validate before invoking expensive work, with the same configuration bounds.
    config=request['config'];validate_config(config,require_calibration=False)
    if config.get('indexing_version')!=2 or config['threads']!=peer.threads:
        raise ValueError('Invalid calibration configuration')
    if config['dataset_id']!=peer.manifest['dataset_id'] or config['artifact_digest']!=ARTIFACT:
        raise ValueError('Calibration dataset/artifact mismatch')
    if peer.id not in [m['id'] for m in config['members']]:raise ValueError('This peer is not a calibration member')
    decode(config['initial_weights'])
    with peer.lock:
        if peer.active is not None:raise ValueError('An experiment or calibration is already active')
        peer.active='calibration'
    started=time.perf_counter()
    try:
        count=min(512,config['train_limit'])
        # Legacy one-peer indexing selects a valid isolated subset. Calibration
        # never changes the initial tensors or any run's optimizer state.
        payload=dict(dataset_id=config['dataset_id'],seed=config['seed'],threads=peer.threads,
            train_limit=config['train_limit'],peers=1,peer=0,epoch=0,start=0,count=count,
            batch_size=config['batch_size'],lr=config['lr'],momentum=config['momentum'],
            weights=config['initial_weights'],optimizer_momentum=None,numerics=config['numerics'])
        process_results=None
        if not peer.runner:
            process_results=isolated('calibrate',dict(training=payload,test_limit=config['test_limit']),peer.data,peer.capacity['cpu_affinity'])
        measurements=[]
        for repetition in range(4):
            before=time.perf_counter()
            result=peer.local_train(payload) if peer.runner else process_results['training'][repetition]
            elapsed=time.perf_counter()-before
            measured_us(result['wall_seconds'],positive=True)
            if result['count']!=count:raise ValueError('Calibration returned the wrong sample count')
            kernel=measured_us(result['train_seconds'],count,positive=True)
            # Python interpreter startup is calibration overhead, not persistent
            # process training setup. Fresh VM startup is a real per-task cost.
            execution=elapsed if peer.runner else result['wall_seconds']
            setup=measured_us(max(0,execution-result['train_seconds']))
            if repetition:measurements.append(dict(train_us=kernel,setup_us=setup))
        evaluation=(isolated('evaluate',dict(encoded=config['initial_weights'],seed=config['seed'],threads=peer.threads,
            test_limit=config['test_limit'],numerics=config['numerics']),peer.data,peer.capacity['cpu_affinity'])
            if peer.runner else process_results['evaluation'])
        cost=dict(train_us=int(statistics.median(m['train_us'] for m in measurements)),
            setup_us=int(statistics.median(m['setup_us'] for m in measurements)),
            evaluation_us=measured_us(evaluation['evaluation_seconds'],positive=True))
        cost.update(audit_us=cost['train_us'],audit_setup_us=cost['setup_us']);validate_cost(cost)
        record=dict(peer_id=peer.id,request_digest=digest(config),count=count,cost=cost,
            measurements=measurements,seconds=time.perf_counter()-started)
        # Admission checks this local record, so a proposer cannot substitute a
        # fabricated local calibration into a signed experiment configuration.
        with peer.lock:peer.calibrations[digest(config)]=record
        return record
    finally:
        with peer.lock:peer.active=None


if __name__=='__main__':
    from mtcp.host import pin_cpus
    from .training import train,evaluate
    request=json.load(sys.stdin);pin_cpus(request['cpus'])
    payload=request['payload']
    if request['operation']=='calibrate':
        training=[train(payload['training'],request['directory']) for _ in range(4)]
        p=payload['training']
        value=dict(training=training,evaluation=evaluate(p['weights'],request['directory'],p['seed'],p['threads'],payload['test_limit'],p['numerics']))
    elif request['operation']=='train':value=train(payload,request['directory'])
    elif request['operation']=='evaluate':value=evaluate(directory=request['directory'],**payload)
    else:raise ValueError('Unknown calibration operation')
    print(json.dumps(value,allow_nan=False))

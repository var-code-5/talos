"""Version-2 peer execution: agreed windows, exact replay and bounded adaptation."""
import hashlib
import json
import platform
import time

import numpy as np
import torch

from mtcp.kernel import digest
from .allocation import make_plan, smooth, validate_cost, validate_plan
from .audit import compare_update
from .calibration import measured_us
from .metrics import distribution, export_rounds, write_json
from .network import call
from .training import ARTIFACT, PARAMETERS, decode, encode, evaluate


def model_digest(encoded):
    return hashlib.sha256(decode(encoded).tobytes()).hexdigest()


class Exchange:
    """Local monotonic intervals: polling and successful HTTP are disjoint."""
    def __init__(self,peer):
        self.peer=peer
        self.seconds=0.;self.request_body_bytes=0;self.response_body_bytes=0
        self.polling_seconds=0.;self.transfer_seconds=0.;self.validation_seconds=0.

    def wait(self,member,path,predicate):
        started=time.perf_counter();deadline=time.monotonic()+180
        while not self.peer.stop.is_set() and time.monotonic()<deadline:
            try:
                value,metric=call(member['endpoint'],path,self.peer.key,timeout=min(10,max(.01,deadline-time.monotonic())))
                self.seconds+=metric['seconds'];self.request_body_bytes+=metric['request_body_bytes'];self.response_body_bytes+=metric['response_body_bytes']
                if value.get('status')=='failed':raise RuntimeError(f"Peer {member['name']} failed: {value.get('error')}")
                before=time.perf_counter();ready=predicate(value);self.validation_seconds+=time.perf_counter()-before
                if ready:
                    self.transfer_seconds+=metric['seconds']
                    self.polling_seconds+=max(0,time.perf_counter()-started-metric['seconds'])
                    return value
            except OSError:
                pass
            self.peer.stop.wait(.05)
        raise TimeoutError(f"Peer {member['name']} did not reach the allocation barrier")


def publish(peer,run,kind,key,record):
    with peer.lock:
        bucket=run[kind]
        if str(key) in bucket and bucket[str(key)]!=record:raise ValueError('Conflicting immutable evidence')
        bucket[str(key)]=record


def collect_evidence(peer,run,kind,key,own,exchange):
    publish(peer,run,kind,key,own)
    records=[]
    for member in run['config']['members']:
        if member['id']==peer.id:record=own
        else:
            response=exchange.wait(member,f"/v1/runs/{run['id']}/evidence/{kind}/{key}",lambda v:v.get('ready',False))
            record=response['record']
        if record.get('peer_id')!=member['id'] or record.get('round')!=key or record.get('run_id')!=run['id']:
            raise ValueError('Incorrectly bound peer evidence')
        records.append(record)
    return records


def observation(result,elapsed,audit_kernel,audit_setup,audit_count,audit_tasks,evaluation,previous,minimum_samples=1):
    cost=dict(previous)
    if result['count']>=minimum_samples:
        cost['train_us']=measured_us(result['train_seconds'],result['count'],positive=True)
        cost['setup_us']=measured_us(max(0,elapsed-result['train_seconds']))
    if audit_count>=minimum_samples:
        cost['audit_us']=measured_us(audit_kernel,audit_count,positive=True)
        cost['audit_setup_us']=measured_us(audit_setup,audit_tasks)
    cost['evaluation_us']=measured_us(evaluation['evaluation_seconds'],positive=True)
    validate_cost(cost)
    return cost


def execute(peer,run_id):
    run=peer.runs[run_id];config=run['config'];members=config['members']
    index=next(i for i,m in enumerate(members) if m['id']==peer.id)
    root=peer.directory/'runs'/run_id;root.mkdir(parents=True,exist_ok=True)
    write_json(root/'config.json',config)
    costs=[r['cost'] for r in config['calibration']];evidence=config['calibration'];previous=None
    totals=dict(network_request_body_bytes=0,network_response_body_bytes=0,audit_seconds=0.,train_seconds=0.,
        evaluation_seconds=0.,barrier_seconds=0.,completion_wait_seconds=0.,plan_agreement_seconds=0.,
        cpu_seconds=0.,samples=0,calibration_seconds=config['calibration'][index]['seconds'])
    current=config['initial_weights'];momentum=None;audit_momenta={};rtts=[]
    started=time.perf_counter()
    try:
        with peer.lock:run.update(status='running',started_at=time.time())
        for member in members:
            if member['id']!=peer.id:
                Exchange(peer).wait(member,f'/v1/runs/{run_id}',lambda v:v['status'] in ('ready','running','completed'))
        write_json(root/'initial.json',dict(weights=current,seed=config['seed'],model_digest=model_digest(current)))
        run['initial_weight_digest']=model_digest(current)
        initial=evaluate(current,peer.data,config['seed'],peer.threads,config['test_limit'],config['numerics'])
        run['initial_evaluation']=initial;totals['evaluation_seconds']+=initial['evaluation_seconds']
        round_id=0
        for epoch in range(config['epochs']):
            for offset in range(0,config['train_limit'],config['sync_samples']):
                if peer.stop.is_set():raise RuntimeError('Peer stopped')
                before_round=time.perf_counter();exchange=Exchange(peer);peer_rtts=[]
                for member in members:
                    if member['id']==peer.id:continue
                    _,metric=call(member['endpoint'],'/v1/ping',peer.key,{'run_id':run_id,'round':round_id})
                    peer_rtts.append(metric['seconds']*1000);rtts.append(metric['seconds']*1000)
                    exchange.seconds+=metric['seconds'];exchange.request_body_bytes+=metric['request_body_bytes'];exchange.response_body_bytes+=metric['response_body_bytes']
                plan=make_plan(config,epoch,round_id,offset,model_digest(current),costs,evidence,previous)
                plan_digest=digest(plan)
                before=time.perf_counter()
                acknowledgments=collect_evidence(peer,run,'plans',round_id,
                    dict(peer_id=peer.id,run_id=run_id,round=round_id,plan_digest=plan_digest),exchange)
                if any(a['plan_digest']!=plan_digest for a in acknowledgments):raise ValueError('Peers disagree on assignment plan')
                plan_seconds=time.perf_counter()-before
                write_json(root/'plans'/f'round-{round_id}.json',plan)
                with peer.lock:run['assignment']=dict(round=round_id,counts=plan['counts'],member_names=[m['name'] for m in members],
                    predicted_seconds=plan['predicted_phase_us']/1e6,costs=costs,audit_limited=plan['audit_limited'],plan_digest=plan_digest)
                count=plan['counts'][index]
                payload=dict(dataset_id=config['dataset_id'],seed=config['seed'],threads=peer.threads,
                    train_limit=config['train_limit'],peers=len(members),peer=index,epoch=epoch,
                    start=offset+sum(plan['counts'][:index]),count=count,batch_size=config['batch_size'],
                    lr=config['lr'],momentum=config['momentum'],weights=current,optimizer_momentum=momentum,
                    numerics=config['numerics'],indexing_version=2,plan=plan,plan_digest=plan_digest,
                    run_id=run_id,config_digest=run['config_digest'],round=round_id)
                validate_plan(plan,payload,model_digest(current))
                before=time.perf_counter()
                result=peer.local_train(payload) if count else dict(weights=current,optimizer_momentum=momentum,count=0,
                    train_loss=0.,train_accuracy=0.,train_seconds=0.,wall_seconds=0.,cpu_seconds=0.,peak_rss_mb=0.,numerics=config['numerics'])
                local_seconds=time.perf_counter()-before
                commit=dict(run_id=run_id,round=round_id,epoch=epoch,peer_id=peer.id,plan_digest=plan_digest,
                    input_digest=model_digest(current),artifact_digest=ARTIFACT,dataset_id=config['dataset_id'],
                    numerics=config['numerics'],start=payload['start'],result=result)
                write_json(root/'commits'/f'round-{round_id}.json',commit)
                with peer.lock:run['commits'][str(round_id)]=commit
                contributions=[];before=time.perf_counter();validation_seconds=0.
                poll_before=exchange.polling_seconds;transfer_before=exchange.transfer_seconds
                for other,member in enumerate(members):
                    if other==index:contribution=commit
                    else:
                        response=exchange.wait(member,f'/v1/runs/{run_id}/rounds/{round_id}',lambda v:v.get('ready',False))
                        contribution=response['commit']
                    checked=time.perf_counter()
                    if any(contribution.get(k)!=commit[k] for k in ('run_id','round','epoch','plan_digest','input_digest','artifact_digest','dataset_id','numerics')):
                        raise ValueError('Stale or incorrectly bound peer update')
                    if contribution.get('peer_id')!=member['id'] or contribution.get('start')!=offset+sum(plan['counts'][:other]):
                        raise ValueError('Incorrect contribution range/identity')
                    update=contribution['result'];decode(update['weights'])
                    if type(update['count']) is not int or update['count']!=plan['counts'][other] or update.get('numerics')!=config['numerics']:
                        raise ValueError('Incorrect contribution sample count or numerics')
                    if update['count']:decode(update['optimizer_momentum'])
                    elif update['weights']!=current or update['optimizer_momentum']!=audit_momenta.get(other):
                        raise ValueError('Zero-work peer changed its state')
                    contributions.append(contribution);validation_seconds+=time.perf_counter()-checked
                barrier_seconds=time.perf_counter()-before
                update_polling=exchange.polling_seconds-poll_before;update_transfer=exchange.transfer_seconds-transfer_before
                audit_started=time.perf_counter();audits=[];audit_kernel=0.;audit_setup=0.;audit_count=0;audit_tasks=0
                if config['audit']=='full':
                    for other,contribution in enumerate(contributions):
                        if other==index or not contribution['result']['count']:continue
                        remote_payload=dict(payload,peer=other,start=contribution['start'],count=contribution['result']['count'],
                            optimizer_momentum=audit_momenta.get(other))
                        replay_started=time.perf_counter();reference=peer.local_train(remote_payload)
                        replay_seconds=time.perf_counter()-replay_started
                        audit_kernel+=reference['train_seconds'];audit_setup+=max(0,replay_seconds-reference['train_seconds'])
                        audit_count+=reference['count'];audit_tasks+=1
                        check=dict(compare_update(reference,contribution['result']),peer_id=members[other]['id'],round=round_id,numerics=config['numerics'])
                        write_json(root/'audits'/f'round-{round_id}-member-{other}.json',check)
                        with peer.lock:run['last_audit']=check
                        audits.append(check)
                        if not check['passed']:raise ValueError(f"Independent local-SGD audit failed: {check}")
                audit_seconds=time.perf_counter()-audit_started if config['audit']=='full' else 0.
                accumulated=np.zeros(PARAMETERS,dtype=np.float64)
                for other,contribution in enumerate(contributions):
                    update=contribution['result']
                    if update['count']:accumulated+=decode(update['weights']).astype(np.float64)*update['count']/plan['total']
                    audit_momenta[other]=update['optimizer_momentum']
                current=encode(accumulated);momentum=result['optimizer_momentum']
                evaluation=evaluate(current,peer.data,config['seed'],peer.threads,config['test_limit'],config['numerics'])
                raw=observation(result,local_seconds,audit_kernel,audit_setup,audit_count,audit_tasks,evaluation,costs[index],config['batch_size'])
                before=time.perf_counter()
                evidence=collect_evidence(peer,run,'completions',round_id,
                    dict(peer_id=peer.id,run_id=run_id,round=round_id,plan_digest=plan_digest,model_digest=model_digest(current),cost=raw),exchange)
                completion_seconds=time.perf_counter()-before
                if any(e['model_digest']!=model_digest(current) or e['plan_digest']!=plan_digest for e in evidence):
                    raise ValueError('Peers disagree on completed round model or plan')
                for e in evidence:validate_cost(e['cost'])
                record=dict(round=round_id,epoch=epoch+1,numerics=config['numerics'],allocation=config['allocation'],
                    indexing_version=2,local_samples=count,global_samples=plan['total'],assigned_samples=plan['counts'],
                    plan_digest=plan_digest,measured_costs=costs,observed_cost=raw,predicted_phase_seconds=plan['predicted_phase_us']/1e6,
                    audit_limited=plan['audit_limited'],kernel_samples_per_second=count/max(result['train_seconds'],1e-9),
                    train_loss=sum(c['result']['train_loss']*c['result']['count'] for c in contributions)/plan['total'],
                    train_accuracy=sum(c['result']['train_accuracy']*c['result']['count'] for c in contributions)/plan['total'],
                    test_accuracy=evaluation['test_accuracy'],test_loss=evaluation['test_loss'],macro_f1=evaluation['macro_f1'],
                    local_compute_seconds=local_seconds,kernel_train_seconds=result['train_seconds'],audit_seconds=audit_seconds,
                    evaluation_seconds=evaluation['evaluation_seconds'],barrier_seconds=barrier_seconds,
                    update_polling_seconds=update_polling,update_transfer_seconds=update_transfer,
                    update_validation_seconds=validation_seconds,completion_wait_seconds=completion_seconds,
                    plan_agreement_seconds=plan_seconds,network_seconds=exchange.seconds,
                    request_body_bytes=exchange.request_body_bytes,response_body_bytes=exchange.response_body_bytes,
                    effective_http_body_mbps=(exchange.request_body_bytes+exchange.response_body_bytes)*8/max(exchange.seconds,1e-9)/1e6,
                    rtt_mean_ms=sum(peer_rtts)/len(peer_rtts) if peer_rtts else 0.,round_wall_seconds=time.perf_counter()-before_round,elapsed_seconds=time.perf_counter()-started,
                    local_cpu_seconds=result['cpu_seconds'],local_peak_rss_mb=result['peak_rss_mb'],model_digest=model_digest(current),
                    audit_max_weight_error=max((a['weights']['max_abs'] for a in audits),default=None),
                    audit_max_momentum_error=max((a['optimizer_momentum']['max_abs'] for a in audits),default=None))
                costs=[smooth(c,e['cost']) for c,e in zip(costs,evidence)]
                # A one-example epoch tail must not redefine the next full
                # window as 100% work on whichever peer received that example.
                if plan['total']==config['sync_samples'] or previous is None:previous=plan['counts']
                write_json(root/'evidence'/f'round-{round_id}.json',evidence)
                with peer.lock:run['rounds'].append(record)
                for k,v in [('train_seconds',local_seconds),('audit_seconds',audit_seconds),('evaluation_seconds',evaluation['evaluation_seconds']),
                    ('barrier_seconds',barrier_seconds),('completion_wait_seconds',completion_seconds),('plan_agreement_seconds',plan_seconds),
                    ('samples',count),('cpu_seconds',result['cpu_seconds']),('network_request_body_bytes',exchange.request_body_bytes),
                    ('network_response_body_bytes',exchange.response_body_bytes)]:totals[k]+=v
                with open(root/'rounds.jsonl','a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
                write_json(root/'checkpoint.json',dict(weights=current,optimizer_momentum=momentum,round=round_id,
                    config_digest=run['config_digest'],plan_digest=plan_digest))
                print(f"{peer.name}: epoch {epoch+1}/{config['epochs']} round {round_id} samples={count} accuracy={evaluation['test_accuracy']:.2%}",flush=True)
                round_id+=1
        elapsed=time.perf_counter()-started
        with peer.lock:
            run.update(status='completed',finished_at=time.time(),wall_seconds=elapsed,end_to_end_seconds=elapsed+config['startup_seconds'],
                final_evaluation=evaluation,final_model_digest=model_digest(current),totals=totals,rtt_ms=distribution(rtts),rtt_samples_ms=rtts,
                local_training_samples_per_second=totals['samples']/elapsed,
                global_training_samples_per_second=config['train_limit']*config['epochs']/elapsed,
                environment=dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),platform=platform.platform()),
                topology='single-host-peers' if len({m['host_id'] for m in members})==1 else 'separate-host-peers')
    except Exception as exc:
        with peer.lock:run.update(status='failed',error=str(exc),finished_at=time.time())
        print(f'{peer.name}: run failed: {exc}',flush=True)
    finally:
        write_json(root/'summary.json',peer.public(run_id));export_rounds(root/'rounds.csv',run['rounds'])
        with peer.lock:peer.active=None

"""Symmetric CIFAR peers: each trains, publishes a commit, audits, and aggregates."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import secrets
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit

import numpy as np
import torch
from mtcp.kernel import canonical,digest
from mtcp.protocol import sign
from mtcp.resources import advertise
from mtcp.host import available_cpus, pin_cpus
from .data import load_manifest
from .metrics import distribution,write_json,export_rounds
from .network import Discovery,call,discover_host
from .training import ARTIFACT,PARAMETERS,NUMERICS,model,weights,decode,encode,train,evaluate
from .audit import compare_update


class Peer:
    def __init__(self,directory,data,key,name,cpus,threads=1,backend='process',assets=None,memory_mb=768,affinity_enforced=False):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        identity_path=self.directory/'identity.json'
        if not identity_path.exists():write_json(identity_path,{'id':uuid.uuid4().hex})
        self.id=json.loads(identity_path.read_text())['id'];self.name=name
        self.data=str(Path(data).resolve());self.manifest=load_manifest(self.data,verify=True)
        self.key=key;self.threads=threads;self.backend=backend;self.stop=threading.Event()
        self.lock=threading.RLock();self.peers={};self.runs={};self.active=None;self.links={};self.calibrations={}
        self.capacity=advertise(cpus,memory_mb,100,100,'kvm-microvm' if backend=='firecracker' else 'trusted-cifar-process')
        self.capacity.update(offered_upload_mbps=None,offered_download_mbps=None,
            bandwidth_source='Uncapped; measured peer links are advertised separately',
            affinity_enforced=affinity_enforced,os=platform.system(),architecture=platform.machine(),
            cpu_description=platform.processor() or platform.machine())
        self.runner=None
        if backend=='firecracker':
            if platform.system()!='Linux':
                raise ValueError('Firecracker requires Linux/KVM. Use --backend process on Windows/macOS.')
            from mtcp.firecracker import Firecracker
            self.runner=Firecracker(assets or '.mtcp/firecracker-cifar',cpus,memory_mb)
            guest_manifest=json.loads((self.runner.assets/'manifest.json').read_text())
            if guest_manifest.get('cifar_artifact_digest')!=ARTIFACT or guest_manifest.get('dataset_id')!=self.manifest['dataset_id']:
                raise ValueError('Firecracker image lacks the matching CIFAR kernel/dataset; prepare the CIFAR image first')
            self.runner.execute(operation='probe')
        self.endpoint=None
        self.auto_advertise=False
        self.http_port=None
        for saved in self.directory.glob('runs/*/summary.json'):
            value=json.loads(saved.read_text())
            if value.get('status') in ('completed','failed'):
                self.runs[value['id']]=dict(value,commits={})

    def description(self):
        with self.lock:
            self.refresh_endpoint()
            links=copy.deepcopy(self.links)
        return dict(id=self.id,name=self.name,endpoint=self.endpoint,host_id=self.capacity['host_id'],
                    resources=self.capacity,threads=self.threads,backend=self.backend,
                    dataset_id=self.manifest['dataset_id'],artifact_digest=ARTIFACT,
                    torch_version=torch.__version__,numpy_version=np.__version__,python_version=platform.python_version(),
                    measured_links=links,numerics_supported=list(NUMERICS),allocation_protocols=[1],indexing_versions=[1,2])

    def refresh_endpoint(self):
        # A run's membership is fixed. Refresh DHCP changes only between runs.
        if self.auto_advertise and self.http_port is not None and self.active is None:
            self.endpoint=f'http://{discover_host()}:{self.http_port}'

    def remember(self,peer):
        if peer['id']==self.id:return
        origin=urlsplit(peer['endpoint'])
        if origin.scheme not in ('http','https') or not origin.hostname or origin.path or origin.username or origin.password:
            raise ValueError('Invalid peer endpoint')
        if len(peer['id'])>128:return
        with self.lock:self.peers[peer['id']]=dict(peer,last_seen=time.time())

    def members(self):
        with self.lock:
            return [self.description()]+[copy.deepcopy(p) for p in self.peers.values() if time.time()-p['last_seen']<10]

    def propose(self,config):
        validate_config(config)
        members=config['members'];ids=[p['id'] for p in members]
        if self.id not in ids:raise ValueError('This peer is not a member of the run')
        if config['dataset_id']!=self.manifest['dataset_id'] or config['artifact_digest']!=ARTIFACT:
            raise ValueError('Dataset or training artifact mismatch')
        if config['threads']!=self.threads:raise ValueError('Thread allocation differs from experiment')
        if config.get('numerics','fp32-native') not in NUMERICS:raise ValueError('Unsupported numerical profile')
        if any(p['backend']!=self.backend or p['torch_version'].split('+')[0]!=torch.__version__.split('+')[0] or p['numpy_version']!=np.__version__ for p in members):
            raise ValueError('All peers must use identical backend and PyTorch/NumPy versions')
        mine=next(p for p in members if p['id']==self.id)
        if mine['endpoint']!=self.endpoint or mine['host_id']!=self.capacity['host_id']:
            raise ValueError('Local membership identity mismatch')
        if config.get('indexing_version',1)==2:
            from .allocation import SETTINGS
            if any(1 not in p.get('allocation_protocols',[]) for p in members):
                raise ValueError('All peers need an allocation protocol update and restart')
            if config.get('planner')!=SETTINGS:raise ValueError('Unsupported planner settings')
            local_record=config['calibration'][ids.index(self.id)]
            expected=self.calibrations.get(config['calibration_request_digest'])
            if local_record!=expected:raise ValueError('Local calibration was changed or is unavailable; recalibrate')
        run_id=config['run_id'];fingerprint=digest(config)
        with self.lock:
            if run_id in self.runs:
                if self.runs[run_id]['config_digest']!=fingerprint:raise ValueError('Conflicting run proposal')
                return self.public(run_id)
            if self.active is not None:raise ValueError('Another run is already active')
            run=dict(id=run_id,status='ready',config=config,config_digest=fingerprint,rounds=[],commits={},
                     created_at=time.time(),peer=self.description(),error=None,plans={},completions={})
            self.runs[run_id]=run;self.active=run_id
        threading.Thread(target=self.execute,args=(run_id,),daemon=True).start()
        return self.public(run_id)

    def public(self,run_id):
        with self.lock:
            run=self.runs[run_id]
            return dict(copy.deepcopy({k:v for k,v in run.items() if k not in ('commits','weights','optimizer_momentum','plans','completions')}),
                        accepting_runs=self.active is None)

    def local_train(self,payload):
        if self.runner:
            return self.runner.execute(payload,timeout=120,operation='cifar-train',memory_mb=self.capacity['offered_memory_mb'])
        return train(payload,self.data)

    def wait_for(self,member,path,deadline,predicate):
        total=dict(seconds=0.,request_body_bytes=0,response_body_bytes=0)
        while not self.stop.is_set() and time.monotonic()<deadline:
            try:
                value,metric=call(member['endpoint'],path,self.key,timeout=3)
                for k in total:total[k]+=metric[k]
                if value.get('status')=='failed':raise RuntimeError(f"Peer {member['name']} failed: {value.get('error')}")
                if predicate(value):return value,total
            except OSError:
                pass
            self.stop.wait(.1)
        raise TimeoutError(f"Peer {member['name']} did not reach the round barrier")

    def execute(self,run_id):
        if self.runs[run_id]['config'].get('indexing_version',1)==2:
            from .execution import execute
            return execute(self,run_id)
        run=self.runs[run_id];config=run['config'];members=config['members'];index=[p['id'] for p in members].index(self.id)
        root=self.directory/'runs'/run_id;root.mkdir(parents=True,exist_ok=True)
        (root/'commits').mkdir(exist_ok=True)
        (root/'audits').mkdir(exist_ok=True)
        write_json(root/'config.json',config)
        totals=dict(network_request_body_bytes=0,network_response_body_bytes=0,audit_seconds=0.,
                    train_seconds=0.,evaluation_seconds=0.,barrier_seconds=0.,cpu_seconds=0.,samples=0)
        rtts=[];network=[];audit_momenta={};momentum=None
        numerics=config.get('numerics','fp32-native')
        try:
            for member in members:
                if member['id']!=self.id:
                    self.wait_for(member,f'/v1/runs/{run_id}',time.monotonic()+60,
                                  lambda value:value['status'] in ('ready','running','completed'))
            initial=config.get('initial_weights') or weights(model(config['seed'],self.threads));current=initial
            write_json(root/'initial.json',{'weights':initial,'seed':config['seed'],
                'model_digest':hashlib.sha256(decode(initial).tobytes()).hexdigest()})
            with self.lock:
                run['initial_weight_digest']=hashlib.sha256(decode(initial).tobytes()).hexdigest()
                run['status']='running';run['started_at']=time.time()
            started=time.perf_counter()
            per_round=max(1,config['sync_samples']//len(members));round_id=0
            shard_sizes=[len(range(i,config['train_limit'],len(members))) for i in range(len(members))]
            initial_eval=evaluate(current,self.data,config['seed'],self.threads,config['test_limit'],numerics)
            with self.lock:run['initial_evaluation']=initial_eval
            totals['evaluation_seconds']+=initial_eval['evaluation_seconds']
            for epoch in range(config['epochs']):
                for start in range(0,max(shard_sizes),per_round):
                    if self.stop.is_set():raise RuntimeError('Peer stopped')
                    round_started=time.perf_counter();round_network_seconds=0.;round_request=0;round_response=0
                    peer_rtts=[]
                    for member in members:
                        if member['id']==self.id:continue
                        _,metric=call(member['endpoint'],'/v1/ping',self.key,dict(nonce=secrets.token_hex(16)))
                        rtts.append(metric['seconds']*1000);peer_rtts.append(metric['seconds']*1000)
                        round_request+=metric['request_body_bytes'];round_response+=metric['response_body_bytes'];round_network_seconds+=metric['seconds']
                    count=min(per_round,max(0,shard_sizes[index]-start))
                    payload=dict(dataset_id=config['dataset_id'],seed=config['seed'],threads=self.threads,
                        train_limit=config['train_limit'],peers=len(members),peer=index,epoch=epoch,start=start,count=count,
                        batch_size=config['batch_size'],lr=config['lr'],momentum=config['momentum'],weights=current,
                        optimizer_momentum=momentum,numerics=numerics)
                    local_started=time.perf_counter()
                    result=self.local_train(payload) if count else dict(weights=current,optimizer_momentum=momentum,
                        count=0,train_loss=0.,train_accuracy=0.,train_seconds=0.,wall_seconds=0.,cpu_seconds=0.,peak_rss_mb=0.,numerics=numerics)
                    local_seconds=time.perf_counter()-local_started
                    commit=dict(run_id=run_id,round=round_id,peer_id=self.id,epoch=epoch,start=start,
                        input_digest=digest({'weights':current,'epoch':epoch,'start':start}),artifact_digest=ARTIFACT,
                        dataset_id=config['dataset_id'],numerics=numerics,result=result)
                    write_json(root/'commits'/f'round-{round_id}.json',commit)
                    with self.lock:run['commits'][str(round_id)]=commit
                    contributions=[None]*len(members);contributions[index]=commit
                    waited=time.perf_counter()
                    for other,member in enumerate(members):
                        if other==index:continue
                        value,metric=self.wait_for(member,f'/v1/runs/{run_id}/rounds/{round_id}',time.monotonic()+180,
                                                  lambda value:value.get('ready',False))
                        contribution=value['commit'];expected=min(per_round,max(0,shard_sizes[other]-start))
                        if any(contribution.get(k)!=commit[k] for k in ('run_id','round','epoch','start','input_digest','artifact_digest','dataset_id','numerics')) or contribution['peer_id']!=member['id']:
                            raise ValueError('Stale or incorrectly bound peer update')
                        received=contribution['result'];decode(received['weights'])
                        if received.get('numerics','fp32-native')!=numerics:raise ValueError('Peer used a different numerical profile')
                        if received['count']!=expected:raise ValueError('Incorrect contribution sample count')
                        decode(received['optimizer_momentum']) if expected else None
                        contributions[other]=contribution
                        round_request+=metric['request_body_bytes'];round_response+=metric['response_body_bytes'];round_network_seconds+=metric['seconds']
                    barrier_seconds=time.perf_counter()-waited
                    audit_started=time.perf_counter()
                    audit_evidence=[]
                    if config['audit']=='full':
                        for other,contribution in enumerate(contributions):
                            if other==index or not contribution['result']['count']:continue
                            audit_payload=dict(payload,peer=other,count=contribution['result']['count'],optimizer_momentum=audit_momenta.get(other))
                            reference=self.local_train(audit_payload)
                            evidence=dict(compare_update(reference,contribution['result']),round=round_id,
                                peer_id=members[other]['id'],numerics=numerics)
                            write_json(root/'audits'/f'round-{round_id}-member-{other}.json',evidence)
                            with self.lock:run['last_audit']=evidence
                            audit_evidence.append(evidence)
                            if not evidence['passed']:
                                raise ValueError(f"Independent local-SGD audit failed ({numerics}): "
                                    f"weights max_abs={evidence['weights']['max_abs']:.6g}; "
                                    f"momentum max_abs={evidence['optimizer_momentum']['max_abs']:.6g}. "
                                    'Detailed evidence saved under the run audits directory.')
                    audit_seconds=time.perf_counter()-audit_started if config['audit']=='full' else 0.
                    total=sum(c['result']['count'] for c in contributions)
                    # Fixed participant ordering gives every peer the same independent aggregation.
                    accumulated=np.zeros(PARAMETERS,dtype=np.float64)
                    for other,contribution in enumerate(contributions):
                        update=contribution['result'];accumulated+=decode(update['weights']).astype(np.float64)*update['count']/total
                        audit_momenta[other]=update['optimizer_momentum']
                    current=encode(accumulated);momentum=result['optimizer_momentum']
                    evaluation=evaluate(current,self.data,config['seed'],self.threads,config['test_limit'],numerics)
                    round_wall=time.perf_counter()-round_started
                    record=dict(round=round_id,epoch=epoch+1,numerics=numerics,local_samples=count,global_samples=total,
                        train_loss=sum(c['result']['train_loss']*c['result']['count'] for c in contributions)/total,
                        train_accuracy=sum(c['result']['train_accuracy']*c['result']['count'] for c in contributions)/total,
                        test_accuracy=evaluation['test_accuracy'],test_loss=evaluation['test_loss'],macro_f1=evaluation['macro_f1'],
                        local_compute_seconds=local_seconds,kernel_train_seconds=result['train_seconds'],
                        audit_seconds=audit_seconds,evaluation_seconds=evaluation['evaluation_seconds'],
                        audit_max_weight_error=max((e['weights']['max_abs'] for e in audit_evidence),default=None),
                        audit_max_momentum_error=max((e['optimizer_momentum']['max_abs'] for e in audit_evidence),default=None),
                        barrier_seconds=barrier_seconds,network_seconds=round_network_seconds,
                        rtt_mean_ms=sum(peer_rtts)/len(peer_rtts) if peer_rtts else 0.,
                        request_body_bytes=round_request,response_body_bytes=round_response,
                        effective_http_body_mbps=(round_request+round_response)*8/max(round_network_seconds,1e-9)/1e6,
                        round_wall_seconds=round_wall,elapsed_seconds=time.perf_counter()-started,
                        local_cpu_seconds=result['cpu_seconds'],local_peak_rss_mb=result['peak_rss_mb'],
                        model_digest=hashlib.sha256(decode(current).tobytes()).hexdigest())
                    with self.lock:run['rounds'].append(record)
                    totals['network_request_body_bytes']+=round_request;totals['network_response_body_bytes']+=round_response
                    for k,v in [('audit_seconds',audit_seconds),('train_seconds',local_seconds),('evaluation_seconds',evaluation['evaluation_seconds']),('barrier_seconds',barrier_seconds),('cpu_seconds',result['cpu_seconds']),('samples',count)]:totals[k]+=v
                    with open(root/'rounds.jsonl','a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
                    write_json(root/'checkpoint.json',dict(weights=current,optimizer_momentum=momentum,round=round_id,config_digest=run['config_digest']))
                    print(f"{self.name}: epoch {epoch+1}/{config['epochs']} round {round_id} accuracy={evaluation['test_accuracy']:.2%} loss={evaluation['test_loss']:.4f}",flush=True)
                    round_id+=1
            elapsed=time.perf_counter()-started
            with self.lock:
                run.update(status='completed',finished_at=time.time(),wall_seconds=elapsed,final_evaluation=evaluation,
                    totals=totals,rtt_ms=distribution(rtts),rtt_samples_ms=rtts,local_training_samples_per_second=totals['samples']/elapsed,
                    global_training_samples_per_second=config['train_limit']*config['epochs']/elapsed,
                    final_model_digest=hashlib.sha256(decode(current).tobytes()).hexdigest(),
                    environment=dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),platform=platform.platform()),
                    topology='single-host-peers' if len({p['host_id'] for p in members})==1 else 'separate-host-peers')
        except Exception as exc:
            with self.lock:run.update(status='failed',error=str(exc),finished_at=time.time())
            print(f'{self.name}: run failed: {exc}',flush=True)
        finally:
            write_json(root/'summary.json',self.public(run_id));export_rounds(root/'rounds.csv',run['rounds'])
            with self.lock:self.active=None


def validate_config(config,require_calibration=True):
    for key,low,high in [('epochs',1,100),('train_limit',2,50000),('test_limit',1,10000),('batch_size',1,512),('sync_samples',2,50000),('threads',1,32),('seed',0,2**31-1)]:
        if type(config.get(key)) is not int or not low<=config[key]<=high:raise ValueError(f'Invalid {key}')
    for key,low,high in [('lr',0,1),('momentum',0,1)]:
        value=config.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<high or (key=='lr' and value==0):raise ValueError(f'Invalid {key}')
    if config.get('audit') not in ('full','none'):raise ValueError('Unknown audit policy')
    if config.get('numerics','fp32-native') not in NUMERICS:raise ValueError('Unknown numerical profile')
    if config.get('initial_weights') is not None:decode(config['initial_weights'])
    members=config.get('members',[])
    if not 1<=len(members)<=8 or len({p['id'] for p in members})!=len(members):raise ValueError('Invalid membership')
    if config['train_limit']<len(members):raise ValueError('Too few samples for membership')
    if config['sync_samples']<len(members):raise ValueError('Sync samples must cover all peers')
    version=config.get('indexing_version',1)
    if type(version) is not int or version not in (1,2):raise ValueError('Unsupported indexing version')
    policy=config.get('allocation','equal')
    if policy not in ('equal','compute-aware') or (policy=='compute-aware' and version!=2):raise ValueError('Invalid allocation policy/indexing version')
    if version==2:
        from .allocation import SETTINGS,validate_cost
        if config.get('planner')!=SETTINGS:raise ValueError('Unsupported planner settings')
        if require_calibration:
            request={k:v for k,v in config.items() if k not in ('calibration','calibration_request_digest','startup_seconds')}
            if digest(request)!=config.get('calibration_request_digest'):raise ValueError('Calibration configuration was changed')
            calibration=config.get('calibration',[])
            if len(calibration)!=len(members):raise ValueError('Missing calibration records')
            for member,record in zip(members,calibration):
                if record.get('peer_id')!=member['id'] or record.get('request_digest')!=config.get('calibration_request_digest'):
                    raise ValueError('Calibration identity mismatch')
                validate_cost(record['cost'])
                from .calibration import measured_us
                measured_us(record['seconds'],positive=True)
            from .calibration import measured_us
            measured_us(config.get('startup_seconds'),positive=True)
        planned_rounds=config['epochs']*math.ceil(config['train_limit']/config['sync_samples'])
    else:
        planned_rounds=config['epochs']*math.ceil(math.ceil(config['train_limit']/len(members))/(config['sync_samples']//len(members)))
    if planned_rounds>500:raise ValueError('This prototype supports at most 500 rounds; increase --sync-samples')
    if members!=sorted(members,key=lambda p:p['id']):raise ValueError('Membership ordering must be deterministic')
    if not isinstance(config.get('run_id'),str) or not config['run_id'].isalnum() or len(config['run_id'])>64:raise ValueError('Invalid run ID')


def handler_for(peer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_):pass
        def send(self,value,status=200,authenticated=True):
            if authenticated:
                nonce=self.headers.get('X-MTCP-Nonce','')
                value={'value':value,'nonce':nonce,'mac':sign({'nonce':nonce,'value':value},peer.key)}
            raw=canonical(value);self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
        def authorized(self):
            return secrets.compare_digest(self.headers.get('Authorization',''),f'Bearer {peer.key}')
        def do_GET(self):
            if self.path in ('/','/ui/cifar.js','/ui/cifar.css'):
                name={'/':'cifar.html','/ui/cifar.js':'cifar.js','/ui/cifar.css':'cifar.css'}[self.path]
                content=(Path(__file__).resolve().parents[2]/'web'/name).read_bytes()
                kind={'cifar.html':'text/html; charset=utf-8','cifar.js':'text/javascript','cifar.css':'text/css'}[name]
                self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(content)))
                self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'")
                self.end_headers();self.wfile.write(content);return
            if self.path=='/v1/status':
                with peer.lock:
                    summaries=[]
                    for run in sorted(peer.runs.values(),key=lambda r:r['created_at'],reverse=True):
                        summaries.append(dict(id=run['id'],status=run['status'],error=run.get('error'),
                            rounds=len(run['rounds']),latest=run['rounds'][-1] if run['rounds'] else None,
                            epochs=run['config']['epochs'],train_limit=run['config']['train_limit'],
                            peers=len(run['config']['members']),audit=run['config']['audit'],
                            numerics=run['config'].get('numerics','fp32-native'),backend=run['peer']['backend'],
                            allocation=run['config'].get('allocation','equal'),assignment=run.get('assignment')))
                self.send({'self':peer.description(),'peers':peer.members(),'runs':summaries},authenticated=False);return
            if self.path=='/health':self.send({'status':'ok'},authenticated=False);return
            if self.path=='/v1/resources':self.send(peer.description(),authenticated=False);return
            if self.path=='/v1/peers':self.send({'peers':peer.members()},authenticated=False);return
            if not self.authorized():self.send({'error':'Invalid experiment key'},401);return
            parts=self.path.strip('/').split('/')
            try:
                if len(parts)==3 and parts[:2]==['v1','runs']:self.send(peer.public(parts[2]));return
                if len(parts)==6 and parts[:2]==['v1','runs'] and parts[3]=='evidence':
                    if parts[4] not in ('plans','completions') or not parts[5].isdigit():raise ValueError('Invalid evidence request')
                    with peer.lock:
                        run=peer.runs[parts[2]];record=run.get(parts[4],{}).get(parts[5])
                        response=dict(ready=record is not None,record=record,status=run['status'],error=run.get('error'))
                    self.send(response);return
                if len(parts)==5 and parts[:2]==['v1','runs'] and parts[3]=='rounds':
                    if not parts[4].isdigit():self.send({'error':'Invalid round'},400);return
                    with peer.lock:
                        run=peer.runs[parts[2]];commit=run['commits'].get(parts[4])
                        saved=peer.directory/'runs'/parts[2]/'commits'/f'round-{int(parts[4])}.json'
                        if commit is None and saved.exists():commit=json.loads(saved.read_text())
                        response={'ready':commit is not None,'commit':commit,'status':run['status'],'error':run.get('error')}
                    self.send(response);return
                self.send({'error':'Not found'},404)
            except KeyError:self.send({'error':'Unknown run'},404)
            except (ValueError,TypeError) as exc:self.send({'error':str(exc)},400)
        def do_POST(self):
            if not self.authorized():self.send({'error':'Invalid experiment key'},401);return
            self.connection.settimeout(10)
            try:
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<=300000:raise ValueError('Invalid control request size')
                value=json.loads(self.rfile.read(length))
                if self.path=='/v1/hello':peer.remember(value['peer']);self.send({'peer':peer.description()})
                elif self.path=='/v1/ping':self.send({'echo':value,'peer_id':peer.id})
                elif self.path=='/v1/initialize':
                    seed=value.get('seed')
                    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Invalid seed')
                    with peer.lock:
                        if peer.active is not None:raise ValueError('An experiment is already active')
                        initial=weights(model(seed,peer.threads))
                    self.send({'weights':initial,'seed':seed})
                elif self.path=='/v1/bandwidth':
                    count=value.get('download_bytes',0)
                    if type(count) is not int or not 0<=count<=131072:raise ValueError('Invalid probe size')
                    self.send({'peer_id':peer.id,'padding':'x'*count})
                elif self.path=='/v1/calibrate':
                    from .calibration import calibrate
                    self.send(calibrate(peer,value))
                elif self.path=='/v1/runs':self.send(peer.propose(value),202)
                else:self.send({'error':'Not found'},404)
            except (ValueError,KeyError,TypeError) as exc:self.send({'error':str(exc)},400)
    return Handler


def serve(args):
    key=os.environ.get('MTCP_CLUSTER_KEY','')
    if len(key)<32:raise ValueError('Set MTCP_CLUSTER_KEY to the same random key (at least 32 characters) on all intended peers')
    if not 1<=args.threads<=32:raise ValueError('Threads must be between 1 and 32')
    cpus=[int(c) for c in args.cpus.split(',')] if args.cpus else available_cpus()[:args.threads]
    if len(set(cpus))!=len(cpus) or len(cpus)<args.threads or not set(cpus)<=set(available_cpus()):raise ValueError('Invalid CPU allocation')
    affinity_enforced=pin_cpus(cpus)
    peer=Peer(args.state_dir or f'.mtcp/cifar-peer-{args.port}',args.data,key,args.name,cpus,args.threads,args.backend,args.assets,args.memory_mb,affinity_enforced)
    server=ThreadingHTTPServer((args.host,args.port),handler_for(peer));peer.endpoint=f'http://{args.advertise_host or discover_host()}:{server.server_port}'
    peer.http_port=server.server_port;peer.auto_advertise=args.advertise_host is None
    def explicit_hello():
        while not peer.stop.is_set():
            for endpoint in args.connect:
                try:
                    value,_=call(endpoint,'/v1/hello',key,{'peer':peer.description()});peer.remember(value['peer'])
                except (OSError,ValueError,KeyError):pass
            peer.stop.wait(2)
    def measure_links():
        while not peer.stop.is_set():
            if peer.active is None:
                for member in peer.members():
                    if member['id']==peer.id or time.time()-peer.links.get(member['id'],{}).get('measured_at',0)<30:continue
                    try:
                        _,ping=call(member['endpoint'],'/v1/ping',key,{'nonce':secrets.token_hex(16)})
                        _,down=call(member['endpoint'],'/v1/bandwidth',key,{'download_bytes':131072})
                        _,up=call(member['endpoint'],'/v1/bandwidth',key,{'padding':'x'*131072})
                        with peer.lock:
                            peer.links[member['id']]=dict(rtt_ms=ping['seconds']*1000,
                                upload_mbps=up['request_body_bytes']*8/up['seconds']/1e6,
                                download_mbps=down['response_body_bytes']*8/down['seconds']/1e6,
                                measured_at=time.time(),method='128 KiB authenticated HTTP body transfers, including RTT')
                    except (OSError,ValueError,KeyError):pass
            peer.stop.wait(2)
    if not args.no_discovery:
        discovery=Discovery(peer.description,peer.remember,key,peer.stop,args.discovery_port);discovery.start()
    threading.Thread(target=explicit_hello,daemon=True).start()
    threading.Thread(target=measure_links,daemon=True).start()
    print(f'{peer.name} [{peer.id}] advertising at {peer.endpoint}; every peer trains and aggregates',flush=True)
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:peer.stop.set();server.server_close()

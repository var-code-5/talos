"""Fixed project1 CNN and deterministic local-SGD rounds; no executable wire objects."""
import base64
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .data import arrays, load_manifest
from mtcp.host import peak_rss_mb
from .allocation import validate_plan

PARAMETERS = 22058
ARTIFACT = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name('allocation.py').read_bytes()).hexdigest()
NUMERICS = ('fp32-native', 'fp64-portable')


def model(seed=42, threads=1, numerics='fp32-native'):
    if numerics not in NUMERICS:
        raise ValueError('Unknown numerical profile')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1) if torch.get_num_interop_threads() != 1 else None
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = numerics == 'fp32-native'
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Conv2d(3,16,5),nn.MaxPool2d(2,2),nn.ReLU(),
        nn.Conv2d(16,32,5),nn.MaxPool2d(2,2),nn.ReLU(),nn.Flatten(),nn.Linear(32*5*5,10))
    return net.double() if numerics == 'fp64-portable' else net


def encode(values):
    values = np.asarray(values, dtype='<f4').reshape(-1)
    if values.size != PARAMETERS or not np.isfinite(values).all():
        raise ValueError('Invalid model/momentum tensor')
    return base64.b64encode(values.tobytes()).decode('ascii')


def decode(encoded):
    if not isinstance(encoded,str) or len(encoded) > PARAMETERS*8:
        raise ValueError('Invalid tensor envelope size')
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != PARAMETERS*4:
        raise ValueError('Wrong tensor byte length')
    result = np.frombuffer(raw,dtype='<f4').copy()
    if not np.isfinite(result).all():
        raise ValueError('Non-finite tensor')
    return result


def weights(net):
    return encode(torch.cat([p.detach().flatten() for p in net.parameters()]).numpy())


def set_weights(net, encoded):
    flat = decode(encoded)
    offset = 0
    with torch.no_grad():
        for p in net.parameters():
            count = p.numel()
            p.copy_(torch.from_numpy(flat[offset:offset+count].reshape(p.shape)))
            offset += count


def indices(seed, train_limit, peers, peer, epoch, start, count):
    if not 0 <= peer < peers <= 32 or not 1 <= train_limit <= 50000 or not 0 <= epoch <= 10000:
        raise ValueError('Invalid dataset partition')
    # Fixed initial selection; epoch ordering changes without changing membership.
    chosen = np.random.default_rng(seed).permutation(50000)[:train_limit]
    partition = chosen[peer::peers]
    order = np.random.default_rng(seed + 1009*epoch + 97*peer).permutation(partition)
    if not 0 <= start < len(order) or not 1 <= count <= len(order)-start:
        raise ValueError('Task exceeds its disjoint epoch shard')
    return order[start:start+count]


def global_indices(seed, train_limit, epoch, start, count):
    if any(type(x) is not int for x in (seed,train_limit,epoch,start,count)) or not 1 <= train_limit <= 50000 or not 0 <= epoch <= 10000:
        raise ValueError('Invalid global partition')
    if not 0 <= start <= train_limit or not 0 <= count <= train_limit-start:
        raise ValueError('Task exceeds global epoch window')
    chosen = np.random.default_rng(seed).permutation(50000)[:train_limit]
    order = np.random.default_rng(seed + 1009*epoch).permutation(chosen)
    return order[start:start+count]


def batch(images, labels, selected, dtype=torch.float32):
    x = torch.from_numpy(np.array(images[selected],copy=True)).to(dtype).div_(255).sub_(.5).div_(.5)
    y = torch.from_numpy(np.array(labels[selected],dtype=np.int64,copy=True))
    return x,y


def train(payload, directory):
    before = time.perf_counter(); cpu_before = time.process_time()
    manifest = load_manifest(directory)
    if payload['dataset_id'] != manifest['dataset_id']:
        raise ValueError('CIFAR dataset identity mismatch')
    if not 1 <= payload['batch_size'] <= 512 or not 0 < payload['lr'] <= 1 or not 0 <= payload['momentum'] < 1:
        raise ValueError('Invalid optimizer configuration')
    numerics = payload.get('numerics', 'fp32-native')
    net = model(payload['seed'],payload.get('threads',1),numerics); set_weights(net,payload['weights'])
    dtype = next(net.parameters()).dtype
    optimizer = torch.optim.SGD(net.parameters(),lr=payload['lr'],momentum=payload['momentum'])
    if payload.get('optimizer_momentum'):
        flat = decode(payload['optimizer_momentum']); offset=0
        for p in net.parameters():
            count=p.numel()
            optimizer.state[p]['momentum_buffer']=torch.from_numpy(flat[offset:offset+count].reshape(p.shape).copy()).to(p.dtype)
            offset+=count
    if payload.get('indexing_version',1)==2:
        validate_plan(payload['plan'],payload,hashlib.sha256(decode(payload['weights']).tobytes()).hexdigest())
        selected = global_indices(payload['seed'],payload['train_limit'],payload['epoch'],payload['start'],payload['count'])
    elif payload.get('indexing_version',1)==1:
        selected = indices(payload['seed'],payload['train_limit'],payload['peers'],payload['peer'],
                           payload['epoch'],payload['start'],payload['count'])
    else:
        raise ValueError('Unsupported indexing version')
    images,labels=arrays(directory,'train')
    loaded=time.perf_counter(); total_loss=0.; correct=0
    net.train()
    for offset in range(0,len(selected),payload['batch_size']):
        x,y=batch(images,labels,selected[offset:offset+payload['batch_size']],dtype)
        optimizer.zero_grad(set_to_none=True)
        out=net(x); value=F.cross_entropy(out,y)
        value.backward(); optimizer.step()
        total_loss+=float(value.detach())*len(y)
        correct+=int((out.detach().argmax(1)==y).sum())
    trained=time.perf_counter()
    buffers = torch.cat([optimizer.state[p].get('momentum_buffer',torch.zeros_like(p)).flatten() for p in net.parameters()]).numpy()
    return dict(weights=weights(net),optimizer_momentum=encode(buffers),count=len(selected),
        train_loss=total_loss/len(selected),train_accuracy=correct/len(selected),
        prepare_seconds=loaded-before,train_seconds=trained-loaded,wall_seconds=trained-before,
        cpu_seconds=time.process_time()-cpu_before,peak_rss_mb=peak_rss_mb(),
        torch_version=torch.__version__,threads=payload.get('threads',1),numerics=numerics,
        compute_dtype=str(dtype),mkldnn_enabled=torch.backends.mkldnn.enabled,
        cpu_capability=torch.backends.cpu.get_cpu_capability())


def evaluate(encoded,directory,seed=42,threads=1,test_limit=10000,numerics='fp32-native'):
    started=time.perf_counter(); net=model(seed,threads,numerics);set_weights(net,encoded);net.eval()
    dtype=next(net.parameters()).dtype
    images,labels=arrays(directory,'test')
    if not 1 <= test_limit <= 10000:
        raise ValueError('Invalid test-set size')
    chosen=np.random.default_rng(seed).permutation(10000)[:test_limit]
    confusion=np.zeros((10,10),dtype=np.int64);total_loss=0.;top5=0
    with torch.no_grad():
        for start in range(0,len(chosen),256):
            x,y=batch(images,labels,chosen[start:start+256],dtype);out=net(x)
            total_loss+=float(F.cross_entropy(out,y,reduction='sum'))
            pred=out.argmax(1).numpy();truth=y.numpy()
            np.add.at(confusion,(truth,pred),1)
            top5+=int((out.topk(5,dim=1).indices==y[:,None]).any(1).sum())
    tp=confusion.diagonal();precision=tp/np.maximum(confusion.sum(0),1);recall=tp/np.maximum(confusion.sum(1),1)
    f1=2*precision*recall/np.maximum(precision+recall,1e-12)
    return dict(test_loss=total_loss/test_limit,test_accuracy=float(tp.sum()/test_limit),
        test_top5_accuracy=top5/test_limit,macro_f1=float(f1.mean()),precision=precision.tolist(),
        recall=recall.tolist(),f1=f1.tolist(),confusion_matrix=confusion.tolist(),test_count=test_limit,
        evaluation_seconds=time.perf_counter()-started)

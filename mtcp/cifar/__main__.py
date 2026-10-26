import argparse
import json
import platform


def main():
    parser=argparse.ArgumentParser(description='Equal-peer MTCP CIFAR-10 experiments')
    commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare',help='Convert checksum-verified CIFAR to safe arrays')
    source=prepare.add_mutually_exclusive_group(required=True)
    source.add_argument('--archive');source.add_argument('--download',action='store_true')
    prepare.add_argument('--output',default='.mtcp/cifar10')
    peer=commands.add_parser('peer',help='Advertise, discover, and train as an equal peer')
    peer.add_argument('--data',default='.mtcp/cifar10');peer.add_argument('--host',default='0.0.0.0')
    peer.add_argument('--port',type=int,default=9101);peer.add_argument('--advertise-host')
    peer.add_argument('--name',default=platform.node());peer.add_argument('--state-dir')
    peer.add_argument('--cpus');peer.add_argument('--threads',type=int,default=1)
    peer.add_argument('--connect',action='append',default=[]);peer.add_argument('--no-discovery',action='store_true')
    peer.add_argument('--discovery-port',type=int,default=47767)
    peer.add_argument('--backend',choices=['process','firecracker'],default='process')
    peer.add_argument('--assets',default='.mtcp/firecracker-cifar');peer.add_argument('--memory-mb',type=int,default=768)
    run=commands.add_parser('run',help='Propose an experiment from either peer; all peers train and aggregate')
    run.add_argument('--endpoint',default='http://127.0.0.1:9101');run.add_argument('--peers',type=int,default=2)
    run.add_argument('--epochs',type=int,default=10);run.add_argument('--train-limit',type=int,default=50000)
    run.add_argument('--test-limit',type=int,default=10000);run.add_argument('--batch-size',type=int,default=32)
    run.add_argument('--sync-samples',type=int,default=10000);run.add_argument('--threads',type=int,default=1)
    run.add_argument('--seed',type=int,default=42);run.add_argument('--lr',type=float,default=.001)
    run.add_argument('--initial-checkpoint',help='Use the exact initial.json artifact from the control run for a cross-platform comparison')
    run.add_argument('--momentum',type=float,default=.9);run.add_argument('--audit',choices=['full','none'],default='full')
    run.add_argument('--discovery-wait',type=float,default=60);run.add_argument('--no-wait',action='store_true')
    run.add_argument('--output',default='results/cifar')
    report=commands.add_parser('compare',help='Compare compatible experiment artifacts')
    report.add_argument('runs',nargs='+');report.add_argument('--output',default='results/cifar/comparison')
    collection=commands.add_parser('collect',help='Collect an existing asynchronous run from the peers')
    collection.add_argument('--endpoint',default='http://127.0.0.1:9101')
    collection.add_argument('--run-id',required=True);collection.add_argument('--output',default='results/cifar')
    args=parser.parse_args()
    if args.command=='prepare':
        from .data import prepare as convert
        if args.download:
            from pathlib import Path
            import urllib.request
            archive=Path('.mtcp/downloads/cifar-10-python.tar.gz')
            archive.parent.mkdir(parents=True,exist_ok=True)
            if not archive.exists():
                temporary=archive.with_suffix('.part')
                print('Downloading the official CIFAR-10 archive; checksum verification follows',flush=True)
                urllib.request.urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',temporary)
                temporary.replace(archive)
            args.archive=str(archive)
        print(json.dumps(convert(args.archive,args.output),indent=2))
    elif args.command=='peer':
        from .peer import serve
        serve(args)
    elif args.command=='run':
        from .experiment import start
        start(args)
    elif args.command=='compare':
        from .report import compare
        print(compare(args.runs,args.output))
    elif args.command=='collect':
        import os
        from .network import call
        from .experiment import collect
        key=os.environ.get('MTCP_CLUSTER_KEY','')
        if len(key)<32:raise ValueError('Set MTCP_CLUSTER_KEY to the experiment key')
        value,_=call(args.endpoint,f'/v1/runs/{args.run_id}',key)
        collect(value['config'],key,args.output)


if __name__=='__main__':main()

import argparse
import json


def main():
    parser=argparse.ArgumentParser(description='Reproducible MTCP CIFAR-10 experiments')
    commands=parser.add_subparsers(dest='command',required=True)
    prepare=commands.add_parser('prepare',help='Convert the verified official archive to safe memory-mapped arrays')
    prepare.add_argument('--archive',required=True)
    prepare.add_argument('--output',default='.mtcp/cifar10')
    args=parser.parse_args()
    if args.command=='prepare':
        from .data import prepare as convert
        print(json.dumps(convert(args.archive,args.output),indent=2))


if __name__=='__main__':
    main()

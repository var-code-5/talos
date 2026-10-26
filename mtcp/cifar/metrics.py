"""Run records use explicit units and separate training, evaluation and network costs."""
import csv
import json
import math
from pathlib import Path
import statistics


def distribution(values):
    values=sorted(values)
    if not values:return {'count':0,'mean':None,'median':None,'p95':None,'max':None}
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
                p95=values[max(0,math.ceil(.95*len(values))-1)],max=values[-1])


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def export_rounds(path,rounds):
    if not rounds:return
    fields=[key for key,value in rounds[0].items() if not isinstance(value,(dict,list))]
    with open(path,'w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rounds)

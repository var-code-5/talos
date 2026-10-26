"""Portable host observations; Linux-only isolation is kept out of peer imports."""
import hashlib
import os
from pathlib import Path
import platform
import uuid


def _psutil():
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def available_cpus():
    if hasattr(os,'sched_getaffinity'):
        return sorted(os.sched_getaffinity(0))
    psutil=_psutil()
    if psutil and hasattr(psutil.Process(),'cpu_affinity'):
        return psutil.Process().cpu_affinity()
    return list(range(os.cpu_count() or 1))


def pin_cpus(cpus):
    if hasattr(os,'sched_setaffinity'):
        os.sched_setaffinity(0,cpus);return True
    psutil=_psutil()
    if psutil and hasattr(psutil.Process(),'cpu_affinity'):
        psutil.Process().cpu_affinity(cpus);return True
    # macOS does not expose the same hard affinity API; torch still limits threads.
    return False


def memory_mb():
    psutil=_psutil()
    if psutil:
        value=psutil.virtual_memory();return value.total//1024**2,value.available//1024**2
    if Path('/proc/meminfo').exists():
        values=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
        return int(values['MemTotal'].split()[0])//1024,int(values['MemAvailable'].split()[0])//1024
    raise RuntimeError('Install psutil from requirements-cifar.txt for host metrics on this platform')


def identity():
    source=Path('/etc/machine-id')
    raw=source.read_text().strip() if source.exists() else f'{platform.node()}:{uuid.getnode()}'
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def peak_rss_mb():
    try:
        import resource
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value/(1024**2 if platform.system()=='Darwin' else 1024)
    except ImportError:
        psutil=_psutil()
        if not psutil:raise RuntimeError('Install psutil for Windows process metrics')
        info=psutil.Process().memory_info()
        return getattr(info,'peak_wset',info.rss)/1024**2

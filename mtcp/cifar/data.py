"""Convert the checksum-verified official CIFAR archive to non-pickle arrays."""
import hashlib
import io
import json
from pathlib import Path
import pickle
import tarfile

import numpy as np

ARCHIVE_MD5 = 'c58f30108f718f92721af3b95e74349a'
FILES = ('train_images.npy', 'train_labels.npy', 'test_images.npy', 'test_labels.npy')
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':')).encode()).hexdigest()


class CIFARUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray','_reconstruct'), ('numpy._core.multiarray','_reconstruct'),
                   ('numpy','ndarray'), ('numpy','dtype')}
        if (module,name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported CIFAR archive global: {module}.{name}')
        return super().find_class(module, name)


def prepare(archive, destination):
    archive, destination = Path(archive), Path(destination)
    with archive.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'md5').hexdigest()
    if checksum != ARCHIVE_MD5:
        raise ValueError('Source does not match the published official CIFAR-10 Python archive checksum')
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        for split, count, batches in [('train',50000,[f'data_batch_{i}' for i in range(1,6)]),
                                       ('test',10000,['test_batch'])]:
            images = np.lib.format.open_memmap(destination/f'{split}_images.npy', mode='w+', dtype=np.uint8, shape=(count,3,32,32))
            labels = np.lib.format.open_memmap(destination/f'{split}_labels.npy', mode='w+', dtype=np.int64, shape=(count,))
            for i,name in enumerate(batches):
                source = bundle.extractfile(f'cifar-10-batches-py/{name}')
                item = CIFARUnpickler(source, encoding='bytes').load()
                x, y = item[b'data'], np.asarray(item[b'labels'], dtype=np.int64)
                if x.shape != (10000,3072) or x.dtype != np.uint8 or y.shape != (10000,) or not np.all((y>=0)&(y<10)):
                    raise ValueError('Invalid CIFAR batch structure')
                images[i*10000:(i+1)*10000] = x.reshape(10000,3,32,32)
                labels[i*10000:(i+1)*10000] = y
            images.flush(); labels.flush()
    content = {'files':{name:sha(destination/name) for name in FILES}, 'classes':CLASSES,
               'train_count':50000,'test_count':10000,'normalization':'float32 / 255, then (x - 0.5) / 0.5'}
    manifest = dict(content, dataset_id=identity(content), source_md5=checksum)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def load_manifest(directory, verify=False):
    root = Path(directory)
    manifest = json.loads((root/'manifest.json').read_text())
    content = {k:manifest[k] for k in ('files','classes','train_count','test_count','normalization')}
    if identity(content) != manifest['dataset_id'] or set(manifest['files']) != set(FILES):
        raise ValueError('Invalid CIFAR manifest')
    if verify:
        for name, checksum in manifest['files'].items():
            if sha(root/name) != checksum:
                raise ValueError(f'CIFAR dataset checksum mismatch: {name}')
    return manifest


def arrays(directory, split):
    root = Path(directory)
    images = np.load(root/f'{split}_images.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(root/f'{split}_labels.npy', mmap_mode='r', allow_pickle=False)
    count = 50000 if split == 'train' else 10000
    if images.dtype != np.uint8 or images.shape != (count,3,32,32) or labels.dtype != np.int64 or labels.shape != (count,):
        raise ValueError('Invalid prepared dataset shape/dtype')
    return images, labels

"""Authenticated peer HTTP and LAN multicast discovery. No permanent coordinator."""
import json
import secrets
import socket
import struct
import threading
import time
import urllib.request
import ssl
import os

from mtcp.kernel import canonical
from mtcp.protocol import sign, authenticated
from mtcp.transport import NoRedirect

GROUP='239.255.77.67'
PORT=47767
# Up to 500 rounds with eight peers' measured costs and the initial checkpoint.
MAX_RESPONSE_BYTES=8*1024*1024


def call(endpoint, route, key, body=None, timeout=10):
    data=canonical(body) if body is not None else None
    nonce=secrets.token_hex(16)
    headers={'Content-Type':'application/json','Authorization':f'Bearer {key}','X-MTCP-Nonce':nonce}
    request=urllib.request.Request(endpoint.rstrip('/')+route,data=data,headers=headers)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect(),
        urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=os.environ.get('MTCP_CA_FILE'))))
    start=time.perf_counter()
    with opener.open(request,timeout=timeout) as response:
        raw=response.read(MAX_RESPONSE_BYTES+1)
    elapsed=time.perf_counter()-start
    if len(raw)>MAX_RESPONSE_BYTES:raise ValueError('Oversized peer response')
    envelope=json.loads(raw)
    value=envelope['value']
    if envelope.get('nonce')!=nonce or not authenticated({'nonce':nonce,'value':value},envelope.get('mac'),key):
        raise ValueError('Peer response authentication failed')
    return value,dict(seconds=elapsed,request_body_bytes=len(data or b''),response_body_bytes=len(raw))


def discover_host():
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
        try:
            sock.connect(('192.0.2.1',9))
            return sock.getsockname()[0]
        except OSError:return '127.0.0.1'


class Discovery:
    def __init__(self,description,remember,key,stop,port=PORT):
        self.description,self.remember,self.key,self.stop=description,remember,key,stop
        self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM,socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        self.sock.bind(('',port))
        self.sock.setsockopt(socket.IPPROTO_IP,socket.IP_ADD_MEMBERSHIP,struct.pack('4s4s',socket.inet_aton(GROUP),socket.inet_aton('0.0.0.0')))
        self.sock.setsockopt(socket.IPPROTO_IP,socket.IP_MULTICAST_TTL,1)
        self.sock.setsockopt(socket.IPPROTO_IP,socket.IP_MULTICAST_LOOP,1)
        self.sock.settimeout(.5)
        self.port=port

    def start(self):
        threading.Thread(target=self.listen,daemon=True).start()
        threading.Thread(target=self.announce,daemon=True).start()

    def announce(self):
        while not self.stop.is_set():
            value=dict(peer=self.description(),timestamp=time.time())
            raw=canonical({'value':value,'mac':sign(value,self.key)})
            try:self.sock.sendto(raw,(GROUP,self.port))
            except OSError:pass
            self.stop.wait(2)

    def listen(self):
        while not self.stop.is_set():
            try:
                raw,_=self.sock.recvfrom(16385)
                if len(raw)>16384:continue
                envelope=json.loads(raw);value=envelope['value']
                if authenticated(value,envelope.get('mac'),self.key) and abs(time.time()-value['timestamp'])<10:
                    self.remember(value['peer'])
            except (OSError,ValueError,KeyError,TypeError):pass

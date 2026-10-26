"""HTTP transport with no redirects/proxies and normal TLS certificate verification."""
import json
import os
import ssl
import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def request(url, route, token=None, body=None, timeout=6):
    context = ssl.create_default_context(cafile=os.environ.get('MTCP_CA_FILE'))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
                                        urllib.request.HTTPSHandler(context=context))
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = urllib.request.Request(url.rstrip('/') + route,
        data=json.dumps(body).encode() if body is not None else None, headers=headers)
    with opener.open(req, timeout=timeout) as response:
        raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError('Response too large')
        return json.loads(raw)

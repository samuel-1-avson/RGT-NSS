import urllib.request, json, urllib.error
req = urllib.request.Request(
    'http://localhost:8000/api/compute/attention', 
    data=json.dumps({'text': 'cat', 'd_model': 64, 'num_heads': 4, 'num_layers': 1, 'show_causal_mask': True}).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
try:
    res = urllib.request.urlopen(req)
    print("Success:")
    print(res.read().decode('utf-8')[:500])
except urllib.error.HTTPError as e:
    print("HTTP Error", e.code)
    print(e.read().decode('utf-8'))

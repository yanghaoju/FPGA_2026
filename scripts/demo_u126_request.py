#!/usr/bin/env python3
"""Display one real model request; all live metrics are full server metrics."""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description='U126 competition live inference demo')
    parser.add_argument('--port', type=int, default=8083)
    parser.add_argument('--prompt', default='Question: What is the capital of France?\nAnswer:')
    parser.add_argument('--output-tokens', type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.output_tokens <= 128:
        parser.error('port must be 1..65535; output-tokens must be 1..128')
    base = f'http://127.0.0.1:{args.port}'

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(base+path, data, {'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)

    print('=== U126 / KV260 Attention Accelerator ===', flush=True)
    health = request('/health')
    if health.get('status') != 'ok':
        raise RuntimeError(f'Server not ready: {health}')
    print('[1/4] Server ready. Run smoke_u126_server.sh separately to verify strict PL identity.', flush=True)
    tokens = request('/tokenize', {'content':args.prompt, 'add_special':True})['tokens']
    if len(tokens)+args.output_tokens > 504:
        raise ValueError('Prompt + output budget must be <= 504 for this demo; shorten the request.')
    print(f'[2/4] Prompt: {len(tokens)} tokens; output budget: {args.output_tokens}; KV capacity: 512', flush=True)
    print('Input:\n'+args.prompt, flush=True)
    print('[3/4] Running real Prefill and Decode. Please wait...', flush=True)
    started = time.monotonic()
    result = request('/completion', {'prompt':tokens, 'n_predict':args.output_tokens,
                                    'temperature':0, 'seed':42, 'cache_prompt':False,
                                    'stop':['\n\nQuestion:', '\nQuestion:']})
    elapsed = time.monotonic()-started
    if result.get('truncated'):
        raise RuntimeError('Unexpected truncation; keep this response for diagnosis: '+json.dumps(result))
    if result.get('tokens_predicted',0) <= 0 or result.get('tokens_evaluated') != len(tokens):
        raise RuntimeError('Incomplete inference response: '+json.dumps(result))
    print('\nOutput:\n'+result.get('content',''), flush=True)
    timings = result.get('timings',{})
    print('\n[4/4] Real request completed', flush=True)
    print(f'Prompt evaluated: {result["tokens_evaluated"]}; output generated: {result["tokens_predicted"]}')
    print(f'Wall time: {elapsed:.3f} s; truncated: {result.get("truncated",False)}')
    for label,key,unit in [('Model Prefill','prompt_ms','ms'),('Model Prefill rate','prompt_per_second','tokens/s'),
                           ('Model Decode','predicted_ms','ms'),('Model Decode rate','predicted_per_second','tokens/s')]:
        value=timings.get(key)
        if isinstance(value,(int,float)):
            print(f'{label}: {value:.3f} {unit}')
    print('Live timings are full server/model metrics, not Attention-only benchmark results.')
    print('After stopping the server normally, inspect its PL report: calls > 0, fallback = 0, failure = 0.')


if __name__ == '__main__':
    try:
        main()
    except (OSError,ValueError,RuntimeError,urllib.error.URLError) as exc:
        print('DEMO FAILED: '+str(exc),file=sys.stderr)
        sys.exit(1)

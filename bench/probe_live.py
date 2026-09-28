#!/usr/bin/env python3
"""Live-engine measurement: cold greedy decode + repeat (cache hit) at a target depth."""
import argparse, json, sys, time
sys.path.insert(0, '/home/ai/bench')
import ninfer_bench as nb

ap = argparse.ArgumentParser()
ap.add_argument('--label', required=True)
ap.add_argument('--long-tokens', type=int, default=230000)
ap.add_argument('--reply-tokens', type=int, default=256)
ap.add_argument('--out', required=True)
ap.add_argument('--salt', default=None)
ap.add_argument('--repeat', action='store_true')
a = ap.parse_args()

salt = a.salt or ('probe-%s-%d' % (a.label, int(time.time())))
prompt = nb.make_prompt(a.long_tokens)
rep = {'label': a.label, 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
       'long_tokens': a.long_tokens, 'chars': len(prompt), 'salt': salt, 'results': []}
for lab in (['long-cold'] + (['long-warm-repeat'] if a.repeat else [])):
    t0 = time.time()
    r = nb.one_measure(lab, prompt, a.reply_tokens, salt + '-long')
    r['wall_s_outer'] = round(time.time() - t0, 1)
    rep['results'].append(r)
    print(json.dumps(r, ensure_ascii=False), flush=True)
rep['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
open(a.out, 'w').write(json.dumps(rep, ensure_ascii=False, indent=2))

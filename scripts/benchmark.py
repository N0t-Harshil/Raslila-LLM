"""Standard benchmarks for scaling-study runs (Raslila v2 and the baseline).

    python scripts/benchmark.py --run raslila_125m          # one run
    python scripts/benchmark.py --all                        # every run with a checkpoint
    python scripts/benchmark.py --run raslila_125m --quick   # 300 examples per task (minutes)
    python scripts/benchmark.py --run raslila_125m --think 4 # allow more thinking cycles than trained
    python scripts/benchmark.py --run baseline_1b --run-code # also EXECUTE HumanEval completions

Zero-shot, scored the way lm-evaluation-harness does it (log-likelihood of each
answer choice), so the numbers are comparable to published small-model results:

  general knowledge / reasoning   hellaswag, arc_easy, arc_challenge, sciq,
                                  openbookqa, winogrande, lambada
  maths                           gsm8k_bpb (how well it predicts worked solutions)
                                  gsm8k (3-shot, greedy, exact final answer)
  code                            humaneval_bpb (predicting reference solutions)
                                  humaneval pass@1 (only with --run-code: it runs
                                  model-written Python in a subprocess)
  language modelling              wikitext_bpb (bits per byte, lower is better)

Raslila is scored after EVERY thinking cycle in the same forward pass, so the
output shows whether the second loop actually improves the answers.

HONEST EXPECTATIONS. Models under 1B trained on <= 20B tokens score near chance
on arc_challenge / winogrande / openbookqa and ~0-3% on gsm8k and humaneval;
that is normal (GPT-2 1.5B: hellaswag ~50%, gsm8k ~0%). The comparison that
matters is Raslila vs the baseline AT THE SAME SIZE, and how the gap moves with
size. The bpb metrics are the most sensitive at small scale.

Writes runs/<run>/benchmark.json (re-used if the checkpoint has not changed).
Datasets are downloaded once from the Hugging Face Hub (parquet, a few MB each).
"""
import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
os.chdir(_ROOT)

import torch
import torch.nn.functional as F

BENCH_VERSION = 2
PAD, EOS = 0, 1

# task -> (primary metric, higher_is_better, chance level)
TASKS = {
    'hellaswag':     ('acc_norm', True, 0.25),
    'arc_easy':      ('acc', True, 0.25),
    'arc_challenge': ('acc_norm', True, 0.25),
    'sciq':          ('acc', True, 0.25),
    'openbookqa':    ('acc_norm', True, 0.25),
    'winogrande':    ('acc', True, 0.50),
    'lambada':       ('acc', True, 0.0),
    'wikitext_bpb':  ('bpb', False, None),
    'gsm8k_bpb':     ('bpb', False, None),
    'humaneval_bpb': ('bpb', False, None),
    'gsm8k':         ('exact_match', True, 0.0),
    'var_chain':     ('acc', True, 0.10),
    'add2':          ('acc', True, 0.25),
    'humaneval':     ('pass@1', True, 0.0),
}
DEFAULT_TASKS = [t for t in TASKS if t != 'humaneval']
GROUPS = {'general': ['hellaswag', 'arc_easy', 'arc_challenge', 'sciq', 'openbookqa', 'winogrande', 'lambada'],
          'reasoning': ['var_chain', 'add2'],
          'math': ['gsm8k_bpb', 'gsm8k'], 'code': ['humaneval_bpb', 'humaneval'], 'lm': ['wikitext_bpb']}


# ---------------------------------------------------------------- data -------
def hub_rows(repo, path):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    local = hf_hub_download(repo_id=repo, filename=path, repo_type='dataset')
    return pq.read_table(local).to_pylist()


def _hs_clean(text):
    text = text.strip().replace(' [title]', '. ')
    text = re.sub(r'\[.*?\]', '', text)
    return text.replace('  ', ' ')


def mc_docs(task):
    """Multiple choice -> list of (context, [continuations], gold) (lm-eval formats)."""
    docs = []
    if task == 'hellaswag':
        for r in hub_rows('Rowan/hellaswag', 'data/validation-00000-of-00001.parquet'):
            ctx = r['ctx_a'] + ' ' + r['ctx_b'].capitalize()
            docs.append((_hs_clean(r['activity_label'] + ': ' + ctx),
                         [' ' + _hs_clean(e) for e in r['endings']], int(r['label'])))
    elif task in ('arc_easy', 'arc_challenge'):
        sub = 'ARC-Easy' if task == 'arc_easy' else 'ARC-Challenge'
        for r in hub_rows('allenai/ai2_arc', sub + '/test-00000-of-00001.parquet'):
            labels = list(r['choices']['label'])
            if r['answerKey'] not in labels:
                continue
            docs.append(('Question: ' + r['question'] + '\nAnswer:',
                         [' ' + t for t in r['choices']['text']], labels.index(r['answerKey'])))
    elif task == 'sciq':
        for r in hub_rows('allenai/sciq', 'data/test-00000-of-00001.parquet'):
            ch = [r['distractor1'], r['distractor2'], r['distractor3'], r['correct_answer']]
            docs.append(('{}\nQuestion: {}\nAnswer:'.format(r['support'].strip(), r['question']).lstrip(),
                         [' ' + c for c in ch], 3))
    elif task == 'openbookqa':
        for r in hub_rows('allenai/openbookqa', 'main/test-00000-of-00001.parquet'):
            labels = list(r['choices']['label'])
            docs.append((r['question_stem'], [' ' + t for t in r['choices']['text']],
                         labels.index(r['answerKey'].strip())))
    else:
        raise KeyError(task)
    return docs


# ------------------------------------------------------------- scoring -------
class Scorer:
    def __init__(self, model, tok, device, think, max_len, batch_tokens):
        self.model, self.tok, self.device = model, tok, device
        self.looped = bool(getattr(model, 'looped', False))
        self.adaptive = model.family == 'raslila' and not self.looped and model.config.max_think_cycles > 0
        self.think = think if (self.adaptive or self.looped) else None
        self.max_len, self.batch_tokens = max_len, batch_tokens
        self.n_exits = len(think) if self.looped else (think if self.adaptive else 1)

    def enc(self, text):
        return self.tok.encode(text).ids

    @torch.no_grad()
    def forward(self, x, gen=False):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            if self.looped and gen:                  # generation: the model's default thinking time
                out = self.model(x)
            elif self.looped:
                out = self.model(x, exit_at=self.think)
            elif self.adaptive:
                out = self.model(x, think_cycles=self.think, halt=False)
            else:
                out = self.model(x)
        return out['exit_logits']

    def loglik(self, reqs, desc=''):
        """reqs: list of (ctx_ids, cont_ids). Returns per request, per exit:
        (sum log p(cont | ctx), all continuation tokens greedy?)."""
        items = []
        for i, (c, k) in enumerate(reqs):
            full = ([EOS] + c + k)[-(self.max_len + 1):]
            n_cont = min(len(k), len(full) - 1)
            items.append((i, full, n_cont))
        items.sort(key=lambda t: -len(t[1]))
        res = [None] * len(reqs)
        t0, done = time.time(), 0
        j = 0
        while j < len(items):
            L = len(items[j][1]) - 1
            b = max(1, self.batch_tokens // max(1, L))
            chunk = items[j:j + b]
            j += b
            x = torch.zeros(len(chunk), L, dtype=torch.long)
            for r, (_, full, _) in enumerate(chunk):
                x[r, :len(full) - 1] = torch.tensor(full[:-1])
            exits = self.forward(x.to(self.device))
            for r, (i, full, n_cont) in enumerate(chunk):
                n = len(full) - 1
                if n_cont <= 0:
                    res[i] = [(0.0, True)] * len(exits)
                    continue
                tgt = torch.tensor(full[-n_cont:], device=self.device)
                per = []
                for ex in exits:
                    lg = ex[r, n - n_cont:n].float()
                    lp = lg.log_softmax(-1)
                    per.append((float(lp.gather(1, tgt[:, None]).sum()),
                                bool((lg.argmax(-1) == tgt).all())))
                res[i] = per
            done += len(chunk)
            if time.time() - t0 > 30:
                print('    {} {}/{}'.format(desc, done, len(items)), flush=True)
                t0 = time.time()
        return res

    @torch.no_grad()
    def generate(self, prompts, max_new, stops, desc=''):
        """Greedy, batched. Right padding is safe: attention is causal, so pad
        tokens after a row's end never influence that row."""
        outs = [''] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        bs = max(1, self.batch_tokens // (max(len(p) for p in prompts) + max_new))
        t0 = time.time()
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            seqs = [list(prompts[i]) for i in idx]
            gen = [[] for _ in idx]
            live = list(range(len(idx)))
            for _ in range(max_new):
                rows = [seqs[r][-self.max_len:] for r in live]
                L = max(len(r) for r in rows)
                x = torch.zeros(len(rows), L, dtype=torch.long)
                for k, r in enumerate(rows):
                    x[k, :len(r)] = torch.tensor(r)
                lg = self.forward(x.to(self.device), gen=True)[-1]
                last = torch.tensor([len(r) - 1 for r in rows], device=self.device)
                nl = lg[torch.arange(len(rows), device=self.device), last].float()
                nl[:, PAD] = float('-inf')
                nxt = nl.argmax(-1).tolist()
                still = []
                for k, r in enumerate(live):
                    t = nxt[k]
                    if t == EOS:
                        continue
                    seqs[r].append(t)
                    gen[r].append(t)
                    text = self.tok.decode(gen[r])
                    if any(st in text for st in stops):
                        continue
                    still.append(r)
                live = still
                if not live:
                    break
            for k, i in enumerate(idx):
                text = self.tok.decode(gen[k])
                cut = min([text.find(st) for st in stops if st in text] or [len(text)])
                outs[i] = text[:cut]
            if time.time() - t0 > 30:
                print('    {} {}/{}'.format(desc, min(s + bs, len(order)), len(order)), flush=True)
                t0 = time.time()
        return outs


def per_exit(fn, n_exits):
    return [fn(e) for e in range(n_exits)]


def run_mc(sc, docs, desc):
    reqs, owner = [], []
    for d, (ctx, conts, gold) in enumerate(docs):
        c = sc.enc(ctx)
        for k in conts:
            reqs.append((c, sc.enc(k)))
            owner.append(d)
    ll = sc.loglik(reqs, desc)
    out = []
    for e in range(sc.n_exits):
        acc = accn = 0
        pos = 0
        for ctx, conts, gold in docs:
            s = [ll[pos + j][e][0] for j in range(len(conts))]
            sn = [v / max(1, len(conts[j].encode('utf-8'))) for j, v in enumerate(s)]
            acc += int(max(range(len(s)), key=s.__getitem__) == gold)
            accn += int(max(range(len(sn)), key=sn.__getitem__) == gold)
            pos += len(conts)
        out.append({'acc': acc / len(docs), 'acc_norm': accn / len(docs)})
    return out


def task_winogrande(sc, limit):
    rows = hub_rows('allenai/winogrande', 'winogrande_xl/validation-00000-of-00001.parquet')[:limit]
    reqs = []
    for r in rows:
        i = r['sentence'].index('_')
        cont = ' ' + r['sentence'][i + 1:].strip()
        for opt in (r['option1'], r['option2']):
            reqs.append((sc.enc(r['sentence'][:i] + opt), sc.enc(cont)))
    ll = sc.loglik(reqs, 'winogrande')
    out = []
    for e in range(sc.n_exits):
        acc = sum(int((0 if ll[2 * n][e][0] > ll[2 * n + 1][e][0] else 1) == int(r['answer']) - 1)
                  for n, r in enumerate(rows))
        out.append({'acc': acc / len(rows)})
    return out, len(rows)


def task_lambada(sc, limit):
    rows = hub_rows('EleutherAI/lambada_openai', 'en/test/en.parquet')[:limit]
    reqs = []
    for r in rows:
        ctx, word = r['text'].rsplit(' ', 1)
        reqs.append((sc.enc(ctx), sc.enc(' ' + word)))
    ll = sc.loglik(reqs, 'lambada')
    out = []
    for e in range(sc.n_exits):
        nll = -sum(v[e][0] for v in ll) / len(ll)
        out.append({'acc': sum(v[e][1] for v in ll) / len(ll), 'ppl': math.exp(min(50, nll))})
    return out, len(rows)


def bpb_of(sc, reqs, n_bytes, desc):
    ll = sc.loglik(reqs, desc)
    return [{'bpb': -sum(v[e][0] for v in ll) / math.log(2) / n_bytes} for e in range(sc.n_exits)]


def task_wikitext(sc, limit):
    rows = hub_rows('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet')
    text = ''.join(r['text'] for r in rows)
    ids = sc.enc(text)
    win = sc.max_len - 1
    chunks = [ids[i:i + win] for i in range(0, len(ids), win)]
    if limit:
        chunks = chunks[:max(1, limit // 10)]
        n_bytes = len(sc.tok.decode([t for c in chunks for t in c]).encode('utf-8'))
    else:
        n_bytes = len(text.encode('utf-8'))
    return bpb_of(sc, [([], c) for c in chunks], n_bytes, 'wikitext'), len(chunks)


def _gsm_clean(ans):
    return re.sub(r'<<.*?>>', '', ans)


def task_gsm8k_bpb(sc, limit):
    rows = hub_rows('openai/gsm8k', 'main/test-00000-of-00001.parquet')[:limit]
    reqs, nb = [], 0
    for r in rows:
        ans = ' ' + _gsm_clean(r['answer'])
        reqs.append((sc.enc('Question: ' + r['question'] + '\nAnswer:'), sc.enc(ans)))
        nb += len(ans.encode('utf-8'))
    return bpb_of(sc, reqs, nb, 'gsm8k_bpb'), len(rows)


def _last_number(s):
    nums = re.findall(r'-?\d[\d,]*\.?\d*', s)
    return nums[-1].replace(',', '').rstrip('.') if nums else None


def task_gsm8k(sc, limit):
    shots = hub_rows('openai/gsm8k', 'main/train-00000-of-00001.parquet')[:3]
    prefix = ''.join('Question: {}\nAnswer: {}\n\n'.format(s['question'], _gsm_clean(s['answer'])) for s in shots)
    rows = hub_rows('openai/gsm8k', 'main/test-00000-of-00001.parquet')[:limit]
    prompts = [sc.enc(prefix + 'Question: ' + r['question'] + '\nAnswer:') for r in rows]
    gens = sc.generate(prompts, 256, ['\n\n', 'Question:'], 'gsm8k')
    correct = 0
    for r, g in zip(rows, gens):
        gold = r['answer'].split('####')[-1].strip().replace(',', '')
        pred = g.split('####')[-1] if '####' in g else g
        correct += int(_last_number(pred) == gold)
    return [{'exact_match': correct / len(rows)}], len(rows), gens[:3]


def task_humaneval_bpb(sc, limit):
    rows = hub_rows('openai/openai_humaneval', 'openai_humaneval/test-00000-of-00001.parquet')[:limit]
    reqs = [(sc.enc(r['prompt']), sc.enc(r['canonical_solution'])) for r in rows]
    nb = sum(len(r['canonical_solution'].encode('utf-8')) for r in rows)
    return bpb_of(sc, reqs, nb, 'humaneval_bpb'), len(rows)


def task_humaneval(sc, limit):
    rows = hub_rows('openai/openai_humaneval', 'openai_humaneval/test-00000-of-00001.parquet')[:limit]
    gens = sc.generate([sc.enc(r['prompt']) for r in rows], 384,
                       ['\ndef ', '\nclass ', '\nif __name__', '\nprint(', '\n#', '\n```'], 'humaneval')
    passed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for n, (r, g) in enumerate(zip(rows, gens)):
            path = os.path.join(tmp, 'p{}.py'.format(n))
            with open(path, 'w', encoding='utf-8') as f:
                f.write(r['prompt'] + g + '\n\n' + r['test'] + '\n\ncheck({})\n'.format(r['entry_point']))
            try:
                rc = subprocess.run([sys.executable, '-I', path], cwd=tmp, capture_output=True,
                                    timeout=10).returncode
            except subprocess.TimeoutExpired:
                rc = 1
            passed += int(rc == 0)
    return [{'pass@1': passed / len(rows)}], len(rows), gens[:2]


# ------------------------------------------------- synthetic reasoning --------
# Small models score ~chance on arc/gsm8k, which hides reasoning differences.
# These procedurally generated, 5-shot tasks need multi-step use of the
# context (not memorised facts) and are where looped models were shown to
# beat even deeper standard models (Saunshi et al. 2025, "reasoning
# primitives"). Fixed seed: identical questions for every model.
def _var_chain_doc(rng, depth):
    import string
    names = rng.sample(string.ascii_lowercase, 6 + depth)
    vals = {}
    lines = []
    for n in names[:6]:
        vals[n] = rng.randrange(10)
        lines.append('{}={}'.format(n, vals[n]))
    prev = rng.choice(names[:6])
    for n in names[6:6 + depth]:
        lines.append('{}={}'.format(n, prev))
        vals[n] = vals[prev]
        prev = n
    rng.shuffle(lines)
    q = prev
    return ', '.join(lines) + ', {}=?'.format(q), vals[q]


def task_var_chain(sc, limit):
    import random as _r
    rng = _r.Random(20261109)
    shots = []
    for d in (0, 1, 2, 1, 2):
        text, ans = _var_chain_doc(rng, d)
        shots.append('{} {}'.format(text, ans))
    prefix = '\n'.join(shots) + '\n'
    n = limit or 600
    by_depth = {d: [] for d in range(4)}            # 0..3 hops of indirection
    for i in range(n):
        d = i % 4
        text, ans = _var_chain_doc(rng, d)
        by_depth[d].append((prefix + text, [' {}'.format(v) for v in range(10)], ans))
    parts = {d: run_mc(sc, docs, 'var_chain d{}'.format(d)) for d, docs in by_depth.items() if docs}
    per = []
    for e in range(sc.n_exits):
        row = {'acc_hops{}'.format(d): parts[d][e]['acc'] for d in parts}
        row['acc'] = sum(parts[d][e]['acc'] * len(by_depth[d]) for d in parts) / n
        row['acc_norm'] = row['acc']
        per.append(row)
    return per, n


def task_add2(sc, limit):
    import random as _r
    rng = _r.Random(20261110)

    def q():
        a, b = rng.randrange(10, 100), rng.randrange(10, 100)
        return a, b, a + b
    prefix = ''.join('{}+{}={}\n'.format(*q()) for _ in range(5))
    n = limit or 600
    docs = []
    for _ in range(n):
        a, b, s = q()
        wrong = list({s + 1, s - 1, s + 10, s - 10, s + 11} - {s})
        rng.shuffle(wrong)
        opts = [s] + wrong[:3]
        rng.shuffle(opts)
        docs.append((prefix + '{}+{}='.format(a, b), [str(o) for o in opts], opts.index(s)))
    return run_mc(sc, docs, 'add2'), n


# ---------------------------------------------------------------- main -------
def is_real_run(run_dir):
    """A study run trained on real data (not smoke / gpucheck / synthetic)."""
    import json as _json
    path = os.path.join(run_dir, 'metrics.json')
    if not os.path.exists(path):
        return False
    with open(path, 'r', encoding='utf-8') as f:
        return _json.load(f).get('data') != 'synthetic'


def load_run(run, device):
    from train_scaling import build_model
    path = os.path.join('runs', run, 'ckpt.pt')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(ck['model_config'])
    model.load_state_dict(ck['model'])
    del ck['opt']
    return model.to(device).eval(), ck


def bench_run(run, args, device, tok):
    out_path = os.path.join('runs', run, 'benchmark.json')
    model, ck = load_run(run, device)
    looped = bool(getattr(model, 'looped', False))
    adaptive = model.family == 'raslila' and not looped and model.config.max_think_cycles > 0
    think = (args.think or model.config.max_think_cycles) if adaptive else None
    if looped:   # loop counts to read out; the headline value is at test_loops
        think = sorted(set([int(v) for v in args.loops.split(',')] if args.loops else model.config.eval_loops)
                       | {model.config.test_loops})
    tasks = [t for t in (args.tasks.split(',') if args.tasks else DEFAULT_TASKS)]
    if args.run_code and 'humaneval' not in tasks:
        tasks.append('humaneval')
    if 'humaneval' in tasks and not args.run_code:
        tasks.remove('humaneval')
        print('  (humaneval pass@1 skipped: add --run-code to execute model-written code)')
    limit = 300 if args.quick else args.limit
    key = {'version': BENCH_VERSION, 'step': ck['step'], 'think': think, 'limit': limit,
           'gen_limit': args.gen_limit}
    prev = {}
    if os.path.exists(out_path) and not args.force:
        with open(out_path, 'r', encoding='utf-8') as f:
            prev = json.load(f)
        if prev.get('key') != key:
            prev = {}
    results = dict(prev.get('tasks', {}))
    todo = [t for t in tasks if t not in results]
    print('=' * 78)
    print('BENCHMARK {} | {} | step {} | {:.2f}B tokens{}'.format(
        run, model.family, ck['step'], ck['tokens_seen'] / 1e9,
        ' | thinking cycles 1..{} scored separately'.format(think) if adaptive else
        (' | read out after {} loops (headline: {})'.format(think, model.config.test_loops) if looped else '')))
    if not todo:
        print('  all requested tasks already scored for this checkpoint (use --force to redo)')
    sc = Scorer(model, tok, device, think, args.max_len, args.batch_tokens)
    for t in todo:
        t0 = time.time()
        samples = None
        try:
            if t in ('hellaswag', 'arc_easy', 'arc_challenge', 'sciq', 'openbookqa'):
                docs = mc_docs(t)[:limit]
                per, n = run_mc(sc, docs, t), len(docs)
            elif t == 'winogrande':
                per, n = task_winogrande(sc, limit)
            elif t == 'lambada':
                per, n = task_lambada(sc, limit)
            elif t == 'wikitext_bpb':
                per, n = task_wikitext(sc, limit)
            elif t == 'gsm8k_bpb':
                per, n = task_gsm8k_bpb(sc, limit)
            elif t == 'humaneval_bpb':
                per, n = task_humaneval_bpb(sc, limit)
            elif t == 'var_chain':
                per, n = task_var_chain(sc, limit)
            elif t == 'add2':
                per, n = task_add2(sc, limit)
            elif t == 'gsm8k':
                per, n, samples = task_gsm8k(sc, args.gen_limit)
            elif t == 'humaneval':
                per, n, samples = task_humaneval(sc, args.gen_limit)
            else:
                print('  unknown task {}, skipped'.format(t))
                continue
        except Exception as e:                       # one broken download must not lose the rest
            print('  {} FAILED: {}: {}'.format(t, type(e).__name__, str(e)[:300]))
            continue
        metric = TASKS[t][0]
        head = think.index(model.config.test_loops) if (looped and len(per) > 1) else -1
        results[t] = {'metric': metric, 'value': per[head][metric], 'n': n, 'all': per[head],
                      'per_cycle': [p[metric] for p in per] if len(per) > 1 else None,
                      'exit_loops': think if (looped and len(per) > 1) else None,
                      'seconds': time.time() - t0, 'samples': samples}
        cyc = ''
        if len(per) > 1:
            cyc = '   per cycle: ' + ' -> '.join('{:.4f}'.format(p[metric]) for p in per)
            if looped:
                cyc = '   by loops: ' + '  '.join('{}:{:.4f}'.format(r, p[metric]) for r, p in zip(think, per))
        print('  {:<14} {:<11} {:.4f}  (n={}, {:.0f}s){}'.format(t, metric, per[head][metric], n,
                                                              time.time() - t0, cyc), flush=True)
        with open(out_path, 'w', encoding='utf-8') as f:     # save after every task
            json.dump({'run': run, 'family': model.family, 'key': key, 'step': ck['step'],
                       'tokens_seen': ck['tokens_seen'], 'think': think, 'tasks': results,
                       'time': time.strftime('%Y-%m-%d %H:%M')}, f, indent=2)
    print('  -> {}'.format(out_path))
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', help='run folder in runs/ (repeatable)')
    ap.add_argument('--all', action='store_true', help='every run with a checkpoint')
    ap.add_argument('--tasks', default=None, help='comma list; default: all except humaneval')
    ap.add_argument('--limit', type=int, default=None, help='max examples per scoring task')
    ap.add_argument('--gen-limit', type=int, default=250, help='examples for gsm8k / humaneval generation')
    ap.add_argument('--quick', action='store_true', help='300 examples per task')
    ap.add_argument('--think', type=int, default=None, help='thinking cycles for Raslila v2/v3 (default: trained max)')
    ap.add_argument('--loops', default=None, help='Raslila v4: loop counts to score, e.g. 1,2,4,8,16,32')
    ap.add_argument('--run-code', action='store_true', help='execute HumanEval completions (pass@1)')
    ap.add_argument('--max-len', type=int, default=1024)
    ap.add_argument('--batch-tokens', type=int, default=16384)
    ap.add_argument('--force', action='store_true', help='re-score even if cached')
    ap.add_argument('--tokenizer', default=None, help='default: the one in configs/data.yaml')
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    runs = list(args.run or [])
    if args.all:
        runs += sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob('runs/*/ckpt.pt')
                       if is_real_run(os.path.dirname(p)))
    if not runs:
        sys.exit('Give --run NAME or --all.')
    from tokenizers import Tokenizer
    from raslila.data_mix import load_data_config
    tok = Tokenizer.from_file(args.tokenizer or load_data_config('configs/data.yaml')['tokenizer'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for run in dict.fromkeys(runs):
        if not os.path.exists(os.path.join('runs', run, 'ckpt.pt')):
            print('no checkpoint for {}, skipped'.format(run))
            continue
        bench_run(run, args, device, tok)
    return 0


if __name__ == '__main__':
    sys.exit(main())

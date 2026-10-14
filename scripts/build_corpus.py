"""Build a large real-text corpus from Project Gutenberg.


Downloads the PG catalog, picks the most-downloaded books, and fetches their
plain-text versions into data/corpus/. Re-running is incremental (skips files
that already exist).
"""

raise SystemExit('This legacy Gutenberg builder is superseded by scripts/get_dataset.py.')

import os
import re
import sys
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

OUT_DIR = 'data/corpus'
# Gutendex: fast, paginated JSON API (the big catalog CSV often times out).
API = 'https://gutendex.com/books/?page={}'
TEMPLATE = 'https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt'


def fetch(url, timeout=60, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Raslila/1.0'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def load_books(max_books):
    """Page the Gutendex API, keeping popular text books in download order."""
    books, seen = [], set()
    page = 1
    while len(books) < max_books and page <= 40:
        try:
            raw = fetch(API.format(page), timeout=90)
            data = json.loads(raw.decode('utf-8', errors='replace'))
        except Exception as e:
            print('  page {} failed: {}'.format(page, e), flush=True)
            break
        results = data.get('results', [])
        if not results:
            break
        for b in results:
            nid = b.get('id')
            if nid is None or nid in seen:
                continue
            # prefer plain text formats
            fmts = b.get('formats', {}) or {}
            mime = None
            for key in ('text/plain; charset=utf-8', 'text/plain; charset=us-ascii',
                        'text/plain', 'text/html; charset=utf-8'):
                if key in fmts:
                    mime = key
                    break
            if mime is None:
                continue
            title = (b.get('title') or '').strip()
            authors = ', '.join(a.get('name', '') for a in (b.get('authors') or []))
            dl = int(b.get('download_count') or 0)
            books.append({'id': nid, 'title': title, 'authors': authors,
                          'downloads': dl, 'mime': mime, 'url': fmts[mime]})
            seen.add(nid)
        print('  page {}: {} books so far'.format(page, len(books)), flush=True)
        page += 1
    books.sort(key=lambda b: -b['downloads'])
    return books[:max_books]


def clean_text(raw):
    """Strip PG license boilerplate head/tail, keep the actual book."""
    txt = raw.decode('utf-8', errors='replace')
    txt = txt.replace('\r\n', '\n').replace('\r', '\n')
    # PG header ends at the '*** START OF ...' marker
    m = re.search(r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\*\*\*', txt, re.I)
    if m:
        txt = txt[m.end():]
    m = re.search(r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG', txt, re.I)
    if m:
        txt = txt[:m.start()]
    return txt


def get_one(book):
    path = os.path.join(OUT_DIR, 'pg{}.txt'.format(book['id']))
    if os.path.exists(path) and os.path.getsize(path) > 2000:
        return (book['id'], 'skip', os.path.getsize(path))
    try:
        raw = fetch(book['url'], timeout=90, retries=2)
        txt = clean_text(raw)
        if len(txt) < 2000:
            return (book['id'], 'tiny', len(txt))
        with open(path, 'w', encoding='utf-8') as f:
            f.write(txt)
        return (book['id'], 'ok', len(txt))
    except Exception as e:
        return (book['id'], 'fail:{}'.format(type(e).__name__), 0)


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    os.makedirs(OUT_DIR, exist_ok=True)
    books = load_books(target)
    print('selected {} books'.format(len(books)), flush=True)

    ok = fail = 0
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        for i, (bid, status, n) in enumerate(ex.map(get_one, books)):
            if status == 'ok':
                ok += 1
                total_bytes += n
            elif status.startswith('fail'):
                fail += 1
            if (i + 1) % 50 == 0:
                print('  {}/{} done | ok={} fail={} | {:.1f} MB'.format(
                    i + 1, len(books), ok, fail, total_bytes / 1e6), flush=True)

    man = os.path.join(OUT_DIR, 'MANIFEST.txt')
    with open(man, 'w', encoding='utf-8') as f:
        f.write('id\tdownloads\ttitle\n')
        for b in sorted(books, key=lambda x: -x['downloads']):
            p = os.path.join(OUT_DIR, 'pg{}.txt'.format(b['id']))
            if os.path.exists(p):
                f.write('{}\t{}\t{}\n'.format(b['id'], b['downloads'], b['title']))
    print('DONE ok={} fail={} total={:.1f} MB'.format(ok, fail, total_bytes / 1e6), flush=True)


if __name__ == '__main__':
    main()

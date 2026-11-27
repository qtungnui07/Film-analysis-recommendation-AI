"""Local side-by-side demo of the same movie model before/after hard negatives."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from train_search_local import bm25_ranks, movie_text, weighted_rrf  # noqa: E402

PAGE = '''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>So sánh tìm kiếm phim</title><style>
*{box-sizing:border-box}body{font-family:Segoe UI,Arial,sans-serif;color:#21233d;background:#f4f5fa;margin:0}
main{max-width:1100px;margin:40px auto;padding:0 22px}h1{font-size:27px;margin:0 0 8px}
.subtitle,.hint{color:#60677f;font-size:14px;line-height:1.5}.search{display:flex;gap:10px;margin:24px 0}
input,button{font:inherit;border-radius:9px;padding:12px 15px}input{border:1px solid #cfd3e2;flex:1;min-width:0}
button{border:0;background:#5244d6;color:white;cursor:pointer;font-weight:600}button:disabled{opacity:.5}
.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}.panel{background:white;border:1px solid #e0e3ef;border-radius:14px;padding:20px}
h2{font-size:18px;margin:0 0 5px}.badge{font-size:12px;color:#716aa7}.results{padding:0;list-style:none}
.results li{border-top:1px solid #edf0f6;padding:12px 0;display:flex;gap:12px}.rank{color:#777e99;width:22px}
.title{font-weight:650}.meta{font-size:13px;color:#70778d;margin-top:4px}.overview{font-size:13px;line-height:1.4;color:#596077;margin-top:5px}
#status{min-height:22px;color:#60677f;font-size:14px}@media(max-width:700px){.columns{grid-template-columns:1fr}}
</style></head><body><main><h1>So sánh tìm kiếm phim</h1>
<div class="subtitle">Cùng dữ liệu và cùng cách tìm: MiniLM + BM25 80/20. Chỉ khác bước huấn luyện hard negatives.</div>
<form class="search" id="form"><input id="query" maxlength="200" placeholder="Nhập từ khóa, ví dụ: time travel" required><button id="submit">Tìm kiếm</button></form>
<div id="status" aria-live="polite"></div><div class="columns">
<section class="panel"><h2>Trước hard negatives</h2><div class="badge">MiniLM đã tinh chỉnh + BM25</div><ol class="results" id="before"></ol></section>
<section class="panel"><h2>Sau hard negatives</h2><div class="badge">Cùng model, train thêm hard negatives + BM25</div><ol class="results" id="after"></ol></section>
</div><p class="hint">Mỗi bên hiển thị 20 kết quả đầu trên cùng danh mục. Kết quả thay đổi chưa tự chứng minh chất lượng tốt hơn.</p>
<script>
const form=document.querySelector('#form'),input=document.querySelector('#query'),button=document.querySelector('#submit'),status=document.querySelector('#status');
function render(id,rows){const list=document.querySelector('#'+id);list.replaceChildren();rows.forEach((movie,i)=>{const li=document.createElement('li'),rank=document.createElement('span'),body=document.createElement('div'),title=document.createElement('div'),meta=document.createElement('div'),desc=document.createElement('div');rank.className='rank';rank.textContent=(i+1)+'.';title.className='title';title.textContent=movie.title+' ('+movie.year+')';meta.className='meta';meta.textContent=movie.genres||'';desc.className='overview';desc.textContent=movie.overview||'';body.append(title,meta,desc);li.append(rank,body);list.append(li)})}
form.addEventListener('submit',async e=>{e.preventDefault();const q=input.value.trim();if(!q)return;button.disabled=true;status.textContent='Đang tìm...';try{const response=await fetch('/api/search?q='+encodeURIComponent(q));if(!response.ok)throw Error('Không tìm được kết quả');const data=await response.json();render('before',data.before);render('after',data.after);status.textContent='Kết quả cho: '+q}catch(error){status.textContent=error.message}finally{button.disabled=false}});
</script></main></body></html>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', type=Path, default=ROOT / 'moviebot-db-old' / 'moviebot.db')
    parser.add_argument('--before', type=Path, default=ROOT / 'moviebot-db-old' / 'minilm-movie-search')
    parser.add_argument('--after', type=Path, default=ROOT / 'data' / 'old-original-hn-recheck')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if not args.db.is_file() or not args.before.is_dir() or not args.after.is_dir():
        raise SystemExit('Database or one of the model directories is missing')
    import torch
    from sentence_transformers import SentenceTransformer
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA not available; use --device cpu')
    with sqlite3.connect(f'{args.db.resolve().as_uri()}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        movies = [dict(row) for row in db.execute("SELECT movie_id,title,year,genres,overview FROM movies WHERE coalesce(overview,'') <> '' ORDER BY movie_id")]
    documents = [movie_text(m['title'], m['year'], m['genres'], m['overview']) for m in movies]
    models, vectors = {}, {}
    for name, path in [('before', args.before), ('after', args.after)]:
        print(f'Loading {name}: {path}', flush=True)
        models[name] = SentenceTransformer(str(path), device=args.device)
        vectors[name] = models[name].encode(documents, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/':
                body = PAGE.encode('utf-8'); content_type = 'text/html; charset=utf-8'
            elif parsed.path == '/api/search':
                query = parse_qs(parsed.query).get('q', [''])[0].strip()
                if not query or len(query) > 200:
                    self.send_error(400, 'Query must contain 1 to 200 characters'); return
                lexical = bm25_ranks(documents, [query])
                data = {}
                for name in ('before', 'after'):
                    vec = models[name].encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
                    ranking = np.argsort(-(vectors[name] @ vec))[None, :]
                    fused = weighted_rrf(ranking, lexical, 0.8, limit=20)[0]
                    data[name] = [movies[int(i)] for i in fused]
                body = json.dumps(data, ensure_ascii=False).encode('utf-8'); content_type = 'application/json; charset=utf-8'
            else:
                self.send_error(404); return
            self.send_response(200); self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    print(f'Ready: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()

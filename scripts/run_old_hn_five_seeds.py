"""Repeat old-data hard-negative fine-tuning with a fixed split and five trainer seeds."""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'data' / 'old-hn-five-seeds'
SEEDS = (42, 43, 44, 45, 46)


def main() -> int:
    RUNS.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in SEEDS:
        output = RUNS / f'seed-{seed}'
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f'Refusing to overwrite {output}')
    for seed in SEEDS:
        output = RUNS / f'seed-{seed}'
        log = RUNS / f'seed-{seed}.log'
        if log.exists():
            raise SystemExit(f'Refusing to overwrite {log}')
        command = [sys.executable, 'scripts/train_search_local.py', '--data-dir', 'moviebot-db-old',
                   '--hard-negatives', '--hard-negative-base-model', 'moviebot-db-old/minilm-movie-search',
                   '--previous-model', 'moviebot-db-old/minilm-movie-search', '--no-title-pairs',
                   '--output', str(output), '--device', 'cuda', '--batch-size', '16',
                   '--seed', '42', '--training-seed', str(seed)]
        print(f'Starting seed {seed} ({seed - 41}/5); log: {log}', flush=True)
        with log.open('x', encoding='utf-8') as stream:
            process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True)
        if process.returncode:
            print(f'Seed {seed} failed with exit code {process.returncode}; see {log}', flush=True)
            return process.returncode
        result = json.loads((output / 'results.json').read_text(encoding='utf-8'))
        baseline = next(row for row in result['results'] if row['setup'] == 'First model + BM25 80/20')
        optimized = next(row for row in result['results'] if row['setup'] == 'Frozen test: First model + hard negatives + BM25 80/20')
        runs.append({'training_seed': seed, 'seen_p10': optimized['seen_tags_p10'],
                     'unseen_p10': optimized['unseen_tags_p10'], 'title_hit1': optimized['title_hit1'],
                     'title_hit10': optimized['title_hit10'], 'baseline_seen_p10': baseline['seen_tags_p10'],
                     'baseline_unseen_p10': baseline['unseen_tags_p10']})
        print(f'Finished seed {seed}: seen={runs[-1]["seen_p10"]}, unseen={runs[-1]["unseen_p10"]}', flush=True)
    columns = ('seen_p10', 'unseen_p10', 'title_hit1', 'title_hit10')
    summary = {key: {'mean': round(statistics.mean(r[key] for r in runs), 4),
                     'sample_std': round(statistics.stdev(r[key] for r in runs), 4),
                     'min': min(r[key] for r in runs), 'max': max(r[key] for r in runs)} for key in columns}
    payload = {'split_seed': 42, 'training_seeds': list(SEEDS), 'runs': runs, 'summary': summary,
               'limitations': 'Same initial model, mined-data method and previously inspected test split; only trainer seed varies. Metrics rounded to 3 decimals before summary.'}
    (RUNS / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Hard negatives: 5 seed huấn luyện trên data cũ', '',
             'Cùng model khởi đầu và cùng split seed 42; chỉ thay seed huấn luyện giai đoạn hard negatives. Test đã được xem trước đó.', '',
             '| Seed | Seen P@10 | Unseen P@10 | Title hit@1 | Title hit@10 |', '| ---: | ---: | ---: | ---: | ---: |']
    for row in runs:
        lines.append(f'| {row["training_seed"]} | {row["seen_p10"]:.3f} | {row["unseen_p10"]:.3f} | {row["title_hit1"]:.3f} | {row["title_hit10"]:.3f} |')
    lines += [f'| Trung bình | {summary["seen_p10"]["mean"]:.4f} | {summary["unseen_p10"]["mean"]:.4f} | {summary["title_hit1"]["mean"]:.4f} | {summary["title_hit10"]["mean"]:.4f} |', '',
              f'Baseline + BM25 80/20: Seen {runs[0]["baseline_seen_p10"]:.3f}, Unseen {runs[0]["baseline_unseen_p10"]:.3f}.',
              'Đây không phải 5 lần train lại model gốc hay 5 phép chia test độc lập.']
    (RUNS / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

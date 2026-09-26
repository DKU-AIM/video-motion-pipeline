#!/usr/bin/env python3
"""One measured model session; preserve raw multi-span output without ranking unlabelled data."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
import traceback


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class TokenProgress:
    """Transformers streamer: report token progress without changing stopping or sampling."""
    def __init__(self, path, trial):
        self.path, self.trial = path, trial
        self.prompt_seen = False
        self.count = 0
        self.start = time.perf_counter()
        self.last = 0

    def emit(self, event):
        with self.path.open('a') as f:
            f.write(json.dumps(dict(trial=self.trial, event=event, generated_tokens=self.count,
                                    seconds=time.perf_counter()-self.start, unix_s=time.time())) + '\n')

    def put(self, value):
        if not self.prompt_seen:
            self.prompt_seen = True
            self.emit('prompt_ready')
            return
        self.count += value.numel()
        now = time.perf_counter()
        if self.count == 1 or self.count % 16 == 0 or now-self.last >= 5:
            self.emit('token_progress')
            self.last = now

    def end(self):
        self.emit('generation_end')


def extract_spans(raw):
    """Keep all explicit final spans; flag noncanonical formats and incomplete arrays."""
    answers = re.findall(r'<answer>(.*?)</answer>', raw, re.S)
    scope = answers[-1] if answers else re.sub(r'<think>.*?(?:</think>|$)', '', raw, flags=re.S)
    scope = scope.strip()
    if scope.startswith('```'):
        scope = re.sub(r'^```(?:json)?\s*', '', scope)
        scope = re.sub(r'\s*```$', '', scope)
    # Qwen occasionally emits unquoted mm:ss values in array positions.
    scope, timestamps = re.subn(r'(?<=[\[,])\s*(\d+):([0-5]\d(?:\.\d+)?)\s*(?=[,\]])',
                                lambda m: str(int(m[1])*60 + float(m[2])), scope)
    def number(value):
        if isinstance(value, bool):
            raise ValueError('boolean is not a timestamp')
        if isinstance(value, str) and re.fullmatch(r'\d+:[0-5]\d(?:\.\d+)?', value):
            a, b = value.split(':')
            return int(a)*60 + float(b)
        return float(value)
    def pair(value):
        if isinstance(value, dict) and 'start_seconds' in value and 'end_seconds' in value:
            return [number(value['start_seconds']), number(value['end_seconds'])]
        if isinstance(value, list) and len(value) == 2 and not any(isinstance(x, (dict,list)) for x in value):
            return [number(x) for x in value]
        raise ValueError('not a span')
    decoder = json.JSONDecoder()
    spans = []
    position = 0
    partial = False
    objects = False
    while position < len(scope):
        match = re.search(r'[\[{]', scope[position:])
        if not match:
            break
        begin = position + match.start()
        try:
            value, size = decoder.raw_decode(scope[begin:])
        except ValueError:
            partial = True
            position = begin + 1
            continue
        position = begin + size
        try:
            if value == [] and not spans and not partial:
                return [], 'json_empty'
            if isinstance(value, list) and value and isinstance(value[0], (list,dict)):
                spans.extend(pair(x) for x in value)
                objects |= any(isinstance(x, dict) for x in value)
            else:
                spans.append(pair(value))
                objects |= isinstance(value, dict)
        except (ValueError, TypeError, OverflowError):
            continue
    if spans:
        label = 'json_objects' if objects else ('timestamp_pairs' if timestamps else 'json')
        return spans, ('partial_' if partial else '') + label
    number_pattern = r'(-?\d+(?:\.\d+)?)'
    hits = re.findall(number_pattern + r'\s*(?:to|–|-)\s*' + number_pattern, scope)
    if hits:
        return [[float(a),float(b)] for a,b in hits], 'text_ranges'
    return [], 'unparsed'


def hit_generation_limit(count, limit, last_token, eos_ids):
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    return count >= limit and last_token not in (eos_ids or [])


def valid_span(pair, duration):
    return all(math.isfinite(x) for x in pair) and 0 <= pair[0] < pair[1] <= duration + .1


class Sampler:
    """Process RSS/CPU plus explicitly device-wide (not MIG-attributed) NVIDIA readings."""
    def __init__(self, path):
        self.path = path
        self.phase = 'startup'
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        previous = time.monotonic(), time.process_time()
        with self.path.open('w') as f:
            while not self.stop.is_set():
                now, cpu = time.monotonic(), time.process_time()
                rss = None
                try:
                    match = re.search(r'^VmRSS:\s+(\d+)', Path('/proc/self/status').read_text(), re.M)
                    rss = int(match[1]) * 1024 if match else None
                except OSError:
                    pass
                record = {'unix_s': time.time(), 'phase': self.phase, 'process_rss_bytes': rss,
                          'process_cpu_percent_one_core': 100 * (cpu - previous[1]) / max(now - previous[0], 1e-6),
                          'gpu_scope': 'physical GPUs visible to nvidia-smi; NOT attributable to this MIG process',
                          'gpu_fields': ['uuid','utilization.gpu','memory.used','power.draw'], 'gpu_units': ['id','percent','MiB','W']}
                previous = now, cpu
                try:
                    p = subprocess.run(['nvidia-smi', '--query-gpu=uuid,utilization.gpu,memory.used,power.draw',
                                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
                    record['gpu_rows'] = list(csv.reader(p.stdout.splitlines())) if p.returncode == 0 else None
                    record['gpu_error'] = p.stderr.strip() or None
                except (OSError, subprocess.TimeoutExpired) as exc:
                    record['gpu_rows'] = None
                    record['gpu_error'] = str(exc)
                f.write(json.dumps(record) + '\n')
                f.flush()
                self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=4)


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--queries', type=Path, default=root / 'experiments/basketball/queries.json')
    p.add_argument('--ids', nargs='+', default=['shot', 'bench', 'closeup'])
    p.add_argument('--variants', choices=['first', 'all'], default='first')
    p.add_argument('--model', default='timelens2-4b')
    p.add_argument('--prompt', choices=['native', 'all-json'], default='native')
    p.add_argument('--fps', type=float, default=1)
    p.add_argument('--max-frames', type=int, default=256)
    p.add_argument('--total-tokens', type=int, default=16384)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--repeat', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not args.video.is_file():
        p.error(f'Missing video: {args.video}')
    if min(args.fps, args.max_frames, args.total_tokens, args.max_new_tokens, args.repeat) <= 0:
        p.error('Sampling, budget and repeat values must be positive')
    specs = json.loads(args.queries.read_text())
    known = {q['id'] for q in specs}
    if set(args.ids) - known:
        p.error(f'Unknown query IDs: {set(args.ids) - known}')
    if args.out.exists() and any(args.out.iterdir()):
        p.error('Use a new/empty output directory; previous results are protected')
    import torch
    import vtg_run as v
    if args.model not in v.MODELS:
        p.error(f'Unknown model: {args.model}')
    # Processor dispatch follows model identity, independently of output instructions.
    class ProfileGrounder(v.Grounder):
        def _vision_info(self, messages):
            active = self.style
            try:
                self.style = v.STYLES[self.cfg['style']]
                return super()._vision_info(messages)
            finally:
                self.style = active
    args.out.mkdir(parents=True, exist_ok=True)
    print(f'Results: {args.out.resolve()}', flush=True)
    video = args.video.resolve()
    settings = {k: str(val) if isinstance(val, Path) else val for k, val in vars(args).items()}
    settings['video'] = str(video)
    settings['video_sha256'] = hashlib.sha256(video.read_bytes()).hexdigest()
    settings['video_duration_s'] = v.video_duration(video)
    settings['queries_snapshot'] = specs
    settings['model_repo'] = v.MODELS[args.model]['repo']
    settings['sampling_note'] = 'Requested fps/frame/token budgets; actual visual grid recorded per trial. total_tokens is a visual pixel budget, not a strict input-token cap.'
    settings['timing_note'] = 'First query is cold; model download/load measured separately. No automatic OOM budget changes.'
    settings['effective_use_cache'] = True
    settings['generation_progress_file'] = 'generation_progress.jsonl'
    settings['evaluation'] = 'unlabelled: no accuracy score; empty JSON and parse failure are distinct'
    write_json(args.out / 'settings.json', settings)
    for name, cmd in [('pip-freeze.txt', [os.sys.executable, '-m', 'pip', 'freeze']), ('gpu.txt', ['nvidia-smi']),
                      ('code-commit.txt', ['git', '-C', str(root), 'rev-parse', 'HEAD'])]:
        result = subprocess.run(cmd, capture_output=True, text=True)
        (args.out / name).write_text(result.stdout + result.stderr)
    failed = False
    with Sampler(args.out / 'resources.jsonl') as monitor:
        monitor.phase = 'preflight'
        v.preflight()
        monitor.phase = 'model_load_including_download'
        load_start = time.perf_counter()
        try:
            g = ProfileGrounder(args.model, total_tokens=args.total_tokens, fps=args.fps, max_frames=args.max_frames, attn='sdpa')
        except Exception as exc:
            write_json(args.out / 'load.json', {'status': 'error', 'seconds': time.perf_counter() - load_start, 'error': f'{type(exc).__name__}: {exc}'})
            raise
        write_json(args.out / 'load.json', {'status': 'ok', 'seconds': time.perf_counter() - load_start,
                                         'model_commit': getattr(g.model.config, '_commit_hash', None),
                                         'checkpoint_use_cache': getattr(g.model.generation_config, 'use_cache', None),
                                         'effective_use_cache': True})
        native_parser = g.style['parse']
        if args.prompt == 'all-json':
            prefix = 'Numbers before each video frame indicate its sampling timestamp in seconds. ' if args.model == 'timelens-7b' else ''
            g.style = dict(g.style, prompt=prefix + 'Find EVERY separate visible occurrence of "{}". Return only a JSON array of [start_seconds, end_seconds] pairs. Do not merge separated occurrences. If none are visible, return [].')
        (args.out / 'prompt-template.txt').write_text(g.style['prompt'])
        count = 0
        fields = ['trial', 'query_id', 'variant', 'repeat', 'status', 'parse_status', 'span_count', 'invalid_span_count',
                  'preprocess_s', 'generate_s', 'decode_parse_s', 'wall_s', 'input_tokens', 'output_tokens',
                  'hit_output_limit', 'incomplete_output', 'legacy_span_count', 'legacy_discards_spans', 'peak_allocated_gib', 'peak_reserved_gib', 'error']
        try:
            with (args.out / 'results.jsonl').open('w') as jf, (args.out / 'summary.csv').open('w', newline='') as cf:
                writer = csv.DictWriter(cf, fields, extrasaction='ignore')
                writer.writeheader()
                for repeat in range(args.repeat):
                    for q in specs:
                        if q['id'] not in args.ids:
                            continue
                        queries = q['queries'] if args.variants == 'all' else q['queries'][:1]
                        for variant, query in enumerate(queries):
                            count += 1
                            row = dict(trial=count, query_id=q['id'], variant=variant, repeat=repeat,
                                       query=query, status='error', cold_query=count == 1, spans=[], error=None)
                            monitor.phase = f'trial_{count}:preprocess'
                            torch.cuda.synchronize()
                            torch.cuda.reset_peak_memory_stats()
                            begin = time.perf_counter()
                            inputs = output = tokens = None
                            try:
                                with torch.inference_mode():
                                    inputs = g._build_inputs(video, query)
                                    torch.cuda.synchronize()
                                    prepared = time.perf_counter()
                                    row['preprocess_s'] = prepared - begin
                                    row['input_tokens'] = int(inputs.input_ids.shape[1])
                                    grid = inputs.get('video_grid_thw')
                                    row['video_grid_thw'] = grid.tolist() if grid is not None else None
                                    monitor.phase = f'trial_{count}:generate'
                                    output = g.model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
                                                              streamer=TokenProgress(args.out / 'generation_progress.jsonl', count))
                                    torch.cuda.synchronize()
                                    generated = time.perf_counter()
                                    row['generate_s'] = generated - prepared
                                    tokens = output[0][row['input_tokens']:]
                                    row['output_tokens'] = len(tokens)
                                    row['hit_output_limit'] = hit_generation_limit(len(tokens), args.max_new_tokens, int(tokens[-1]) if len(tokens) else None, g.model.generation_config.eos_token_id)
                                    row['raw'] = g.processor.batch_decode([tokens], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                                    # A legacy parser error must not lose an otherwise valid raw response.
                                    try:
                                        row['legacy_spans'] = native_parser(row['raw'])
                                    except Exception as exc:
                                        row['legacy_spans'] = None
                                        row['legacy_parser_error'] = str(exc)
                                    row['spans'], row['parse_status'] = extract_spans(row['raw'])
                                    row['invalid_span_count'] = sum(not valid_span(s, settings['video_duration_s']) for s in row['spans'])
                                    row['span_count'] = len(row['spans'])
                                    row['incomplete_output'] = bool(row['hit_output_limit'] or row['parse_status'].startswith('partial_'))
                                    row['legacy_span_count'] = len(row['legacy_spans']) if row.get('legacy_spans') is not None else None
                                    row['legacy_discards_spans'] = row['legacy_span_count'] is not None and row['legacy_span_count'] < row['span_count']
                                    row['decode_parse_s'] = time.perf_counter() - generated
                                    row['status'] = 'ok'
                            except Exception as exc:
                                failed = True
                                row['error'] = f'{type(exc).__name__}: {exc}'
                                row['exception_traceback'] = traceback.format_exc()
                                print(row['exception_traceback'], flush=True)
                            finally:
                                row['wall_s'] = time.perf_counter() - begin
                                row['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3
                                row['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 1024**3
                                row['finished_unix_s'] = time.time()
                                monitor.phase = f'trial_{count}:write'
                                jf.write(json.dumps(row, ensure_ascii=False) + '\n'); jf.flush()
                                writer.writerow(row); cf.flush()
                                inputs = output = tokens = None
                                if row['status'] == 'error':
                                    torch.cuda.empty_cache()
                            print(f"trial={count} {q['id']} variant={variant} {row['status']} spans={row.get('span_count')} wall={row['wall_s']:.2f}s", flush=True)
        finally:
            monitor.phase = 'model_free'
            g.free()
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())

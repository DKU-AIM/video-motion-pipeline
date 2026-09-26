#!/usr/bin/env python3
"""Sequential, resumable basketball experiment matrix; one GPU process at a time."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['timelens2-4b', 'timelens2-8b', 'timelens-8b', 'timelens-7b',
          'time-r1-7b', 'time-r1-3b', 'qwen3-vl-8b', 'qwen2.5-vl-7b']


def matrix(models, repeats=3):
    base = dict(fps=1, max_frames=256, total_tokens=16384, max_new_tokens=1024, prompt='all-json', repeat=1)
    configs = [('native', dict(prompt='native', repeat=repeats)),
               ('all_json', dict(repeat=repeats)),
               ('budget_8192', dict(total_tokens=8192)),
               ('budget_32768', dict(total_tokens=32768)),
               ('fps_05', dict(fps=.5, max_frames=512, total_tokens=32768)),
               ('fps_10', dict(fps=1, max_frames=512, total_tokens=32768)),
               ('fps_20', dict(fps=2, max_frames=512, total_tokens=32768)),
               ('output_256', dict(max_new_tokens=256)),
               ('output_2048', dict(max_new_tokens=2048))]
    # Baseline for every model first; parameter sweeps follow.
    return [dict(id=f'{model}__{name}', model=model, condition=name, **(base | changes))
            for name, changes in configs for model in models]


def quick_matrix(models):
    return [dict(id=f'{model}__quick', model=model, condition='quick', fps=1,
                 max_frames=256, total_tokens=16384, max_new_tokens=512,
                 prompt='all-json', repeat=1, variants='first') for model in models]


def v2_matrix(models):
    queries = ['shot', 'dark_shot', 'bench', 'closeup', 'football']
    return [dict(id=f'{model}__{condition}', model=model, condition=condition,
                 fps=1, max_frames=256, total_tokens=16384, max_new_tokens=2048,
                 prompt=prompt, repeat=1, variants='first', query_ids=ids)
            for condition, prompt, ids in [('all_json_5q', 'all-json', queries),
                                           ('default_shot_control', 'native', ['shot'])]
            for model in models]


def parameter_matrix(models):
    conditions = [('A_fps05_tokens32768', .5, 32768),
                  ('B_fps1_tokens32768', 1, 32768),
                  ('C_fps2_tokens32768', 2, 32768),
                  ('D_fps1_tokens16384', 1, 16384)]
    return [dict(id=f'{model}__{name}', model=model, condition=name, fps=fps,
                 max_frames=512, total_tokens=tokens, max_new_tokens=2048,
                 prompt='all-json', repeat=1, variants='first',
                 query_ids=['shot','dark_shot','bench','closeup','football'])
            for name, fps, tokens in conditions for model in models]


def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A killed worker may have left an incomplete final line.
    return rows


def command(job, python, video, queries, ids, out):
    ids = job.get('query_ids', ids)
    args = [str(python), str(ROOT / 'grounding/basketball_probe.py'), '--video', str(video),
            '--out', str(out), '--queries', str(queries), '--ids', *ids, '--variants', job.get('variants', 'all'), '--model', job['model']]
    for key in ['fps', 'max_frames', 'total_tokens', 'max_new_tokens', 'prompt', 'repeat']:
        args += ['--' + key.replace('_', '-'), str(job[key])]
    return args


def summarize(out, plan, state):
    flat = []
    for job in plan['jobs']:
        current = state.get(job['id'], {})
        if not current.get('attempt_dir'):
            continue
        attempt = Path(current['attempt_dir'])
        result_file = attempt / 'results.normalized.jsonl'
        if not result_file.exists():
            result_file = attempt / 'results.jsonl'
        for row in read_rows(result_file):
            flat.append(dict(job_id=job['id'], model=job['model'], condition=job['condition'],
                             job_status=current['status'], fps=job['fps'], max_frames=job['max_frames'],
                             total_tokens=job['total_tokens'], max_new_tokens=job['max_new_tokens'],
                             prompt=job['prompt'], **row))
    fields = ['job_id','model','condition','job_status','fps','max_frames','total_tokens','max_new_tokens','prompt',
              'trial','query_id','query','spans','variant','repeat','status','parse_status','span_count','invalid_span_count',
              'wall_s','preprocess_s','generate_s','decode_parse_s','peak_allocated_gib','peak_reserved_gib',
              'input_tokens','output_tokens','hit_output_limit','incomplete_output','legacy_span_count','legacy_discards_spans','error']
    tmp = out / 'all_trials.tmp'
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(flat)
    tmp.replace(out / 'all_trials.csv')
    counts = {s: sum(state.get(j['id'], {}).get('status', 'pending') == s for j in plan['jobs'])
              for s in ['pending','running','success','failed','timeout','interrupted']}
    text = ['# Basketball batch status', '', f'Updated UTC: {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}',
            f'Jobs: {len(plan["jobs"])}; planned query calls: {plan["planned_trials"]}',
            f'Status: {json.dumps(counts)}', f'Recorded query calls: {len(flat)}', '',
            'Accuracy is not scored without reviewed ground truth. Counts are NOT correctness.', '',
            '| Model | Condition | Status | Calls | Session seconds |', '|---|---|---|---:|---:|']
    for job in plan['jobs']:
        s = state.get(job['id'], {})
        n = sum(r['job_id'] == job['id'] for r in flat)
        text.append(f"| {job['model']} | {job['condition']} | {s.get('status','pending')} | {n}/{job['expected_trials']} | {s.get('elapsed_s','')} |")
    (out / 'progress.md').write_text('\n'.join(text) + '\n')
    if plan.get('suite') in ('quick','v2','parameters'):
        report = ['# Quick model comparison', '',
                  'Provisional outputs, NOT an accuracy ranking. Human verification of ground truth is pending.',
                  'Conditions are shown separately: all-json requests all spans; native uses the repository default template. Counts are not accuracy.', '',
                  '| Model | Condition | Query | Status | Predicted spans (s) | Wall s | Peak allocated GiB |',
                  '|---|---|---|---|---|---:|---:|']
        for row in flat:
            report.append(f"| {row['model']} | {row['condition']} | {row.get('query_id')} | {row.get('status')} / {row.get('parse_status','')}{' (OUTPUT LIMIT)' if row.get('hit_output_limit') else ''} | {json.dumps(row.get('spans',[]))} | {row.get('wall_s',0):.2f} | {row.get('peak_allocated_gib',0):.2f} |")
        report += ['', 'Session-level failures/timeouts and unstarted models: see progress.md.',
                   'All response text, errors and settings: all_responses.jsonl. Detailed resources: each attempt directory.']
        (out / 'comparison.md').write_text('\n'.join(report) + '\n')
        tmp = out / 'all_responses.tmp'
        tmp.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in flat))
        tmp.replace(out / 'all_responses.jsonl')
        if plan.get('suite') == 'parameters':
            table = ['# FPS and visual budget comparison', '',
                     'A/B/C vary requested FPS at fixed visual budget; B/D vary budget at fixed FPS.',
                     'These are descriptive resource measurements, NOT accuracy scores. Actual visual resolution may change with FPS.', '',
                     '| Model | Condition | Status | OK calls | Mean wall s | Mean preprocess s | Mean generate s | Peak GPU GiB | Mean input tokens | Incomplete |',
                     '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
            for job in plan['jobs']:
                rs = [r for r in flat if r['job_id'] == job['id']]
                ok = [r for r in rs if r.get('status') == 'ok']
                def avg(field):
                    return f"{sum(r.get(field, 0) for r in ok)/len(ok):.2f}" if ok else '-'
                peak = max((r.get('peak_allocated_gib',0) for r in rs), default=0)
                table.append(f"| {job['model']} | {job['condition']} | {state.get(job['id'],{}).get('status','pending')} | {len(ok)}/5 | {avg('wall_s')} | {avg('preprocess_s')} | {avg('generate_s')} | {peak:.2f} | {avg('input_tokens')} | {sum(bool(r.get('incomplete_output')) for r in ok)} |")
            table += ['', 'Query-level spans and raw answers: comparison.md / all_responses.jsonl.',
                      'Error details and failed calls are preserved; averages above include successful calls only.']
            (out / 'parameter_summary.md').write_text('\n'.join(table) + '\n')

        if plan.get('suite') == 'v2':
            comparison = ['# Prompt and parser comparison: shot query', '',
                          'Default = repository template, not a claim about the authors’ exact inference setup.',
                          'Counts measure returned spans, not verified events. Compare raw responses before drawing conclusions.', '',
                          '| Model | All-json spans / legacy parser | Default spans / legacy parser | Notes |',
                          '|---|---|---|---|']
            for model in dict.fromkeys(j['model'] for j in plan['jobs']):
                selected = {r['condition']: r for r in flat if r['model'] == model and r.get('query_id') == 'shot'}
                cells, notes = [], []
                for condition in ['all_json_5q', 'default_shot_control']:
                    row = selected.get(condition)
                    if row is None:
                        cells.append('pending')
                    elif row.get('status') != 'ok':
                        cells.append('ERROR')
                    else:
                        cells.append(f"{row.get('span_count')} / {row.get('legacy_span_count')}")
                        if row.get('incomplete_output') or row.get('parse_status') == 'unparsed':
                            notes.append(condition + ': incomplete/unparsed')
                comparison.append(f"| {model} | {cells[0]} | {cells[1]} | {', '.join(notes)} |")
            (out / 'prompt_comparison.md').write_text('\n'.join(comparison) + '\n')



def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, default=Path(os.environ.get('MOTION_WORKSPACE', Path.home() / 'aim-workspace')))
    p.add_argument('--out', type=Path)
    p.add_argument('--models', nargs='+', choices=MODELS, default=None)
    p.add_argument('--baseline-repeats', type=int, default=3)
    p.add_argument('--suite', choices=['full', 'quick', 'v2', 'parameters'], default='full')
    p.add_argument('--deadline-minutes', type=float, default=0, help='Wall-clock cap for this invocation; 0 disables')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--retry-failed', action='store_true', help='Retry failed/time-out jobs once in this invocation')
    p.add_argument('--timeout-hours', type=float, default=6, help='Limit per model/condition session, including download; 0 disables')
    args = p.parse_args()
    if args.baseline_repeats < 1 or args.timeout_hours < 0 or args.deadline_minutes < 0:
        p.error('Repeats and timeout must be positive')
    workspace = args.workspace.resolve()
    folder = {'quick':'basketball_quick', 'v2':'basketball_v2', 'full':'basketball_all', 'parameters':'basketball_parameters'}[args.suite]
    out = (args.out or workspace / 'results_grounding' / folder).resolve()
    video = workspace / "inputs/grounding/2026_AsianGames_men's_basketball_final.mp4"
    python = Path(os.environ.get('GROUNDING_PYTHON', str(workspace / 'grounding_env/bin/python')))
    queries = ROOT / 'experiments/basketball/queries.json'
    if not video.is_file() or not python.is_file():
        p.error('Video or grounding Python missing')
    spec = json.loads(queries.read_text())
    ids = (['shot','dark_shot','bench','closeup','football'] if args.suite in ('v2','parameters') else
           ['shot', 'bench', 'closeup', 'football'] if args.suite == 'quick' else [q['id'] for q in spec])
    n = len(ids) if args.suite in ('quick','v2','parameters') else sum(len(q['queries']) for q in spec)
    models = list(dict.fromkeys(args.models or (['timelens2-4b','timelens-8b'] if args.suite == 'parameters' else MODELS)))
    if args.suite == 'parameters':
        jobs = parameter_matrix(models)
    elif args.suite == 'v2':
        jobs = v2_matrix(models)
    elif args.suite == 'quick':
        jobs = quick_matrix(models)
    else:
        jobs = matrix(models, args.baseline_repeats)
    for job in jobs:
        job['expected_trials'] = len(job['query_ids']) * job['repeat'] if 'query_ids' in job else n * job['repeat']
    plan = dict(video=str(video), suite=args.suite, query_ids=ids, jobs=jobs, planned_trials=sum(j['expected_trials'] for j in jobs),
                video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                queries_sha256=hashlib.sha256(queries.read_bytes()).hexdigest(),
                code_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                             for name in ['grounding/basketball_probe.py', 'grounding/vtg_run.py', 'grounding/basketball_batch.py']})
    print(f"{len(jobs)} sessions / {plan['planned_trials']} query calls; output: {out}", flush=True)
    if args.dry_run:
        for job in jobs:
            print(job['id'], {k: job[k] for k in ['fps','max_frames','total_tokens','max_new_tokens','prompt','repeat','expected_trials']})
        return 0
    out.mkdir(parents=True, exist_ok=True)
    # Across different batch output directories as well, allow only one batch to own this workspace GPU.
    with (workspace / '.basketball-batch.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            p.error('Another basketball batch is running in this workspace')
        if (out / 'plan.json').exists():
            if json.loads((out / 'plan.json').read_text()) != plan:
                p.error('Plan/source/code changed; choose a new --out directory to avoid mixing experiments')
        else:
            if any(out.iterdir()):
                p.error('Output has unrelated files; use a new directory')
            atomic(out / 'plan.json', plan)
        state_path = out / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        for s in state.values():
            if s['status'] == 'running':
                s['status'] = 'interrupted'
        summarize(out, plan, state)
        deadline = time.monotonic() + args.deadline_minutes * 60 if args.deadline_minutes else float('inf')
        for index, job in enumerate(jobs, 1):
            if time.monotonic() >= deadline:
                print('Invocation time limit reached; remaining jobs preserved as pending.', flush=True)
                return 2
            previous = state.get(job['id'], {})
            if previous.get('status') == 'success':
                continue
            if previous.get('status') in ('failed','timeout') and not args.retry_failed:
                continue
            if shutil.disk_usage(workspace).free < 5 * 1024**3:
                print('Stopped: less than 5 GiB free; free space and rerun to resume.', flush=True)
                return 2
            folder = out / job['id']
            folder.mkdir(exist_ok=True)
            attempt = 1
            while (folder / f'attempt_{attempt:03}').exists():
                attempt += 1
            attempt_dir = folder / f'attempt_{attempt:03}'
            # Keep worker log outside attempt_dir because the worker protects nonempty outputs.
            log = folder / f'attempt_{attempt:03}.log'
            cmd = command(job, python, video, queries, ids, attempt_dir)
            record = dict(status='running', attempt=attempt, attempt_dir=str(attempt_dir), log=str(log),
                          started_unix_s=time.time(), command=cmd)
            state[job['id']] = record
            atomic(state_path, state)
            print(f"[{index}/{len(jobs)}] START {job['id']} -> {log}", flush=True)
            started = time.monotonic()
            process = None
            try:
                with log.open('w') as f:
                    process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True,
                                               env=dict(os.environ, PYTHONUNBUFFERED='1'))
                    record['pid'] = process.pid
                    atomic(state_path, state)
                    while process.poll() is None:
                        if (args.timeout_hours > 0 and time.monotonic() - started > args.timeout_hours * 3600) or time.monotonic() >= deadline:
                            terminate(process)
                            record['status'] = 'timeout'
                            break
                        summarize(out, plan, state)
                        time.sleep(10)
                    code = process.wait()
                    rows = read_rows(attempt_dir / 'results.jsonl')
                    if record['status'] != 'timeout':
                        record['status'] = 'success' if code == 0 and len(rows) == job['expected_trials'] and all(r.get('status') == 'ok' for r in rows) else 'failed'
                    record['returncode'] = code
            except KeyboardInterrupt:
                if process is not None:
                    terminate(process)
                record['status'] = 'interrupted'
                print('Interrupted; rerun the same command to resume.', flush=True)
                return 130
            except Exception as exc:
                record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            finally:
                if process is not None and process.poll() is None:
                    terminate(process)
                record['elapsed_s'] = round(time.monotonic() - started, 2)
                record['finished_unix_s'] = time.time()
                atomic(state_path, state)
                summarize(out, plan, state)
            print(f"[{index}/{len(jobs)}] {record['status'].upper()} {job['id']} ({record['elapsed_s']} s)", flush=True)
        failed = sum(s['status'] != 'success' for s in state.values())
        print(f'Batch finished. Non-success sessions: {failed}. See {out / "progress.md"}', flush=True)
        return 1 if failed else 0


if __name__ == '__main__':
    # tmux session shutdown/TERM must also stop the active child GPU process.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    raise SystemExit(main())

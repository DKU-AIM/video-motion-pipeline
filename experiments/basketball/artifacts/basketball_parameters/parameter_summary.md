# FPS and visual budget comparison

A/B/C vary requested FPS at fixed visual budget; B/D vary budget at fixed FPS.
These are descriptive resource measurements, NOT accuracy scores. Actual visual resolution may change with FPS.

| Model | Condition | Status | OK calls | Mean wall s | Mean preprocess s | Mean generate s | Peak GPU GiB | Mean input tokens | Incomplete |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| timelens2-4b | A_fps05_tokens32768 | success | 5/5 | 28.72 | 15.91 | 12.80 | 16.43 | 32255.00 | 0 |
| timelens-8b | A_fps05_tokens32768 | success | 5/5 | 40.52 | 16.27 | 24.25 | 25.68 | 32255.00 | 0 |
| timelens2-4b | B_fps1_tokens32768 | success | 5/5 | 37.51 | 27.48 | 10.03 | 15.94 | 30412.00 | 0 |
| timelens-8b | B_fps1_tokens32768 | success | 5/5 | 53.53 | 29.21 | 24.31 | 25.11 | 30412.00 | 0 |
| timelens2-4b | C_fps2_tokens32768 | success | 5/5 | 52.86 | 40.79 | 12.07 | 16.84 | 34131.00 | 0 |
| timelens-8b | C_fps2_tokens32768 | success | 5/5 | 65.80 | 38.91 | 26.89 | 26.14 | 34131.00 | 0 |
| timelens2-4b | D_fps1_tokens16384 | success | 5/5 | 32.03 | 26.00 | 6.03 | 12.56 | 17092.00 | 0 |
| timelens-8b | D_fps1_tokens16384 | success | 5/5 | 41.31 | 27.04 | 14.27 | 21.24 | 17092.00 | 0 |

Query-level spans and raw answers: comparison.md / all_responses.jsonl.
Error details and failed calls are preserved; averages above include successful calls only.

# Prompt and parser comparison: shot query

Default = repository template, not a claim about the authors’ exact inference setup.
Counts measure returned spans, not verified events. Compare raw responses before drawing conclusions.

| Model | All-json spans / legacy parser | Default spans / legacy parser | Notes |
|---|---|---|---|
| timelens2-4b | 14 / 14 | 5 / 5 |  |
| timelens2-8b | 1 / 1 | 1 / 1 |  |
| timelens-8b | 30 / 0 | 1 / 1 |  |
| timelens-7b | 13 / 0 | 1 / 1 |  |
| time-r1-7b | 11 / 0 | 1 / 1 |  |
| time-r1-3b | 0 / 0 | 1 / 1 |  |
| qwen3-vl-8b | 98 / 0 | 1 / 1 |  |
| qwen2.5-vl-7b | 91 / 0 | 1 / 1 | all_json_5q: incomplete/unparsed |

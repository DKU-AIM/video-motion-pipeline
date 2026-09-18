# Grounding → CoMotion → MotionGPT demo

User-approved scope: one bounded video/query demo with visual review; keep existing batch defaults.

- Prepare separate pose/caption environments from pinned official source revisions. Require authorized SMPL neutral asset; do not redistribute it. Record actual versions. No local multi-GB downloads.
- Add CoMotion-only/full-frame options to existing pose batch and remove eager Multi-HMR2 imports on that route. Propagate inference/render failure as nonzero exit.
- Add stdlib orchestrator: validate prerequisites, take a 30-second source window, run one grounder, clip first chronological valid span to max six seconds, run pose/export/caption sequentially in child processes. Use unique run workspace; reject overwrites; persist status and logs.
- Keep original timestamps and per-track caption intervals. Clearly state longest-track selection is not identity grounding. Render escaped offline HTML containing comparison video, track IDs, timestamps, captions and output links. Never invent outputs when stages fail or no track survives.
- Tests: real ffmpeg clipping if available; mocked model subprocess pipeline including failure propagation; interval validation and unsafe HTML escaping; parser regressions and setup shell checks. GPU/model output validation remains a separate server check.

Owners: parent scripts/run_motion_demo.py, tests, README/docs and export provenance; pose_adapter mesh_recovery files; motion_setup setup script/requirements/docs/MOTION_SETUP.md. Preserve bootstrap work.

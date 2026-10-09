Evidence for the seedling coding agent (BASELINE -> PATCH -> VERIFY roles) on DeepSWE:
- 'logs/DOSSIERS.md' -- START HERE. Ranks this node's failed tasks and points at each one's dossier.
- 'logs/scratch/<task>/<run>/dossier.md' -- one task run in ~3 KB: reward, hidden-test pass counts
  (f2p/p2p), failure classes (e.g. verify_false_pass, near_miss, role_wall), each role's wall time and
  stop reason, the task's requirement checklist vs what PATCH listed and what VERIFY tested, and
  contrasts with other nodes' runs of the same task.
- 'logs/scratch/<task>/<run>/transcripts/<role>.<attempt>.txt' -- the full rendered conversation of
  one role attempt (e.g. verify.2.txt). Open the one a dossier cites; grep them, they are large.
- 'logs/scratch/<task>/<run>/exec_log.txt' -- every shell command each role ran, with exit codes.
- 'logs/scratch/<task>/<run>/model.patch' and 'report.md' -- the submitted patch and the run summary.
- 'logs/case_<id>.json' -- the scored result (details: reward, f2p, per-role stats, verdicts).
- 'logs/trace.jsonl' -- synthetic tool_call/llm_call events per role (grep it).
Hidden test names and grader output are deliberately NOT in logs/.

"""Optional LLM layer: natural-language intake and report narrative.

Both are off by default and sit outside the analysis itself. Nothing an
LLM writes here becomes a claim, changes a confidence, or reaches the
report unchecked:

- `intake.plan_goal` turns a free-text question into the two opt-in flags
  on `AnalyticalGoal`. It cannot add a modality, skip QC, or bypass Critic
  -- Coordinator still plans the run procedurally (Constitution Art. III §1).
- `narrative.write_narrative` drafts a prose summary of claims that have
  already passed Critic. Every sentence must cite the findings it rests on,
  and `narrative.verify_narrative` -- plain code, no model -- drops any
  sentence that cites nothing, cites something that isn't in the report, or
  names a result its citations don't support. A kept sentence is rendered
  at the confidence tier of its weakest citation (Art. VI §2).

Requires the `anthropic` package (`pip install -e ".[llm]"`) and API
credentials in the environment.
"""

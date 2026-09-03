"""Phase 5 -- Evolutionary Learning Loop (Constitution Art. VII, Art. XII §6).

Expert corrections (Art. VII §1) must not merely be logged: they are fed into
a PEFT/LoRA fine-tuning loop so the corrected behavior is reflected in future
predictions, not only the corrected instance (Art. VII §2). This package
implements:

- `celltyping`: the trainable cell-type head + checkpoint (de)serialization
  consumed by prediction.
- `lora_finetune`: the trigger rule (N>=50 corrections OR 7 days, Art. XII §6)
  and the actual training loop that turns `correction_log` rows into a new
  checkpoint.
- `promotion`: the Art. XII §6 promotion gate (no more than 1pt regression on
  any previously-validated platform).
- `feedback_value`: Art. VII §4's per-tissue-type benchmarking of the value of
  expert feedback itself.

See `lora_finetune.py`'s module docstring for the honest, disclosed scope
decision on where PEFT/LoRA is and is not literally applied -- analogous to
how Phase 3's `analyst/fusion.py` and `vision/*` disclosed their own scope
limitations rather than overclaiming.
"""

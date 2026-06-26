"""
extras — optional, project-specific add-ons that are NOT part of the generic core.

Nothing here is imported by the framework by default. These modules are examples
of higher-level logic you can opt into for a specific project. Currently:

  fusion.py — TranscriptFuser: merges transcript + nonverbal-event + sentiment
              streams onto a single media timeline. Specific to multimodal
              streaming use cases (e.g. call triage). Enable in the inference
              API with ENABLE_FUSION=true.
"""

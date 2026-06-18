# TRAIL-PAS (Planning-Alignment Subset)

Extracted from Patronus AI [TRAIL](https://arxiv.org/abs/2505.08638) dataset for Intent Auditor evaluation.

## Files

- `schema.py` — `TrailPASSample` dataclass + planning category definitions
- `extract_pas.py` — TRAIL annotations + traces → trail_pas.jsonl
- `trail_pas.jsonl` — Generated samples (one JSON per line)
- `trail_pas_stats.json` — Category/impact/source distribution

## Statistics

| Metric | Value |
|--------|-------|
| Total samples | 555 |
| Positive (alignment_error=true) | 259 |
| Negative (alignment_error=false) | 296 |
| Sources | GAIA: 432, SWE-Bench: 123 |

### Planning Categories

| Category | Count |
|----------|-------|
| Goal Deviation | 65 |
| Resource Abuse | 57 |
| Task Orchestration | 49 |
| Context Handling Failures | 47 |
| Poor Information Retrieval | 41 |

## Regenerate

```bash
python benchmarks/trail_pas/extract_pas.py
```

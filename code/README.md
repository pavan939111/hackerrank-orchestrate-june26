# Multi-Modal Evidence Review Agent

An AI-powered agent for reviewing insurance-style damage claims using multimodal evidence analysis.

## Architecture

**Principle: The LLM perceives. Python decides.**

4-layer pipeline:
- **Layer 0**: Deterministic ingest — CSV loading, image pre-checks (blur, brightness, pHash)
- **Layer 1**: Bounded agent loop — 4 tool calls via Gemini 3.1 Flash Lite with 10-key round-robin rotation
- **Layer 2**: Deterministic adjudicator — 3 hard gates + verdict engine + justification builder
- **Layer 3**: Schema validator — enum snapping, column order, consistency checks

Three hard gates enforce safety:
1. User history can only ADD risk flags, never flip a verdict
2. Text embedded in images is flagged and discarded — no pathway to the verdict
3. `valid_image` (authenticity) and `evidence_standard_met` (sufficiency) are computed by separate code paths

## Setup

```bash
cd code
pip install -r requirements.txt
```

Create `.env` in the repo root:
```
GEMINI_API_KEY_1=your-key-1
GEMINI_API_KEY_2=your-key-2
...
GEMINI_API_KEY_10=your-key-10
```

## Usage

```bash
# Run on test claims (produces output.csv)
python main.py

# Run on sample claims (for evaluation)
python main.py --sample

# Run with stub (no API calls, for testing plumbing)
python main.py --stub
```

## Evaluation

```bash
# From repo root (not from code/):
cd evaluation

# Run accuracy evaluation on 20 labeled sample claims
python eval_main.py

# Compare batched vs per-image configurations
python compare_configs.py

# Or use the entry point inside code/ (per AGENTS.md §6.1):
python code/evaluation/main.py
```

## File Structure

| File | Purpose |
|------|---------|
| main.py | Entry point, pipeline orchestration |
| config.py | Environment vars, paths, constants |
| schemas.py | Allowed enums, output column order, types |
| ingest.py | CSV loading, image path parsing |
| cv_checks.py | Blur, brightness, pHash, file integrity |
| vlm_client.py | Gemini API wrapper, key round-robin |
| tools.py | 4 tool definitions (parse, evidence, history, inspect) |
| orchestrator.py | Agent loop with tool dispatch |
| adjudicator.py | 3 gates + verdict + severity + justification |
| validator.py | Enum snap, column order, consistency |

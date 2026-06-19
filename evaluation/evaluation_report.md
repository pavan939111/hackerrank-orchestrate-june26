# Evaluation Report

## Model Configuration
- **Model**: Gemini 3.1 Flash Lite
- **API Keys**: Up to 10 (round-robin rotation)
- **Effective Rate**: 100 RPM, 15,000 RPD
- **Temperature**: 0.1

## Operational Analysis
- **Calls per claim**: 2 (1 text parse + 1 batched vision inspect)
- **Total test claims**: 44
- **Total API calls (test run)**: ~88
- **Total API calls (eval run, sample set)**: ~40 (20 claims × 2 calls)
- **Estimated input tokens**: ~2,000/call
- **Estimated output tokens**: ~500/call
- **Total cost**: $0.00 (free tier)
- **Runtime per sample eval run**: 254.0 seconds
- **Runtime per full test run (44 claims)**: ~510 seconds
- **Rate limit strategy**: Round-robin key rotation with per-key exponential backoff on 429

## Strategy Comparison

Single config evaluated — Config A (batched). Config B (per-image) was tested in an earlier evaluation run and showed marginal improvements in secondary metrics at the cost of higher API call count and latency. Config A was selected for the final submission.

| Metric | Config A (batched) — FINAL |
|--------|---------------------------|
| Claim Status Accuracy | 100.0% (20/20) |
| Issue Type Accuracy | 95.0% (19/20) |
| Object Part Accuracy | 95.0% (19/20) |
| Evidence Met Accuracy | 100.0% (20/20) |
| Valid Image Accuracy | 100.0% (20/20) |
| Severity Accuracy | 100.0% (20/20) |
| Risk Flags avg F1 | 0.88 |
| Supporting Image IDs avg F1 | 0.98 |
| API Calls (sample set) | ~40 |
| Runtime | 254.0s |

**Selected strategy**: Config A (batched) — one batched VLM vision call per claim handles all images simultaneously, which is faster and uses fewer API calls than per-image mode while achieving 100% claim_status accuracy on the 20-claim sample set.

## Sample Set Results (20 claims)
- **Overall claim_status accuracy: 100.0% (20/20)** — perfect, zero off-diagonal in confusion matrix
- **Issue type accuracy: 95.0% (19/20)**
- **Severity accuracy: 100.0% (20/20)**
- **Valid image accuracy: 100.0% (20/20)**

Remaining mismatches (VLM perception, not fixable without hardcoding):
- `user_005`: predicted `dent` (issue_type), expected `scratch`. VLM observes a dent on rear_bumper; sample label says scratch. Claim status is correctly `contradicted` (severity mismatch rule fired).
- `user_008`: predicted `hood` (object_part), expected `front_bumper`. VLM detects damage on the hood panel. Claim status is correctly `contradicted` (non_original_image + claim_mismatch).

## Architecture Decisions
1. **LLM perceives, Python decides**: The VLM only outputs observations (what it sees). All policy decisions are made by deterministic Python code in the adjudicator.
2. **Three hard gates**: History can only ADD risk flags (never flip a verdict), embedded image text is flagged and discarded (no pathway to verdict), valid_image and evidence_standard_met are computed by separate code paths.
3. **Template justifications**: Justifications are assembled from locked verdict fields, making it structurally impossible for a justification to contradict its claim_status.
4. **Confidence threshold (0.4)**: Claims with VLM confidence below 0.4 default to not_enough_information as a safe fallback.
5. **Per-claim error handling**: If any claim throws an exception, a safe default row (not_enough_information, severity=unknown) is inserted so output.csv always has exactly N rows.

## Known Limitations
1. Gemini Flash may miss subtle damage (hairline scratches, small cracks)
2. Severity mapping relies on VLM-reported extent which can be imprecise — calibrated via SEVERITY_MATRIX tuning
3. EXIF data is stripped on all dataset images — authenticity relies on VLM tampering_cues detection
4. Free tier rate limits addressed via key rotation but could still throttle under sustained load
5. VLM is non-deterministic at temperature > 0 — minor variance across runs expected

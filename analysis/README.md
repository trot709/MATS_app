# Reportable-results visualization

This analysis compares the sentence-steering condition **S** against:

- **I0:** the original sentence is retained and the continuation is resampled;
- **S-random:** norm-matched random steering;
- **E:** a directly written reconsideration sentence.

Run from the project root:

```bash
/Users/ashersered/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  analysis/visualize_reportable_results.py
```

The script reads `results/reportable_results/`, retrieves each target's I0
continuations from `data/blackmail/`, and writes the figures, target-level CSV,
machine-readable summary, and prose analysis to `analysis/output/`.

The code validates that every included generated arm has 5 sentence bursts and
4 continuations per burst. It uses only the 19 targets shared by S, S-random,
and E; this avoids changing the sample between pairwise S comparisons.

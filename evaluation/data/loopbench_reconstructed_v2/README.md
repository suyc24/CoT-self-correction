# LoopBench Reconstructed

This directory contains a paper-faithful reconstruction of LoopBench based on arXiv:2601.05693.

- Total examples: `700`
- Task split: `7` subtasks, `100` samples each
- Public prompt fragment source: `Figure 10`
- Decoding setup for the baseline script: `{'setting_name': 'conservative', 'temperature': 0.1, 'top_k': 5, 'top_p': 0.95, 'repetition_penalty': 1.1}`

Important caveat:

The original 700 GPT-5-synthesized prompts are not public. This dataset reconstructs the same task families from Table 4 and Table 5 rather than claiming exact recovery of the hidden benchmark.

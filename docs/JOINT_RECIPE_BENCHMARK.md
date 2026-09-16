# Joint recipe benchmark

This smoke benchmark compares the legacy FL recipe with `ai_toolkit_joint_v1` after 300 optimizer steps. It uses two training recordings (8 and 67 seconds), one held-out 8-second recording, rank 32, learning rate 0.0001, accumulation 1, and seed 42. SheetSage2 produced full ABC targets for the joint run.

The common evaluator loads each final AR/NAR pair and measures the same held-out recording with direct and full-score AR prefixes. NAR flow loss uses the same center window, noise seed, and timesteps 0.2, 0.5, and 0.8.

| Result | Base | Legacy | Joint score |
|---|---:|---:|---:|
| Direct AR CE | 5.5408 | 5.7250 | 5.6688 |
| Full-score AR CE | 5.4998 | 5.6410 | 5.5424 |
| NAR flow loss | 1.1484 | 1.1039 | 1.2675 |
| Training time | — | 444.1 s | 236.0 s |
| Peak allocated VRAM | — | 11.60 GB | 9.03 GB |

The joint recipe was 1.88× faster and used 2.57 GB less peak VRAM. Its held-out full-score AR CE was 1.75% lower than legacy, and its score reduced CE by 0.1265 versus 0.0839 for legacy, indicating stronger use of the score. Its held-out NAR flow loss was 14.8% higher than legacy and 10.4% higher than the base model.

This tiny split exposes overfitting and is not a perceptual-quality verdict. Neither trained AR adapter beat the base model on the single held-out recording, and the joint base-NAR path needs more data or training before it can replace the community-NAR legacy path. Use a larger artist dataset and listening tests before changing the default recipe.

Raw results are written to `output/yue2_training/recipe_benchmark.json`. Reproduce with `python tools/benchmark_joint_recipe.py --steps 300` from this node pack using ComfyUI's Python environment.

# Parity fixtures

Each script writes a tiny random-weight checkpoint plus reference logits to
`Tests/MLXLMTests/Resources/Parity/<name>/`. `ParityFixture` in the tests loads it and
compares fMLX's logits with the reference.

Run a script with a Python environment that has the reference implementation installed
(`torch` and `transformers`, or `mlx` and `mlx-lm`):

```sh
python scripts/parity/<name>.py
```

Keep fixtures small (well under 1 MB) and seeded so they regenerate identically.

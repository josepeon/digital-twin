# CI secrets

GitHub Actions in this repo reads one secret. Set it in
**Settings → Secrets and variables → Actions → New repository secret** at:
https://github.com/josepeon/digital-twin/settings/secrets/actions

| Secret | Required by | What it does |
|---|---|---|
| `GROQ_API_KEY` | `eval-identity` job (`.github/workflows/ci.yml`) | When set, the workflow runs the 10-question identity eval against the remote Groq provider on every push to `main`. Free tier — no charges. When unset, that job is a no-op (logs "skipping" and exits 0). Soft-gated with `continue-on-error: true` until a baseline is established. |

## Promoting `eval-identity` to a hard gate

Once the baseline is stable, edit `.github/workflows/ci.yml` and:

1. Remove `continue-on-error: true` from the `eval-identity` job.
2. Switch from `--quick` (10 questions) to the full 46.
3. Add a pass-rate threshold (currently identity eval scores 100%; gate at 95%).

That turns pushes to `main` into a contract: a regression in identity
accuracy fails CI.

## Local equivalent

```bash
export GROQ_API_KEY=gsk_...
python -m tests.eval_identity --quick
```

Same command CI runs.

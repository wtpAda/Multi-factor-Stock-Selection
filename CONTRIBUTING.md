# Contributing

Thanks for your interest in improving this research repo.

## Before Opening a Change

- Keep commits focused on one research or infrastructure change at a time.
- Do not commit local market data, API secrets, or large generated files.
- If you change factor logic or backtest rules, explain the research motivation in the pull request.

## Development Notes

- Install dependencies from `requirements.txt`.
- Keep the repository code-first and data-light.
- Prefer changes that preserve the anti-lookahead design described in `README.md`.

## Data Handling

The repository intentionally does not track raw local market datasets.

Expected local-only files include:

- `combined_sp500_data.csv`
- `spx_data.csv`
- `risk_free_rate.csv`

## Pull Request Checklist

- Update `README.md` if the workflow, inputs, or outputs changed.
- Note any assumptions that affect reproducibility.
- Mention whether charts or performance numbers changed because of logic changes or data changes.

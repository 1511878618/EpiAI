# Contributing to EpiAI

## Development setup

```bash
git clone https://github.com/xutingfeng/EpiAI.git
cd EpiAI
pip install -e ".[dev]"
pre-commit install
```

## Code style

- **Formatter**: [black](https://github.com/psf/black) (line-length=100)
- **Linter**: [ruff](https://github.com/astral-sh/ruff)
- **Type checker**: mypy (optional, for CI)

Run before committing:

```bash
black src/ tests/
ruff check src/
```

## Testing

```bash
pytest                          # all tests
pytest tests/test_trainer.py    # single file
pytest -k "test_sklearn"        # keyword match
```

## Pull request

1. Fork the repo
2. Create a feature branch
3. Run tests and lint locally
4. Open a PR against `main`
5. CI runs automatically

## Adding a new model

```python
from EpiAI.models.base import SklearnMixin
from EpiAI.models.registry import register

@register("MyModel")           # ← one line
class MyForecaster(SklearnMixin):
    def fit(self, train_x, train_y, val_x=None, val_y=None): ...
    def predict(self, x): ...
```

## Reporting issues

Open a GitHub issue with:

- EpiAI version (`pip show EpiAI`)
- Python version
- Full traceback
- Minimal reproduction code or data

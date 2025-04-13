# SWE Bench Infinite

## Development Setup

This project uses pre-commit hooks to ensure code quality:

0. Install
```
firefox
geckodriver
```

1. Install dependencies:
   ```
   pip install -e .[dev]
   ```

2. Install pre-commit hooks:
   ```
   pre-commit install
   ```

3. The following checks will run automatically on commit:
   - Mypy for type checking
   - Ruff for linting and formatting

You can also run pre-commit manually:
```
pre-commit run --all-files
```

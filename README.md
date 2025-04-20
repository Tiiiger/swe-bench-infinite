# SWE Bench Infinite: Scaling RL Environments for SWE Agents

<p align="center">
    <a href="https://www.python.org/">
        <img alt="Build" src="https://img.shields.io/badge/python-3.12%2B-blue">
    </a>
    <a href="https://copyright.princeton.edu/policy">
        <img alt="License" src="https://img.shields.io/badge/License-MIT-blue">
    </a>
</p>


I hope this repo can provide a good starting point for continuing this line of work.

## Development Setup

This project uses pre-commit hooks to ensure code quality:

0. Install browser drivers. This step is platform-dependent.
   ```
   firefox
   geckodriver
   ```

   You can verify your installation by running `python src/version_finder.py` to check if all test cases pass.

1. Install dependencies:
   ```
   pip install -e .[dev]
   ```

2. Install pre-commit hooks:
   ```
   pre-commit install
   ```

   The following checks will run automatically on commit:
   - Mypy for type checking
   - Ruff for linting and formatting

3. Install submodules:
   ```
   git submodule update --init --recursive
   cd src/swe_bench
   pip install -e .
   ```

   Note: this repository currently uses my fork of `swe-bench` with fixes to the log parsers. You can review all changes on this [branch](https://github.com/Tiiiger/SWE-bench/pull/1/files). We've imported limited functionality from `SWE-bench`, and migrating these functions directly into this project may be beneficial in the future.

4. Please set `ANTHROPIC_API_KEY` and `GITHUB_TOKEN` in your environment variables.

## Experiments

### Replicating my results

1. Download experiment logs from [here](https://drive.google.com/file/d/1vpGyNfQTu8hZMgwkgX2oP53sw_cEC5f9/view?usp=drive_link). Code for generating figures and analysis is available in `src/report/plot.ipynb`.

2. Examining these logs provides valuable insights into model behavior, including retry patterns and failure modes.

### Running the experiments

The main entrypoint is `src/main.py`. Run experiments with:

```
python src/main.py --start-index 0 --num-processes 32 --exp-name report
```

Please remember to adjust the number of processes to your machine's capacity.

You should add the `--use-test-lock` flag when running tests for `requests`. This prevents running tests concurrently, which caused problems in my experiments.

```
python src/main.py --start-index 0 --num-processes 32 --exp-name report/requests --dataset psf/requests --clean-up --use-test-lock
```

Please bear in mind that the current docker infra could lead to dangling images, and other build caches. Running these experiments could take up a lot of disk space. I ended up babysitting the experiments I reported in the blogpost, and call `docker system prune` in between.

## Contributing

I have described a list of research directions in my blogpost, and here we mention a few engineering todos.

1. The most important item is to use remote code execution so that this data collection process becomes scalable. Based on `SWE-bench`, `modal` is a good candidate.

2. Current implementation only supports Claude. Expanding to other LLMs could be a good first issue.

3. Current implementation only uses file-based logging, and integrating with some monitoring platforms would be crucial for scaling.

4. Tweak the prompt structure can enable prompt caching and save costs.

## Acknowledgements

Many things are borrowed from `SWE-bench`.

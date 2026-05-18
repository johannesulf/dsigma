# Contributing to ``dsigma``

We welcome contributions to ``dsigma`` from community members. Please find the necessary steps and requirements for code contributions below.

## Workflow

All code contributions should be submitted via pull requests on GitHub. To do this, you will **fork** ``dsigma`` and create a feature branch that contains your code contributions. Once you are satisfied, create a pull request against the ``main`` branch of <https://github.com/johannesulf/dsigma>. A package maintainer will then work with you on your proposed changes.

## Requirements

Below, we list requirements that successful code contributions must fulfill. All commands assume that you are in the main ``dsigma`` directory.

### Installation

Make sure your modified version of ``dsigma`` installs correctly via:

```
pip install -e .
```

In case your modifications require updated dependencies, make sure to list those in the ``pyproject.toml`` file.

### Unit Tests

We use ``pytest`` and GitHub Actions for continuous integration. Once you create a pull request, GitHub Actions will automatically trigger unit tests. Those must pass before your contribution can be accepted. To run the unit tests locally, from the main ``dsigma`` directory, run:

```
pytest tests
```

If your code contributions add new features, we also encourage you to add unit tests for those new features under ``tests`` that test those new features.

### Docstrings

Please ensure that all public functions you contribute have valid [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).

### Syntax

Please ensure that the modified code complies with the PEP8 syntax standard. GitHub Actions uses ``ruff`` to test for this and your code contributions can only be accepted after they comply with PEP8. To run the syntax check locally, from the main ``dsigma`` directory, run:

```
ruff check dsigma
```

### Generative AI

If you used generative AI to help you draft your code contributions, you must disclose that in the pull request. Additionally, if new code was substantially written by AI, you must outline what steps you have taken to ensure that the code works correctly.

Generally, we discourage community members from contributing code that they could not have written without AI since, in this case, they may not be able to verify its accuracy. On the other hand, using AI to check existing code or to copyedit documentation may be a good idea.
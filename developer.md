# Instructions for Developers

## Environment Setup

Minimal conda environment with all the dependencies of `seasmon-xr` and all the
linting tools is provided in `dev-env.yml` file.

```
mamba env create -f dev-env.yml
conda activate vam-seasmon
pip install -e .
```


## VScode Workspace Configuration

```json
{
    "python.formatting.provider": "black",
    "python.sortImports.args": [
        "--profile=black",
        "--py=38"
    ],
    "autoDocstring.startOnNewLine": true,
    "python.linting.mypyEnabled": true,
    "python.analysis.typeCheckingMode": "basic",
    "python.linting.pydocstyleEnabled": true,
    "python.linting.pydocstyleArgs": [
        "--max-line-length=120"
    ],
   "python.linting.pylintEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "-s",
        "-v"
    ],
    "python.linting.flake8Enabled": false,
    "python.linting.flake8Args": [
        "--max-line-length=120",
        "--ignore=E731,W503,F401"
    ],
    "python.linting.pycodestyleEnabled": false,
    "python.linting.pycodestyleArgs": [
        "--max-line-length=120"
    ]
  }
```

# Root README 

## Python setup
Set up a virtual environment and install the required dependencies:

```shell
pyenv install 3.14.2
pyenv local 3.14.2
python --version
python -m venv .venv

# macos:
source .venv/bin/activate

# windows:
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\activate.bat

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

For deactivation of the virtual profile, run:
```shell
deactivate
```

Application README:
![README.md](biomed_knowledge_platform/README.md)

BDD Test README:
![README.md](tests/README.md)

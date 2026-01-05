# Root README 

## Local Run
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
The main to run file is:
``/src/app/python/main.py``

To run the tests, open a terminal in the project root and execute:
``pytest``

For deactivation of virtual profile run:
```shell
deactivate
```
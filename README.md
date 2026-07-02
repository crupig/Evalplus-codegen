## PEP notes
This fork was created as a support generation tool for the work "How Should We Rank LLM Code Generations? A Comparative Study of  Selection Strategies". It was modified so that also the log probabilities of the generated tokens are extracted. Also the test scripts were modified accordingly.

**To setup:**
* clone the repo;
* create virtual environment;
* install `requirements.txt` (designed to work on Python `3.12`)

**Replace scripts**:
Replace all the scripts in `scripts_to_replace` in the `.venv/lib/python3.12/site-packages/evalplus` path.

**To run code generation:**

```bash run_gen.sh <DEVICE_ID> <MODEL_PATH>```

For example:

```bash run_gen.sh 0 Qwen/Qwen2.5-Coder-3B-Instruct```

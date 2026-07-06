## Notes for replication of "Comparative Study of Selection Strategies"
This repo was created as a support generation tool for the work "How Should We Rank LLM Code Generations? A Comparative Study of  Selection Strategies". It was modified so that also the log probabilities of the generated tokens are extracted. Also the evaluation scripts were modified to extract the test execution feedbacks.

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

**Evaluation:**

Example of commands to run the scripts (input file and number of workers):

```bash run_eval.sh ./path_to_generation_file.jsonl```

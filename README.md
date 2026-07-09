## Notes for replication of "Comparative Study of Selection Strategies"
This repo was created as a support generation tool for the work **"How Should We Rank LLM Code Generations? A Comparative Study of  Selection Strategies"**. 

The original code generation frameworks have been modified in order to:
* extract the log-probabilities when generating code solutions;
* extracting test execution feedbacks when evaluting the generated solutions;
* generate test cases instead of code solutions (for the CodeT approach);
* run the generated test cases against the previously generated code solutions.
  
**To setup:**
* clone the repo;
* create virtual environment;
* install `requirements.txt` (designed to work on Python `3.12`)

### Code generation:

**Replace scripts**:

Replace all the scripts in `scripts_to_replace` in the `.venv/lib/python3.12/site-packages/evalplus` path.

```bash run_gen.sh <DEVICE_ID> <MODEL_PATH>```

For example:

```bash run_gen.sh 0 Qwen/Qwen2.5-Coder-3B-Instruct```

**Evaluation:**

Example of command to run the scripts (input file as argument):

```bash run_eval.sh ./path_to_generation_file.jsonl```


### Testcases generation (CodeT approach):

**Replace scripts**:

Replace all the scripts in `scripts_to_replace_testcases` in the `.venv/lib/python3.12/site-packages/evalplus` path.

```bash run_gen_tests.sh <DEVICE_ID> <MODEL_PATH>```

For example:

```bash run_gen_tests.sh 0 Qwen/Qwen2.5-Coder-3B-Instruct```

**To execute tests:**

Within each model (generator) and each coding problem (task id), we have to run each generated code solution against each generated test case.

`merge_generations_and_tests_b4_test_exec.py` creates and saves all the `<code_solution, test_statement>` pairs.

After generating all the pairs, the tests can be run with:

```bash run_eval_tests.sh <PATH_TO_TESTS>```

For example:

```bash run_eval_tests.sh ./path_to_generation_file.jsonl```

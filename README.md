# Symbolic Reasoning

This repository reproduces the experiments described in the paper: **"[Symbolic Reasoning in Language Models](https://aclanthology.org/2023.findings-acl.364.pdf)"**. The goal is to evaluate the reasoning capabilities of language models using various prompting templates and assess their performance.

---

## Repository Structure

- **`forward.py`**: Runs experiments using the first prompting template described in the paper.
- **`forwardCOT.py`**: Uses the Chain-of-Thought (CoT) prompting template for experiments.
- **`forwardSPRaw.py`**: Executes experiments with a simplified reasoning template without intermediate steps.
- **`forwardSPRawCOT.py`**: Combines the simplified reasoning approach with CoT prompting.
- **`eval.py`**: Evaluates the outputs of the experiments, calculating metrics like accuracy, BLEU scores, and Levenshtein similarity.
- **`data/`**: Directory to store input data or generated outputs from the experiments.

---

## How to Run the Experiments

To replicate the experiments described in the paper, run each of the provided scripts corresponding to the prompting templates. The outputs will be saved as CSV files.

### Running the Experiments
Run the following commands in your terminal:

1. **Experiment with `forward.py`**:
    ```bash
    python forward.py > data/forward_results.csv
    ```

2. **Experiment with `forwardCOT.py`**:
    ```bash
    python forwardCOT.py > data/forwardCOT_results.csv
    ```

3. **Experiment with `forwardSPRaw.py`**:
    ```bash
    python forwardSPRaw.py > data/forwardSPRaw_results.csv
    ```

4. **Experiment with `forwardSPRawCOT.py`**:
    ```bash
    python forwardSPRawCOT.py > data/forwardSPRawCOT_results.csv
    ```

Make sure the `data/` directory exists to store the output files.

---

## Evaluating the Results

Once the experiments are complete, use the `eval.py` script to evaluate the outputs. The script calculates the following metrics:
- **Accuracy**: Measures the proportion of correct matches.
- **BLEU Score**: Evaluates the similarity between generated and reference responses.
- **Levenshtein Similarity**: Computes the similarity based on edit distance.

### Running the Evaluation
Run the following command to evaluate a specific result file:

```bash
python eval.py

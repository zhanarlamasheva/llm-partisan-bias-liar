# Counterfactual Audit of Partisan Bias in LLM-Based Fact-Checking

Code and results for the paper *"Counterfactual audit of partisan bias in LLM-based fact-checking"* (submitted to the Herald of the Kazakh-British Technical University).

## Overview

The study tests whether open instruction-tuned large language models change their veracity judgments when only the political party attributed to the speaker is changed. Each statement from the LIAR test set is presented in three conditions that differ only in the speaker description:

| Condition | Speaker description |
|---|---|
| Neutral | "A politician" |
| Democrat | "A politician from the Democratic Party" |
| Republican | "A politician from the Republican Party" |

The speaker's name, job, state and context are withheld. Statements whose text explicitly mentions a party are excluded (1,283 → 1,208 statements).

Models: Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct, Gemma-2-9B-it (4-bit NF4, greedy decoding).

Instead of parsing generated text, the notebook reads the model's probability distribution over the six ordinal LIAR labels (0 = pants on fire … 5 = true) and computes the expected truthfulness score for every statement. Bias is measured with paired comparisons: mean shift with bootstrap 95% CI, Cohen's dz, flip rate, Wilcoxon signed-rank test with Bonferroni correction.

## Repository structure

```
├── liar_party_bias.ipynb     # full experiment (Google Colab)
├── results/
│   ├── final_table.csv       # paired comparisons, prompt P1 (CI, dz, flip rate, p)
│   ├── robustness_table.csv  # Democrat − Republican shift under prompts P1–P3
│   ├── method_check.csv      # original vs fixed-prefix scoring
│   └── fig1.png – fig3.png   # figures from the paper
└── README.md
```

## Notebook sections

1. Data loading and filtering (LIAR test split)
2. Counterfactual conditions and prompt
3. Model loading and probability scoring
4. Main run (prompt P1)
5. Analysis: classification quality, paired comparisons, subgroup by actual party
6. Robustness: paraphrased prompt (P2) and reversed rating scale (P3)
7. Measurement validity check: fixed-prefix scoring (`Rating:` + log-probabilities of the six answers)

## How to run

1. Open `liar_party_bias.ipynb` in Google Colab and select a GPU runtime (*Runtime → Change runtime type → T4 GPU*).
2. Create a Hugging Face access token (https://huggingface.co/settings/tokens) and accept the licences of the gated models on their Hugging Face pages (Llama 3.1 and Gemma 2).
3. Run the cells in order. Results are saved to Google Drive after each model and condition, so an interrupted session can be resumed.

A free Colab T4 session is sufficient; the full run takes several hours. For a quick test, set `N_LIMIT = 50` in Section 1.

## Data

The LIAR dataset (Wang, 2017) is downloaded automatically from the authors' website. It is not redistributed in this repository.

## Citation

If you use this code, please cite the paper:

```
[Authors] (2026). Counterfactual audit of partisan bias in LLM-based fact-checking.
Herald of the Kazakh-British Technical University. [volume(issue), pages, DOI]
```

## License

MIT License (see `LICENSE`).

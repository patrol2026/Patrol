# PATROL: Defending LLMs Against Jailbreak Attacks via Periodic Adversarial Token Rollback
This repository contains the official implementation and reproduction scripts
for **PATROL**, a training-free decoding-time defence against jailbreak attacks
on autoregressive LLMs.

> **Paper:** _PATROL: Defending LLMs Against Jailbreak Attacks via Periodic Adversarial Token Rollback_ (NeurIPS 2026 submission, anonymous)
> **Anonymous code:** this repository

PATROL wraps any autoregressive LLM served through a decoding-loop interface
(e.g. Hugging Face `transformers`, vLLM, TGI). During generation, it
periodically queries a safety classifier on the partial response; on an unsafe
verdict it backtracks the decoder to a recent state and resamples under a
position-wise blocklist that prevents re-entry into the same harmful trajectory.

---

## Repository layout

```
patrol_release/
├── README.md                           # this file
├── LICENSE                             # MIT
├── requirements.txt                    # Python dependencies
├── src/                                # core implementation + analysis
│   ├── defense_patrol.py            # PATROL + baseline defenders
│   ├── safety_constrainers.py          # safety classifier wrappers
│   ├── run_patrol_sweep.py          # hyperparameter sweep
│   ├── judge_all.py                    # GPT-4.1 ASR judge
│   ├── judge_alpaca_eval.py            # AlpacaEval pairwise judge
│   ├── run_alpaca_eval_official.py     # AlpacaEval 2.0 protocol
│   ├── compute_early_commitment.py     # detection-CDF analysis
│   ├── plot_early_commitment.py        # Figure 2 (per-prefix detection)
│   ├── plot_ablations.py               # 
├── scripts/                            # end-to-end reproduction scripts
│   ├── run_fill_gaps.sh                # generate any missing baseline cells
│   ├── run_ablations_mct_bl_ci.sh      # hyperparameter ablations
│   └── run_patrol_multiseed.sh         # multi-seed RandomSearch evaluation
├── configs/                            # custom configurations
│   └── weighted_alpaca_eval_gpt41/     # AlpacaEval 2.0 with GPT-4.1 judge
│       ├── configs.yaml
│       └── alpaca_eval_clf.txt
└── examples/                           # placeholder for sample outputs
```

---

## 1. Installation

```bash
git clone <ANONYMIZED-REPO-URL>
cd patrol_release
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

PATROL's runtime safety classifier is **LlamaGuard-2-8B**, served as a
standard Hugging Face model. The first invocation downloads the weights from
the Hugging Face Hub; ensure you have a token with access to
`meta-llama/Meta-Llama-Guard-2-8B`.

To use the GPT-4.1 ASR judge or the AlpacaEval 2.0 win-rate judge, export an
OpenAI key:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## 2. External assets you must obtain separately

These artefacts are not redistributed in this repository.

| Asset | Source | Used for |
|---|---|---|
| `meta-llama/Llama-2-7b-chat-hf` | Hugging Face | target model |
| `meta-llama/Meta-Llama-3-8B-Instruct` | Hugging Face | target model |
| `lmsys/vicuna-7b-v1.5` | Hugging Face | target model |
| `meta-llama/Meta-Llama-Guard-2-8B` | Hugging Face | runtime classifier $\phi$ |
| AdvBench harmful behaviours | https://github.com/llm-attacks/llm-attacks | direct-prompt attack |
| GCG attack artefacts | https://github.com/llm-attacks/llm-attacks | white-box gradient attack |
| AutoDAN | https://github.com/SheltonLiu-N/AutoDAN | template-evolution attack |
| PAIR | https://github.com/patrickrchao/JailbreakingLLMs | LLM-as-attacker |
| DeepInception | https://github.com/tmlr-group/DeepInception | nested-fictional-frame attack |
| RandomSearch | https://github.com/tml-epfl/llm-adaptive-attacks | adaptive black-box attack |
| AlpacaEval | https://github.com/tatsu-lab/alpaca_eval | utility benchmark |
| SafeDecoding LoRA experts (LLaMA-2, Vicuna) | https://github.com/uw-nsl/SafeDecoding | optional baseline only |

After cloning each attack repository, place its prompt artefacts under
`./datasets/promptrs/` (RandomSearch) and `./datasets/<attack>/` (others) so
that the scripts in `src/` can locate them. The exact file names expected by
`defense_patrol.py` are documented at the top of that file.

---

## 3. Reproducing the paper

All commands below assume `pwd` is `patrol_release/` and that the external
assets above have been placed alongside this repository (see
`defense_patrol.py:_PROMPTRS_DIR` for the expected layout).

### 3.1 Main results (Table 2)

Run PATROL and the seven baseline defences against the six jailbreak attacks on
each of the three target models. The default PATROL configuration is
$(c{=}4, \ell{=}20, m{=}128)$ throughout.

```bash
# Generate PATROL + baseline defender outputs
bash scripts/run_fill_gaps.sh

# Judge ASR with GPT-4.1
python src/judge_all.py \
    --judge_backend openai \
    --openai_model gpt-4.1 \
    --include_dirs results \
    --model_filter llama2 llama3 vicuna \
    --no_strict_filter \
    --spreadsheet results_summary_gpt41.xlsx

```

### 3.2 Utility (AlpacaEval 2.0)

The official AlpacaEval 2.0 protocol is supported via a custom annotator
configuration that uses GPT-4.1 instead of GPT-4-Turbo.

```bash
# Make the custom annotator visible to alpaca_eval
ln -s "$(pwd)/configs/weighted_alpaca_eval_gpt41" \
      "$(python -c 'import alpaca_eval, os; print(os.path.dirname(alpaca_eval.__file__))')/evaluators_configs/weighted_alpaca_eval_gpt41"

# Compute length-controlled win rate for every defender
python src/run_alpaca_eval_official.py \
    --output_dir results_alpaca_eval_official \
    --annotators_config weighted_alpaca_eval_gpt41 \
    --skip_existing \
    --input "results/*alpaca_eval*.json"
```

### 3.3 Ablations (Section 4 figures)

```bash
# Sweep c, ℓ, m on RandomSearch for all three target models
bash scripts/run_ablations_mct_bl_ci.sh

# Render the three ablation figures
python src/plot_ablations.py
```

### 3.4 Early-commitment validation (Figure 2)

The empirical evidence for the early-commitment hypothesis. Reuses the
no-defence generations produced in §3.1 and runs LlamaGuard-2 on
prefix slices.

```bash
python src/compute_early_commitment.py
python src/plot_early_commitment.py
# Per-model breakouts for the appendix
python src/plot_early_commitment.py --models llama2 --out figures/ec_llama2.pdf
python src/plot_early_commitment.py --models llama3 --out figures/ec_llama3.pdf
python src/plot_early_commitment.py --models vicuna --out figures/ec_vicuna.pdf
```



## 4. Default hyperparameters

PATROL's three runtime hyperparameters are set to the values selected by the
ablation in Section 4 of the paper:

| symbol | meaning | default |
|---|---|---|
| `c` | classifier check interval (tokens) | 4 |
| `ℓ` | backtrack length on an unsafe verdict (tokens) | 20 |
| `m` | safety horizon (max prefix length checked) | 128 |

These values are the defaults of `defense_patrol.py` and are referenced
directly throughout the reproduction scripts.

---

## 5. Hardware

All experiments in the paper were conducted on a single
NVIDIA RTX 6000 Ada Generation GPU (48 GB VRAM, CUDA 12.8). The same GPU
hosts both the target language model and LlamaGuard-2-8B during PATROL
decoding. 

---



## 7. License

MIT (see [LICENSE](LICENSE)). Note that several external attack and benchmark
artefacts referenced from §2 are released under their own respective licenses.

---

## 8. Citation

```bibtex
@inproceedings{patrol2026,
  title     = {PATROL: Defending LLMs Against Jailbreak Attacks
via Periodic Safety Monitoring and Token Rollback},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```



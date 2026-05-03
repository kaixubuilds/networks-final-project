# Paper Figures

Each figure communicates exactly one idea.

---

## Figure 1 — The Thundering Herd Is Real

**One idea:** Routing to the shortest *reported* queue causes one worker to absorb all load while the others idle, in a repeating cycle.

**Type:** Line plot  
**X-axis:** Simulation time (ms), a window of ~300 ms  
**Y-axis:** True queue length  
**Lines:** One per worker (8 lines), under JSQ-stale only  

The lines should show one or two workers repeatedly spiking to 20–50 jobs while the others sit at zero, then collapsing as the heartbeat catches up and the herd moves on. This is the thundering herd mechanism made concrete. No summary statistic captures it; you have to see it.

---

## Figure 2 — The Baseline Ladder

**One idea:** Random routing is a surprisingly strong baseline; avoiding stale-JSQ is more important than any other design choice.

**Type:** Violin or box plot, horizontal  
**One row per strategy:** Random, Po2C (stale), JSQ (stale)  
**X-axis:** Mean σ across 100 seeds  

The three distributions are tight and clearly separated: JSQ-stale ≈ 15, Po2C ≈ 4.4, Random ≈ 3.8. This establishes the reference scale for every LLM result. The counterintuitive point — that the naive, information-using strategy (JSQ-stale) is *worse* than the information-ignoring strategy (Random) — should be immediately visible.

---

## Figure 3 — No Model Reliably Beats Random

**One idea:** Across all models tested, the median LLM trial performs worse than flipping a coin.

**Type:** Dot plot or lollipop  
**X-axis:** Median mean σ across valid trials  
**Y-axis:** Model (ordered by parameter count within family)  
**Reference lines:** Random (≈ 3.77), Po2C (≈ 4.40), JSQ-stale (≈ 14.98)  

Every model's median sits between Po2C and JSQ-stale — worse than Random. The reference lines make this immediately legible. This is the headline result. No error bars needed here; this is about medians, not means, and the medians are far from the Random line.

---

## Figure 4 — The Mean Is Misleading: Distributions Are Heavy-Tailed

**One idea:** Most models have bimodal or catastrophically heavy-tailed distributions — a small number of trials saturate the system, dominating the mean.

**Type:** Violin plot (or horizontal strip plot for small-n models)  
**Y-axis:** Per-trial mean σ, log scale  
**X-axis:** Model, ordered by parameter count  
**Overlay:** Horizontal reference lines for Random and JSQ-stale  
**Violin width:** Proportional to n_valid (so parse failure is visible as narrowness)  

The key contrasts: Qwen3-0.6B's median is near 14,000 (system saturation), but its best trial is 37 — the distribution is split between near-collapse and merely bad. Qwen3-1.7B is similar. Qwen3-8B and 14B are tightly clustered around 25–27. gpt-oss-20b has only 10 valid trials but they split between ≈3 and ≈21 — a bimodal distribution of its own. The log scale is essential.

---

## Figure 5 — Generating Valid Code Was Itself Unreliable

**One idea:** Parse success rates range from 0% to 84% with no monotonic relationship to model size or quality.

**Type:** Bar chart  
**X-axis:** Parse success rate (0 to 1)  
**Y-axis:** Model (same order as other figures)  

Values: Llama-3.1-8B: 0%, gpt-oss-20b: 20%, Qwen3-8B: 43%, DeepSeek-R1: 45%, Qwen3-0.6B: 50%, Qwen3-4B: 55%, Qwen3-14B: 68%, Qwen3-1.7B: 84%. Llama-3.1-8B's 0% is an infrastructure failure (missing tokenizer chat template), not a reasoning failure — mark it distinctly (e.g., hatched bar or footnote) so it is not compared directly. Among the remaining models: the best performer (gpt-oss-20b) has the lowest parse rate; the highest parse rate (Qwen3-1.7B) produces near-catastrophic load balancing. Instruction-following and load-balancing reasoning are completely dissociated.

---

## Figure 6 — The Correct Strategy Is Simple and Achievable

**One idea:** When models do get it right, they converge on the same idea: subtract the age as an estimate of completions, then take the minimum.

**Type:** Annotated code figure (not a data plot)  
**Content:** Two full code blocks side by side, one from each model, labeled with model name and per-trial mean σ:

- gpt-oss-20b trial 53, σ = 2.87
- Qwen3-14B trial 84, σ = 2.98

Showing one example per model (not multiple from gpt-oss-20b) emphasizes that two entirely different models, from different families, independently arrived at the same solution. Both implement `adj = max(q - age, 0)` and return `argmin(adj)` with random tie-breaking. These two trials, along with two other gpt-oss-20b trials (σ = 2.94, 2.99) that follow the same pattern, are the only ones across all 800 total trials that reach or beat Random. The figure's message is that the correct strategy is elementary once you reason about what age *means* — but most models did not.

---

## Figure 7 — The Sign of Age-Weighting Determines Everything

**One idea:** Adding age (wrong direction) is as catastrophic as ignoring age entirely; only subtracting it produces competitive performance.

**Type:** Horizontal strip plot (individual dots + median line per class)  
**X-axis:** Mean σ, log scale  
**Y-axis:** Strategy class  

Four classes derived from inspecting the generated code across all valid trials:
- **Subtract age** (`adj = max(q - age, 0)`): σ ≈ 2.9–3.0 — only ~4 trials; show as individual dots, not a box
- **Add age with positive weight** (`score = q + α·age, α > 0`): σ ≈ 18–36 (most Qwen3-8B, Qwen3-14B, gpt-oss-20b)
- **Ignore age** (pure JSQ on stale queue lengths): σ ≈ 15–30
- **Wrong direction / degenerate** (route to max, sort by index, etc.): σ ≈ 100–14,000

Use a strip plot uniformly (individual dots + median line) rather than box plots, since class sizes are unequal and "subtract age" has too few points for a box to be meaningful. Reference lines for Random and JSQ-stale. The visual message: adding age and ignoring age both produce indistinguishable mediocre performance. Only subtraction achieves good performance, and it achieves very good performance. The reasoning error — treating age as a penalty on *fresh* workers rather than a correction for *stale* information — is what separates the failed majority from the successful minority.

---

## Figure 8 — Model Family Predicts Performance Better Than Parameter Count

**One idea:** gpt-oss-20b (2B parameters) outperforms Qwen3-14B (15B parameters); scale within a family shows no consistent trend.

**Type:** Scatter plot  
**X-axis:** Parameter count, log scale  
**Y-axis:** Median mean σ across valid trials, log scale  
**Points:** One per model, shaped/colored by model family (Qwen3 vs. other)  
**Reference lines:** Random and JSQ-stale  

Within the Qwen3 family: 0.6B and 1.7B are catastrophic (median σ near saturation), 4B is slightly less catastrophic, 8B and 14B cluster together around σ ≈ 25–27. No clean scaling law. Note: gpt-oss-20b reports ~2B parameters in the experiment metadata but is named "20B" — its true parameter count is uncertain. Plot it at the reported value (~2B) and mark it with a distinct symbol and a note. Regardless of whether it is 2B or 20B, it sits well below all Qwen3 models, so the "family matters" conclusion holds; the ambiguity only affects how dramatic the gap looks on the x-axis. DeepSeek-R1 (8B-class) has a median similar to Qwen3-8B despite being a reasoning-distilled model. The figure resists the tempting conclusion that "bigger is better."

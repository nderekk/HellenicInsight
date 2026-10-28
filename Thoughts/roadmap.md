# HellenicInsight — Roadmap From Here

October 25, 2026 · builds on info.txt (product idea) and research.md (labelling strategy)

## How to read this document

Every step follows the same template:

- **Goal**: what the step produces.
- **Why**: the problem it solves, and what goes wrong if you skip it.
- **How**: concrete actions, in order.
- **Read**: papers/resources worth your time for this step (and what to look for in them).
- **Done when**: a checkable exit criterion. Don't move on until it is met.

The work splits into two tracks that can run in parallel with two sub-teams:

- **Track A — Labels & Models** (Phases 0–5): make the bias scores trustworthy.
- **Track B — Events & Product** (Phases 6–9): build the actual news-aggregator experience.

They meet in Phase 10 (beta launch). Rough timeline for a small student team: Track A ≈ 10–12 weeks, Track B ≈ 8–10 weeks, beta around week 12–14.

```
Week:     1  2  3  4  5  6  7  8  9  10 11 12 13 14
Track A:  [P0][--P1--][----P2----][---P3---][P4][P5]
  P5 prep:      [--training code + model screening on weak labels--]
Track B:     [-----P6-----][---P7---][--P8--][P9]
Both:                                          [P10 beta]
```

---

## Where we actually stand (honest audit)

What exists today:

- A scraping → Kafka → Spark → vLLM → MongoDB → React pipeline that worked at the hackathon. The inference step has **no model behind it any more** (see below).
- `old-dep/datasets/labeled_dataset.jsonl`: 3,000 articles with one continuous `ai_labels.bias` score from one LLM (Gemini). Source split: **efsyn 1,443 / protothema 1,438 / iefimerida 119**.
- Synthetic `right` / `far_right` data, created to fix the base model's habit of labelling right-wing articles as "neutral".
- `old-dep/1st_training_dump/nemo_sft_*.jsonl`: the same 3,000 articles formatted as SFT prompts, including the old prompt's anchor definitions. These are reusable.
- Metrics showing a ~10% F1 gain after fine-tuning. Those metrics are measured against the *same LLM's* labels, not against humans.

What does **not** exist any more: **the fine-tuned model.** The only trace is `qwen_greek_lora_final.tar.gz`, which holds a single NeMo checkpoint (`qwen-nemo-lora-2gpu-optimized.nemo`). It only works with the exact Qwen base checkpoint and NeMo version used on the hackathon's B200s, and neither is recorded. **Treat model training as starting from zero.** That is less bad than it sounds: the old model was trained on labels we're replacing anyway (single LLM, no human check, outlet-confounded), so it would have been retrained regardless. Phase 5 is the full model decision, and Steps 0.1 and 5.6 make sure a model can't be lost again.

The three structural weaknesses this roadmap fixes:

1. **Source confound.** With two outlets from opposite sides making up ~96% of the data, "efsyn vs protothema" and "left vs right" are almost the same variable. Our model may be detecting the outlet (vocabulary, house style, recurring columnists) rather than the article's stance. The F1 number can't tell these apart.
2. **No ground truth.** Every number we have is "agreement with Gemini". We don't know how Gemini compares with Greek human judgement.
3. **No event clustering.** The core product feature (one event, many sources) doesn't exist yet. `info.txt` literally says "HOW".

The synthetic data is a fourth, smaller problem: LLM-written "right-wing articles" tend to be caricatures. They can teach the model stereotypes ("mentions migration → right") rather than real framing. Keep them out of evaluation entirely, and re-check whether they still help once real data exists.

---

# TRACK A — Labels & Models

## Phase 0 — Salvage what survives and measure the confound (week 1)

### Step 0.1 — Salvage what survives, and set up artifact storage

**Goal:** everything reusable from the hackathon (data, prompts, metrics, lessons) saved in one documented place, plus a storage setup that makes losing a model impossible from now on.

**Why:** the model is gone because it lived on one machine, in one framework-specific format, with no record of its base checkpoint. The data and prompts are still valuable. The 3,000 Gemini-labelled articles become **weak labels** for building and debugging the training code (Phase 5) long before the real labels exist. The old prompt's anchor definitions are a starting point for the codebook (Phase 1).

**How:**
- `git tag v0-hackathon`. Create a **private Hugging Face organisation** for the team (free). Upload `labeled_dataset.jsonl`, the `nemo_sft_*` files and the synthetic sets as a versioned private dataset. Large binaries don't belong in git.
- Write `DATA_CARD_v0.md`: where the articles came from, the label source (Gemini, which version, which prompt — recover the prompt from `nemo_sft_train.jsonl`), the source imbalance, and the metrics from `old-dep/model_metrics/`, with a note that they measure agreement with Gemini, not with humans.
- Write a one-paragraph post-mortem on why the model was lost, and adopt the rule from Step 5.6: **every trained checkpoint goes to the HF org in standard safetensors/PEFT format, with its base-model ID, config and training-data version, before it is used anywhere.**

**Done when:** the data is on the HF org, the data card is in the repo, and the team has agreed on the artifact rule.

### Step 0.2 — The "is it just predicting the outlet?" probe

**Goal:** numbers that tell us how much any model trained on the existing data would learn the outlet instead of the stance. We don't need the old model for this: the problem is in the *data*, so we test the data.

**Why:** if a model is mostly an outlet detector, it will fail exactly where the product needs it. It would give every efsyn article "left" and every protothema article "right", including protothema's neutral wire reports and efsyn's occasional centrist piece. It would also have nothing reliable to say about an outlet it never saw. This is the CNR failure described in research.md §2.

**How (three cheap experiments):**
1. **Outlet classifier.** Train a TF-IDF + logistic regression to predict `source` from `text`. If accuracy is ~95%+, the outlet is trivially recoverable from the text, so any model *can* take that shortcut.
2. **Within-outlet variance.** Plot the distribution of `ai_labels.bias` per outlet. If efsyn is all 0.2–0.35 and protothema all 0.6–0.75, the labels themselves barely vary inside an outlet. In that case the labeller may also have leaned on outlet cues (research.md §2: LLMs recognise outlets from names and style).
3. **Leave-one-outlet-out test.** Train a TF-IDF + ridge regression on the Gemini scores using only efsyn + protothema. Test it on iefimerida and on ~100 articles each from 2–3 new outlets (Step 2.1), labelled with the same Gemini prompt. If performance collapses on unseen outlets, the data teaches outlet memorisation, which is the lesson for whatever model we pick in Phase 5. This model is also **rung 0** of the Phase 5 model ladder, so the work isn't wasted.

**Read:**
- "Media Source Matters More than Content" (EMNLP 2025), cited in research.md. Read the method section to see how they separated source effects from content effects.
- Geirhos et al., 2020, "Shortcut Learning in Deep Neural Networks" (Nature Machine Intelligence). This is the general idea behind the problem: models pick the easiest predictive signal, not the one you intended. Short and very readable.

**Done when:** you have three numbers (outlet-classifier accuracy, within-outlet label spread, unseen-outlet performance) written into the v0 data card. These numbers are also good material for the next pitch: "we found and fixed a flaw most bias detectors have".

---

## Phase 1 — Define what "bias" means for Greek news (weeks 1–3)

### Step 1.1 — Read the political-science foundations

**Goal:** the team shares a vocabulary for "political position", based on how political scientists measure it.

**Why:** "0 = far-left, 1 = far-right" sounds simple, but Greek politics breaks a single axis in several known ways (research.md §1). KKE is economically far-left but socially conservative on some issues. ANEL governed with SYRIZA. Anti-memorandum populism crossed left and right. Being for or against the current government cuts across all of these. If the codebook doesn't decide these cases explicitly, each annotator, and each LLM, will decide them differently. The disagreement will then look like noise when it is really a definition problem.

**How / Read (in this order, ~1 week total):**
1. **Chapel Hill Expert Survey (CHES) codebook** (chesdata.eu). Look at how it separates *economic left–right* from *GAL–TAN* (Green/Alternative/Libertarian vs Traditional/Authoritarian/Nationalist), and find the Greek parties' scores. This is the most practical template for a two-axis scheme, and the party scores give you external anchors.
2. **Benoit & Laver / Gemenis on RILE and Greece** (cited in research.md). Read only the Greek case to understand *why* a single axis puts KKE on the right. It's the best argument for two axes.
3. **Manifesto Project coding handbook** (manifesto-project.wzb.eu). Skim the category list (e.g. per401 "free market", per601 "national way of life"). The categories are a ready-made list of "what counts as a left/right signal".
4. **AgoraSpeech paper** (Greek 2023 election speeches). See how a Greek team defined polarization and populism labels; you can borrow their definitions.
5. **Fan et al., 2019, "In Plain Sight: Media Bias Through the Lens of Factual Reporting" (BASIL dataset, EMNLP).** Key idea: separate **lexical bias** (loaded words: "regime", "thugs") from **informational bias** (which facts are included or left out). Most bias in mainstream news is informational, and a sentence-level word model misses it. This matters for both the codebook and Phase 8.
6. **Card et al., 2015, "The Media Frames Corpus".** Read about framing dimensions (economic, morality, security, …). These are useful for the "how each side frames this event" feature later.

**Done when:** each team member can explain why we need more than one axis, using a Greek example.

### Step 1.2 — Write the codebook v1

**Goal:** a written document (Greek and English, 5–10 pages) that any trained annotator, or any LLM prompt, can follow to label an article.

**Why:** in a subjective task, "correct" can only mean "what trained coders following this codebook would agree on". Without a codebook there is nothing to agree on, and human-vs-LLM comparisons mean nothing.

**How — recommended label schema:**

| Field | Values | Why it exists |
| --- | --- | --- |
| `is_political` | yes / no | Sports, weather and celebrity news shouldn't get a stance score at all. |
| `stance_detectable` | yes / no | Neutral wire copy (ΑΠΕ-ΜΠΕ reposts) is political but has no stance. Without this gate it gets forced to 0.5 and fills the centre with noise. |
| `econ_lr` | 0–1 (or 1–7 Likert) | State vs market, redistribution, labour, privatisation. |
| `cultural_gal_tan` | 0–1 | Migration, nationalism, church, LGBTQ+, law & order. |
| `gov_stance` | −1 (hostile) … +1 (supportive) | In Greece this is often *the* most visible dimension, and readers feel it directly. It's also the easiest for humans to agree on. |
| `confidence` | low / med / high | Lets us down-weight hard items later. |
| `evidence` | short free text / highlighted spans | Forces the annotator (or LLM) to justify the label, and gives explanation data for free. |

For each field, write:
- a definition,
- 2–3 **anchor examples** per scale point (real Greek articles, with outlet names removed), and
- **decision rules for hard cases**: opinion piece vs news report; an article that quotes both sides; an article criticising the government from the right (e.g. on migration) vs from the left; KKE-style positions; foreign news (Trump, Israel/Gaza, Ukraine) with a Greek angle.

Leave "overall bias 0–1" out of the stored labels. It can be *derived* for the UI (e.g. a weighted combination, or just showing two bars), but storing the components keeps the information.

**Done when:** codebook v1 exists, and a pilot (Step 2.3) has been run on it.

---

## Phase 2 — Data: more outlets, cleaner text, a human gold set (weeks 2–6)

### Step 2.1 — Expand from 3 to ~10–15 outlets across the spectrum

**Goal:** a corpus where each outlet has hundreds of articles, the outlets cover the whole spectrum, and *within* each part of the spectrum there are several outlets.

**Why:** this is the real fix for the source confound. If the "left" side contains efsyn, avgi, documento and kommon, the model can't succeed by memorising one outlet's style. It has to learn what they have in common, which is the stance. Several outlets also allow a **leave-outlet-out** evaluation, the honest test of generalisation.

**How:**
- Pick outlets to roughly fill this grid (examples; verify each outlet's current line yourselves):
  - Left / anti-government: efsyn, avgi, documento, kommon/902 (KKE)
  - Centre / public / wire: ertnews, naftemporiki, in.gr, kathimerini (centre-right), capital.gr
  - Right / pro-government: protothema, iefimerida, liberal.gr, newsbomb
  - Nationalist / far-right: pick 1–2 carefully. Check that they're actual news sites and check their terms.
- Check `robots.txt` and terms for each site. Rate-limit (≤1 request per few seconds per domain). Identify your crawler with a contact address in the User-Agent. Prefer RSS feeds and sitemaps over crawling, because they're cheaper and less likely to get you blocked.
- Use **trafilatura** (Python) for main-text extraction, instead of or alongside the current regex `purify_markdown`. It's built for exactly this, handles many layouts, and gives cleaner text with less maintenance.
- **Strip outlet fingerprints** from the text the models see: outlet names, bylines, "Διαβάστε επίσης", newsletter prompts, social embeds, recurring signatures. Keep the original separately for display.
- **Deduplicate** near-identical wire stories with MinHash (`datasketch` library). Otherwise one ΑΠΕ report reposted on 8 sites counts 8 times.
- Store `published_at`, `section/category` and `is_opinion` (from URL patterns like `/opinions/`, `/apopseis/`). Opinion and news need different treatment.
- Target: ≥300 political articles per outlet over at least 3 months (so the corpus isn't dominated by one news cycle).

**Read:** the trafilatura docs and the original paper (Barbaresi, 2021, ACL demo). It's short and explains why boilerplate removal is harder than it looks.

**Done when:** 10+ outlets, deduplicated, fingerprint-stripped, with the per-outlet count table in the repo.

### Step 2.2 — Choose the gold-set sample (stratified)

**Goal:** a list of ~500 articles to be human-labelled.

**Why:** a random sample would be ~70% non-political or neutral, so you'd spend human time confirming "no stance" while the extremes, which are the hard and important cases, are under-sampled. Stratifying ensures every region of the scale, every outlet and every topic is measured.

**How:**
- Stratify by **outlet × topic × month**, so each cell gets a few articles.
- Oversample likely-extreme articles using the existing Gemini scores, or the Step 0.2 TF-IDF baseline for new outlets (a few hundred from the tails). Keep a purely random slice of ~100 too, so you can estimate how the real distribution looks.
- Split once, permanently: **dev ≈ 200** (used to choose prompts and models) and **test ≈ 300** (touched only to grade final systems; nobody looks at it while iterating). Also hold out 1–2 whole outlets as a "never seen" test.
- Why ~300 for the test set: at a model–human correlation around 0.7, the standard error is about (1 − r²)/√n ≈ 0.03 with n = 300. That's small enough to tell apart systems that differ by ~0.05–0.1, which is the size of improvement you'll care about.

**Done when:** the dev/test ID lists are committed, and the test list is treated as read-only.

### Step 2.3 — Recruit annotators, pilot, then label

**Goal:** ~500 articles each labelled by 3 Greek annotators, with a measured inter-annotator agreement.

**Why:** human agreement is the realistic ceiling for any model. If humans agree with each other at α = 0.65, a model reaching α = 0.63 against them is excellent, and chasing 0.9 would be pointless. Choosing annotators with **different political leanings** stops the gold set itself from carrying one side's view.

**How:**
1. Recruit 3–5 annotators (fellow students, ideally from political science or journalism; pay them or give credit). Record each person's self-placed political lean, privately, so you can check later whether labels follow the annotator's politics.
2. Tooling: **Argilla** (free, can run on HF Spaces) or **Label Studio**. Show text without outlet name or byline.
3. **Pilot**: all annotators label the same 40 articles. Compute agreement and discuss every disagreement in a meeting. Most disagreements will show a gap in the codebook, so fix the codebook (v1.1) and repeat once if needed. This loop is the single most valuable hour of the whole labelling project.
4. Full round: 3 labels per article. Use the median (or the mean of the continuous scores) as the gold value, and keep all three labels plus the disagreement.
5. Compute **Krippendorff's α** per field (use the `krippendorff` Python package with the interval metric for scales and nominal for yes/no). Rough interpretation: ≥ 0.8 reliable, 0.667–0.8 acceptable for tentative conclusions. Expect `gov_stance` to come out higher than `cultural_gal_tan`.

**Read:**
- Krippendorff, "Computing Krippendorff's Alpha-Reliability" (a free short PDF by the author). Read it to understand *why* α is used instead of percent agreement: it corrects for chance and handles missing labels and continuous scales.
- "LLMs as a substitute for human experts in annotating political text" (Research & Politics), cited in research.md. Read its annotation design and how it handled disagreements.

**Done when:** the gold set is labelled, α is computed per field, and α is reported in the repo. It is your ceiling number.

---

## Phase 3 — LLM annotation at scale (weeks 5–9)

### Step 3.1 — Annotator bake-off on the dev set

**Goal:** a ranked list of (model × prompting method) pairs by agreement with the human dev set.

**Why:** research.md §4: the right test is measured agreement with *our* Greek gold set. Whether a model is "Greek-fine-tuned" or "frontier" doesn't decide it. This step replaces opinions with numbers.

**How:**
- Candidates (≥3 model families, so their biases don't all point the same way):
  - 1–2 frontier API models (e.g. Claude, GPT, Gemini; one of each family if budget allows)
  - **Llama-Krikri-8B-Instruct** (ILSP), local, Greek-native
  - **Meltemi** (ILSP), an older Greek LLM, as a further point of comparison
  - Anything relevant in the **KIEFERSA Hugging Face collection** mentioned in info.txt (good politically for the Kiefer relationship; it gets judged by the same numbers)
  - the **old hackathon Gemini prompt** (recovered in Step 0.1), as the baseline: "how good were our hackathon labels, actually?"
- Methods, per model:
  - **(a) Direct scoring**: "rate this on each axis using the codebook". Cheap, one call per article.
  - **(b) Pairwise + Bradley–Terry**: "which of these two is further right on axis X?" (Step 3.2)
  - **(c) CGCoT + pairwise**: first extract a structured "concept breakdown" per article, then compare breakdowns (Step 3.2)
- Metrics on dev: Spearman correlation with the human mean; MAE; **error by region** (far-left / centre / far-right separately; research.md §5); accuracy on `stance_detectable`.

**Done when:** a table of models × methods × metrics exists. Choose the top 2–3 *different-family* combos as the production annotator ensemble.

### Step 3.2 — Implement pairwise comparison + Bradley–Terry scoring

**Goal:** continuous scores per article per axis, derived from many "A vs B" judgments.

**Why it works better than "give a number 0–1":** a model asked for a number has no stable internal scale. Its "0.6" today may not equal its "0.6" on another article, and scores cluster on round values (0.5, 0.7). Comparing two texts is a simpler, more natural judgment for both humans and LLMs. The **Bradley–Terry (BT) model** then turns win/loss outcomes into a single consistent scale. It's the same mathematics as chess Elo ratings.

**The idea in one formula:** each article *i* has a latent position θᵢ. The probability that "A is further right than B" is σ(θ_A − θ_B), where σ is the logistic function. Given thousands of comparison outcomes, you find the θ values that make the observed outcomes most likely (maximum likelihood). Libraries: `choix` (Python, small and simple) or a 10-line PyTorch/`scipy.optimize` fit. The fit also gives an uncertainty per article, which you can later use as a training weight.

**How:**
1. **Concept breakdown (CGCoT step, once per article):** prompt the LLM to output JSON: actors supported/attacked, policies supported/opposed, framing used, loaded terms, omitted counter-perspectives. This costs N calls in total.
2. **Pairing:** for each article, sample ~10 opponents (random) + 3–5 fixed **anchor articles** (gold-set items with known human scores spread across the scale). Anchors fix the BT scale so scores are comparable across labelling batches.
3. **Compare** breakdowns, not full texts. They're short, so this is much cheaper and less sensitive to article length (long articles were the weak spot in the Research & Politics study).
4. **Run each pair in both orders (A,B) and (B,A).** LLMs have position bias. If the answer doesn't flip when the order flips, treat it as a tie or low-confidence (arXiv 2510.22219).
5. Fit BT per axis, then rescale to [0, 1] using the anchors (e.g. map anchors' θ to their human scores with a monotone fit).

**Cost sanity check:** N = 5,000 articles × ~12 comparisons each ≈ 30k unique pairs. Both orders give 60k calls per model, plus 5k breakdown calls. Because the comparisons are on short breakdowns, this is affordable on a mid-tier API model or a local Krikri on one GPU. Do the bake-off on the dev set first, so you only pay for the full run with methods that proved better.

**Read:**
- Wu et al., "Concept-Guided Chain-of-Thought Prompting for Pairwise Comparison Scaling of Texts with LLMs" (CGCoT). Read the prompt design, then copy and adapt the structure.
- PairScale (NAACL Findings 2025). Read how they extract attitude evidence before comparing.
- arXiv 2506.06540 (pairwise scaling) and arXiv 2510.22219 (errors of LLM pairwise comparison). Read the failure modes so you know which checks to run.
- Bradley & Terry, 1952 (the original, for the math), or the `choix` docs for a practical intro.

**Done when:** BT scores exist for the dev set and beat direct scoring on dev-set correlation. If they *don't*, keep direct scoring and save the money: the gold set decides, not this document.

### Step 3.3 — Measure and control the annotators' own bias

**Goal:** evidence of whether the LLM annotators are politically skewed, and a correction if they are.

**Why:** research.md §5. LLMs rely on party cues and some have a measurable left-liberal lean. A product that claims to measure bias can't have an unmeasured bias in its own measuring device. That would be the first thing a critic or journalist checks.

**How:**
- **Counterfactual swaps:** take 100 dev articles. Create variants that swap party or politician names (ΝΔ ↔ ΣΥΡΙΖΑ/ΠΑΣΟΚ, Μητσοτάκης ↔ another leader) while keeping the content's logic. If scores move a lot, the model is reacting to *who* is named rather than *how* they're treated. Report the mean shift per model.
- **Region error:** using the dev set, check whether a model consistently pulls the right tail toward the centre (the "neutrality retreat" the hackathon model had) or pulls the left tail. Report signed error per region.
- **Correction options:** a simple per-model calibration (isotonic regression from model score to human score, fit on dev), plus the ensemble. Errors in different directions partly cancel.

**Read:**
- Vallejo Vera & Driggers, "LLMs as annotators: the effect of party cues" (Palgrave 2025)
- arXiv 2601.04207 (logit steering for annotator alignment): an advanced option if a local model is clearly skewed

**Done when:** a "bias of our annotators" table is published in the repo (and eventually on the public methodology page).

---

## Phase 4 — Quality control: spend human time where models disagree (weeks 8–10)

### Step 4.1 — Disagreement routing

**Goal:** the final training labels, with humans having reviewed the most doubtful ~5–10%.

**Why:** when 3 different model families agree, they're usually right. Errors concentrate in the items where they disagree (in the medRxiv study, the most-disputed 6.5% held ~80% of errors). This makes a few hours of human review worth many thousands of labels.

**How:**
1. For each article, compute the ensemble disagreement (std of calibrated scores across models, or BT uncertainty).
2. Send the top ~5–10% to human annotators (same Argilla project, same codebook).
3. Replace their labels with the human ones, and keep a `label_source` field (`ensemble` / `human_review` / `gold`).

### Step 4.2 — Confident learning pass

**Goal:** catch labels that are still wrong after routing.

**Why:** some errors happen when all the LLMs agree but are wrong together (e.g. all fooled by one framing). A model trained with cross-validation can flag items whose label looks inconsistent with similar items.

**How:** train a quick model (TF-IDF+ridge or the encoder from Phase 5) with 5-fold cross-validation, and use **cleanlab** (it supports regression via `cleanlab.regression`) to rank likely label errors. Human-review the top ~100.

**Read:**
- Northcutt et al., 2021, "Confident Learning: Estimating Uncertainty in Dataset Labels" (JAIR). The idea behind cleanlab.
- MoLLIA (AAAI 2026), cited in research.md. A more advanced active-learning loop you can adopt once the basic version runs.

**Done when:** the training set is frozen as `dataset_v1`, with label provenance on every row.

---

## Phase 5 — Choose and train the models, from zero (code: weeks 3–8 on weak labels · final training: weeks 10–12)

There is no model to improve on. We are choosing and training from scratch, so this phase is written as a **decision process**: what to decide, in which order, using which evidence. The timing matters too. Write and debug the whole training pipeline **now**, on the 3,000 existing Gemini labels (weak but free). When `dataset_v1` lands in week ~10, final training is then a config change, not a new project.

### Step 5.0 — First decision: which jobs actually need a trained model?

**Goal:** a written list of the model "jobs" in the product, each with its own choice.

**Why:** at the hackathon, one fine-tuned generative LLM did everything (score + reasoning + JSON). That made one model responsible for jobs with very different needs, and when it disappeared, everything went with it. Most of these jobs don't need training at all.

| Job | Train it? | Model type | Runs |
| --- | --- | --- | --- |
| **Scorer** (article → `is_political`, `stance_detectable`, 3 axes + uncertainty) | **Yes**, the only model we must train | Encoder + regression/classification heads (or an LLM backbone with a regression head) | Every article, CPU |
| **Annotators** (labelling the training set) | No, prompted | 2–3 LLMs from different families (Phase 3) | Offline, once per labelling round |
| **Explainer** ("why this score?") | Not at first; prompt, and distil later only if cost requires | Generative LLM | On demand, cached |
| **Summariser** (per event) | No, prompted | Generative LLM | Once per event |
| **Embeddings / NER** for clustering | No, off-the-shelf | BGE-M3 / multilingual-e5, spaCy Greek | Every article, CPU |

**Done when:** the team agrees on this table. The rest of Phase 5 is about the **Scorer**, plus a short step on the Explainer.

### Step 5.1 — Don't train an LLM to write the score as text again

**Goal:** understand why the scorer should output numbers from a *head*, not *generated tokens*.

**Why:** the hackathon model generated JSON like `{"bias": 0.32}`. Training then uses cross-entropy over the digit tokens, which has four problems:
1. **The loss ignores distance.** If the target is `0.32`, predicting `0.42` and predicting `0.92` both get the tenths digit wrong (4 or 9 instead of 3), and cross-entropy penalises them almost equally. The model is never told that 0.42 is "closer". Regression loss (MSE/Huber) measures exactly that distance.
2. **No native uncertainty.** You get one number with no error bar. Token probabilities over digits are a poor, uncalibrated substitute.
3. **Fragility.** Sampling randomness, malformed JSON and refusals all need parsing and retries.
4. **Cost.** Generation runs autoregressively on a GPU. A regression head is one forward pass, and with an encoder it runs on CPU.

**How:** whatever backbone wins, the scorer is `backbone → pooled vector → small heads`. With Hugging Face this is `AutoModelForSequenceClassification` (with `num_labels` and `problem_type="regression"`) or a custom 20-line `nn.Module` for multi-task heads. This works for decoder LLMs (Llama/Qwen/Krikri) too: they have sequence-classification classes that pool the last token.

### Step 5.2 — Screen candidates for Greek ability (cheap, before any fine-tuning)

**Goal:** a short list of 3–4 backbones worth fine-tuning, chosen with an afternoon of measurements instead of opinions.

**Why:** info.txt asks to "validate Greek capability". General benchmarks (GreekMMLU etc.) test knowledge and reasoning, not "does this model's internal representation of a Greek news article encode its political stance". Two cheap tests answer the question we actually have.

**How:**
1. **Tokenizer fertility.** Count average tokens per Greek word over ~1,000 articles for each candidate tokenizer:
   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained(name)
   fertility = sum(len(tok(t).input_ids) for t in texts) / sum(len(t.split()) for t in texts)
   ```
   English-centric tokenizers often split Greek words into many fragments. A fertility of 3–4 versus ~1.5 means a 512-token window holds roughly half as much article, inference is slower, and quality is usually lower. Greek-specific models (GREEK-BERT, Krikri, which extended its tokenizer for Greek) should do much better here. This is a 10-minute experiment.
2. **Frozen-embedding linear probe.** For each candidate, embed the 3,000 articles *without* fine-tuning (mean-pooled hidden states), then fit a ridge regression to the Gemini scores. Evaluate by **leave-one-outlet-out** (train on efsyn+protothema, test on iefimerida + the new outlets). A model whose frozen embeddings already carry stance information across outlets is a strong fine-tuning candidate. A model that only does well within outlets is encoding outlet style.
3. Keep the candidates whose probe beats the TF-IDF baseline from Step 0.2 on unseen outlets.

**Read:** the Krikri paper (cited in research.md) section on tokenizer extension and Greek data, and the GREEK-BERT paper (Koutsikakis et al., 2020). Both explain *why* Greek-specific pretraining helps.

### Step 5.3 — Climb the model ladder, only as far as the gold set justifies

**Goal:** the smallest scorer whose quality is within noise of the best one.

**Why a ladder:** bigger models cost more to train, to serve and to iterate on. You only accept that cost if it buys a measurable gain on *our* gold dev set. Starting small also gives quick, cheap feedback while the data pipeline is still changing.

| Rung | Model | Size | Why it's on the ladder | Hardware to fine-tune |
| --- | --- | --- | --- | --- |
| 0 | TF-IDF + ridge | – | The floor. If a transformer can't beat this, something is broken. | laptop |
| 1 | **GREEK-BERT** (`nlpaueb/bert-base-greek-uncased-v1`) | 110M | Greek-native, tiny, CPU-servable. Likely the best value. | free Colab/Kaggle T4 |
| 2 | **XLM-RoBERTa-large**, or **mDeBERTa-v3-base** | 280–560M | Stronger multilingual encoders; usually beat base-size BERTs on harder semantic tasks. | 16GB GPU (fp16, grad-accumulation) |
| 3 | **Krikri-8B** (or a current 7–8B Qwen/Llama) + QLoRA + regression head | 8B | Upper bound: more world knowledge (who is who in Greek politics) and long context. | 24GB GPU (RTX 4090 / L4 / A10), hours |
| + | Matching models from the **KIEFERSA collection** | – | Slot each into the rung matching its size, judged by the same numbers. | – |

**Decision rule:**
- Train every rung with **3 random seeds**, and report mean ± std on the gold dev set (correlation with humans, MAE per region, unseen-outlet score).
- Move up a rung only if the gain is **larger than the seed spread** *and* worth the serving cost. As a rule of thumb, rung 3 must beat rung 2 by ≥ 0.05 Spearman on unseen outlets to justify a GPU in production.
- Record the decision in `docs/decisions/0001-scorer-model.md` (an "architecture decision record": context, options, evidence, choice). Next year's team will know *why*.

**Expected outcome (a guess for the gold set to confirm or reject):** rung 1 or 2 wins on cost/quality. Rung 3 is somewhat better on subtle opinion pieces but not by enough to justify serving it.

### Step 5.4 — The training recipe

**Goal:** a training script that is correct, leak-free and reproducible.

**How, and why each choice:**

- **Input text.** Use the fingerprint-stripped text (Step 2.1): `title [SEP] lead + body`, truncated to the model's window. Compare against **chunk-and-pool** (split into 512-token chunks, average the predictions) on dev. Opinion pieces often put their stance at the end, where truncation cuts it off.
- **Heads and losses (multi-task):**
  - `is_political`, `stance_detectable` → binary cross-entropy.
  - `econ_lr`, `cultural_gal_tan`, `gov_stance` → **Gaussian negative log-likelihood**: the head predicts a mean *and* a variance, and the loss rewards honest uncertainty. That gives the UI's "uncertain" indicator for free (Kendall & Gal, 2017).
  - **Mask the axis losses when `stance_detectable = no`.** Otherwise the model learns that neutral articles are "0.5" and drifts towards the centre, the same "neutrality retreat" the hackathon model had.
- **Sample weights** by label provenance: gold / human-reviewed > high-agreement ensemble > low-agreement ensemble (inverse BT variance).
- **Leak-free splits.** This is the most common silent bug in this task. Split by **event** (or MinHash near-duplicate group), not by random article. Otherwise the same story from 5 outlets ends up in both train and validation, and the model is graded on stories it has effectively already seen. Also hold out whole **outlets** for the generalisation test.
- **Two-level model selection.** Early-stop on the LLM-labelled validation split (large, noisy). Choose between finished runs on the **human gold dev set** (small, trustworthy). The gold **test** set is used once at the end.
- **Starting hyperparameters** (encoders): lr 2e-5 (base) / 1e-5 (large), batch 16–32, 3–5 epochs, 6–10% warmup, weight decay 0.01, fp16/bf16. For QLoRA on rung 3: rank 16, alpha 32, lr 1e-4–2e-4, 4-bit base. Sweep only the learning rate and epochs at first; the rest rarely matters much.
- **Optional, if the data allows: domain-adaptive pretraining.** Continue masked-language-model training of GREEK-BERT/XLM-R on 100k+ *unlabelled* Greek news articles (which the scrapers produce for free) before fine-tuning. Gururangan et al. (2020) showed consistent gains from this for in-domain tasks, and it costs one GPU-day.
- **Optional, if the Step 0.2 probe still shows outlet leakage after fine-tuning: adversarial de-biasing.** Add an outlet-classifier head behind a **gradient-reversal layer** (Ganin & Lempitsky, 2015). The encoder is trained to make the outlet *unpredictable* while keeping the stance predictable.
- **Tracking.** Log every run to W&B/MLflow with git commit, dataset version, seed and config.

**Read:**
- Gururangan et al., 2020, "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks" (ACL)
- Kendall & Gal, 2017, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" (NeurIPS). The idea behind predicting a variance; ignore the vision specifics.
- Ganin & Lempitsky, 2015, "Unsupervised Domain Adaptation by Backpropagation" (ICML). The gradient-reversal trick.
- Hinton et al., 2015, "Distilling the Knowledge in a Neural Network". Why soft (BT) targets carry more information than hard labels.
- Dettmers et al., 2023, "QLoRA" (NeurIPS). Only if you reach rung 3.

**Done when:** `train.py --config configs/greekbert.yaml` reproduces a run from scratch, and the pipeline has already been exercised end-to-end on the Gemini weak labels.

### Step 5.5 — Compute plan (we no longer have B200s)

- **Rungs 0–2:** free Kaggle (≈30 GPU-hours/week) or Colab GPUs are enough. A GREEK-BERT run on ~10k articles takes well under an hour on a T4.
- **Rung 3 / domain-adaptive pretraining:** a 24GB GPU for a few hours, rented hourly from a cloud provider, or academic HPC. Ask the university (CEID) and GRNET about student/research allocations, and ask Kiefer whether they can sponsor compute. That's a natural ask given the hackathon relationship.
- **Serving:** export the chosen encoder to ONNX and apply int8 quantisation. That runs on a cheap CPU server at the 2–5k articles/day this product sees.

### Step 5.6 — Artifact discipline (so a model is never lost again)

**Goal:** any trained model can be reloaded by anyone on the team, months later, in one line.

**Why:** this is exactly what failed. The `.nemo` file can't be used without the exact base checkpoint and framework version, and neither was recorded.

**How (non-negotiable rules):**
- Every checkpoint used for an evaluation number is `push_to_hub`'d to the team's private HF org as **safetensors** (adapters in standard **PEFT** format), with a model card listing: base model ID **and revision hash**, dataset version, git commit, config, seed and metrics.
- If any tool saves in a framework-specific format (NeMo, DeepSpeed shards), convert to HF format **the same day**.
- **Reload test:** before a model is deployed, a *different* team member loads it from the Hub on a different machine and reproduces the gold-dev numbers with `make eval`.
- The production service pins the model by Hub revision, and every stored score records it (`model_version`, Step 7.2).

**Done when:** the reload test passes for the chosen scorer.

### Step 5.7 — Explanations: prompt first, distil only if needed

**Goal:** a short, grounded explanation per article ("why this score"): which actors and policies are framed how, with quoted spans.

**Why:** explanations were the original reason for choosing an LLM (docs/design-decisions.md), and they build user trust. But they don't require training a model on day one. The best annotator from the Phase 3 bake-off, prompted with the CGCoT breakdown format, already produces exactly this.

**How:**
- Generate explanations **lazily**: only when a user clicks "why?", then cache them. Most articles are never opened, so this cuts LLM calls by an order of magnitude.
- The **score shown in the UI always comes from the scorer**; the LLM only explains. Add a consistency check: if the explanation says "strongly critical of the government" while the scorer says pro-government, flag it.
- Only if API cost or latency becomes a problem: distil into a local generative model (Krikri-8B or a Qwen 7–8B with LoRA), trained on the cached CGCoT breakdowns. Here a generative fine-tune *is* the right tool, because the output is text (unlike the score, Step 5.1).

**Done when:** a human spot-check of 50 explanations finds them accurate (quoted spans exist in the text; claims match the article).

### Step 5.8 (optional, research-grade) — Statistically valid outlet-level claims

**Goal:** honest statements like "outlet X moved 0.08 toward the government in 2026 (95% CI …)".

**Why:** averaging model predictions gives biased estimates when the model has systematic error. **PPI** (Prediction-Powered Inference) and **DSL** (Design-based Supervised Learning) use the small gold set to correct those averages, giving valid confidence intervals. That's needed if you ever publish rankings of outlets, which carries legal and reputational weight.

**Read:**
- Angelopoulos et al., 2023, "Prediction-Powered Inference" (Science)
- Egami et al., 2023, "Using Imperfect Surrogates for Downstream Inference" (DSL, NeurIPS)
- Chalmers benchmark (cited in research.md) comparing the two

**Phase 5 done when:** the chosen scorer beats the TF-IDF floor and the prompted annotator ensemble's *direct* scores on the human gold test set *and* on unseen outlets; its model–human agreement is close to human–human α; and it has passed the reload test.

---

# TRACK B — Events & Product

## Phase 6 — Event clustering: the missing "HOW" (weeks 2–7)

### Step 6.1 — Define "event" precisely

**Goal:** a one-paragraph definition plus examples.

**Why:** "same event" has levels. *Topic* (economy) ⊃ *storyline* (ΟΠΕΚΕΠΕ scandal, months long) ⊃ *event* (today's parliamentary vote on the ΟΠΕΚΕΠΕ committee). Clustering too coarsely means the event page mixes different happenings; too finely means each source ends up alone. You can't tune the clustering without deciding which level you want.

**How:** recommended: **event** = one real-world happening (statement, vote, incident, announcement) and its immediate reactions, within ~72 hours. Then link events into **storylines** as a second level (a great UI feature: "this event is part of the ΟΠΕΚΕΠΕ story — 14 earlier events").

### Step 6.2 — Build an evaluation set first

**Goal:** ~600 labelled article pairs ("same event" yes/no) and ~30 manually built full clusters.

**Why:** without labels, the similarity threshold is guesswork, and every tweak is judged by eyeballing. With labels, it's a 5-minute tuning loop.

**How:** sample pairs from the same 3-day windows (half with high embedding similarity, so the set includes hard near-miss cases like two different Parliament stories on the same day). Two team members label them. Also build 30 full clusters by hand from one busy news week.

**Read:** **SemEval-2022 Task 8, "Multilingual News Article Similarity"** (Chen et al., 2022). Directly relevant: it defines graded similarity between news articles (same event vs. same topic) and its system papers show what works. Read the task paper and the top 2 system descriptions.

### Step 6.3 — Implement online clustering

**Goal:** each newly scraped article is assigned to an existing event or starts a new one, in near-real time.

**Why online (incremental) rather than re-clustering everything:** news arrives continuously, and event IDs must stay stable. A user bookmarks an event page, so its URL can't change meaning at the next re-cluster.

**How (a proven, simple design):**
1. **Embed** each article (title + lead, ~first 256 tokens) with a multilingual embedding model that handles Greek well: **BGE-M3** or **multilingual-e5-large**. Compare both on your pair set.
2. **Extract entities** (people, organisations, places) with a Greek NER model: spaCy `el_core_news_lg`, or a GREEK-BERT NER fine-tune.
3. For a new article, take **candidate events active in the last ~72h**. Score each candidate:
   `score = a · cosine(article, event_centroid) + b · entity_overlap(Jaccard) + c · time_decay`
   Assign to the best candidate if `score > τ`, else create a new event.
4. Update the event centroid (running mean) and entity set.
5. A **nightly merge pass** joins events that drifted together (e.g. the same incident reported first as "fire in Attica" and later "evacuations in Varnavas").
6. Tune `a, b, c, τ` on the pair set; evaluate on the 30 hand clusters with **B-cubed precision/recall** (the standard cluster-quality metric: for each article, how much of its predicted cluster is correct, and how much of its true cluster was found).
7. Storage: use a vector index for candidate search. **Postgres + pgvector** is simple and robust, or MongoDB Atlas Vector Search if you stay on Mongo.

**Read:**
- Miranda et al., 2018, "Multilingual Clustering of Streaming News" (EMNLP). Almost exactly this design (centroid + time + entities, online), from a production news system. The most useful single paper for this phase.
- BGE-M3 paper (Chen et al., 2024). Skim it for why it handles long and multilingual text well.

**Done when:** B-cubed F1 ≥ ~0.8 on the hand clusters, and a live week of scraping gives events a human would agree with.

### Step 6.4 — Coverage-gap detection ("blindspots")

**Goal:** for each event, which parts of the spectrum covered it and which didn't.

**Why:** this is probably the **most powerful and cheapest feature** in the product. Much real bias is *selection*: what an outlet chooses not to report. It's informational bias, as in the BASIL paper. Ground News built its brand on its "Blindspot" feature. Once clustering exists, it's just counting: "Covered by 6 pro-government outlets, 0 opposition outlets."

**How:** per event, aggregate the article-level `gov_stance` and the axes into a coverage distribution. Flag events where one side contributes ≥ 80% of coverage and there are ≥ 4 articles. Use *article* scores, not hard-coded outlet labels, so it stays content-based.

---

## Phase 7 — Rework the backend for the product (weeks 5–9)

### Step 7.1 — Simplify the architecture to fit the scale

**Goal:** a pipeline that's cheap to run continuously and easy for the team to maintain.

**Why:** Kafka + Spark were right for a hackathon demo on B200s and show off engineering skill. But 10–15 sites produce maybe 2–5k articles/day, which is a few per minute. That's far below the scale these tools are built for. They add operational cost (JVM, ZooKeeper/KRaft, memory) and failure points with no benefit at this scale. Recommendation: **keep them only if showing that stack matters to you (e.g. for Kiefer); otherwise replace them** with:
- a scheduler (cron / APScheduler / Prefect) running the scrapers,
- a light queue (Redis + RQ/Celery, or simply a Postgres `jobs` table),
- workers: `clean → dedup → embed → score (encoder, CPU) → cluster → (per event) summarise (LLM)`.

**How:** follow design-decisions.md Step 1 (split into containers), but with this simpler service set: `scraper`, `worker`, `api`, `frontend`, `db`, and optionally `llm` (vLLM, only for summaries and explanations).

### Step 7.2 — Engineering hygiene that pays off immediately

- **Evaluation harness as code**: `make eval` runs every metric (gold test, unseen outlets, outlet probe, clustering B-cubed) and writes a report. Run it before merging any model change.
- **Experiment tracking**: MLflow or Weights & Biases (free for students). Without it you'll lose which prompt or seed gave which number.
- **Data versioning**: DVC, or versioned private HF datasets.
- **Scraper monitoring**: sites change their HTML without warning. Alert when a source returns 0 articles or the text length drops sharply.
- **Schema**: `articles` (raw + cleaned + fingerprint-stripped text, scores, uncertainty, model version), `events`, `storylines`, `user_feedback`. Always store `model_version` with each score, so you can re-score later and compare.

**Done when:** the full pipeline runs unattended for a week, on one modest server.

---

## Phase 8 — The objective event summary (weeks 8–11)

### Step 8.1 — Grounded multi-source summary

**Goal:** per event, a short neutral summary plus a "where coverage differs" section, with every sentence traceable to sources.

**Why:** this is the promise of info.txt ("an objective summary … then the reader chooses where to read"). It's also the riskiest feature. An LLM summary can hallucinate facts or pick up one side's framing, and on a bias product, a biased summary destroys credibility instantly.

**How:**
- Input: the event's articles (fingerprint-stripped), ordered to balance the spectrum, so the prompt isn't dominated by whichever side wrote more.
- Output structure (JSON → UI):
  1. **What happened**: only facts reported by ≥2 sources from *different* sides, or clearly attributed ("according to the ministry…").
  2. **Where coverage differs**: e.g. "Opposition-leaning outlets emphasise X; government-leaning outlets emphasise Y". Use the framing breakdowns (Step 3.2) of each article.
  3. **Citations** per sentence (article IDs).
- Model: frontier API model or Krikri/Qwen locally. Choose with Step 8.2's evaluation, as in the annotator bake-off.
- Neutral language rules in the prompt: no loaded terms, attribute all claims, use neutral names for contested things (follow the codebook's glossary).

### Step 8.2 — Evaluate faithfulness and neutrality

**Goal:** evidence that summaries are factually supported and politically balanced.

**How:**
- **Automatic faithfulness**: for each summary sentence, check with a Greek-capable NLI model (multilingual NLI such as an XLM-R/mDeBERTa NLI fine-tune) that it is *entailed* by its cited article(s). Flag unsupported sentences, and regenerate or drop them.
- **Neutrality test with our own scorer**: run the Phase 5 encoder on the summaries. Good summaries should score near the centre, with low variance across events.
- **Human panel**: 50 events, rated by a left-leaning and a right-leaning annotator on faithfulness, balance and completeness. The key metric: **both sides rate it as fair**.

**Read:**
- Laban et al., 2022, "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization" (TACL). How to check summaries sentence-by-sentence with NLI.
- Min et al., 2023, "FActScore" (EMNLP). Breaking text into atomic facts and verifying each, a stricter version of the same idea.

**Done when:** ≥95% of summary sentences pass the NLI check, and the human panel shows no significant difference in fairness ratings between the left and right raters.

---

## Phase 9 — Frontend & crowdsourcing (weeks 9–12)

### Step 9.1 — The core screens

- **Front page**: events ranked by (number of distinct outlets × spectrum spread × recency). A small **coverage bar** per event shows how coverage splits across the spectrum, plus a "Blindspot" badge from Step 6.4.
- **Event page**: neutral summary → "where coverage differs" → list of articles placed on the axes (2D plot or two bars), each with its score, an **uncertainty indicator**, and a "why?" expander (Step 5.7 explanation). Clicking goes to the original article (link out, don't republish; see Phase 11).
- **Storyline page**: a timeline of related events.
- **Methodology page**: codebook summary, gold-set α, model–human agreement, annotator-bias table, known limitations. Transparency is the best defence against "your bias meter is biased".

**Read (product research, not papers):** spend an afternoon studying **Ground News**, **AllSides** and **Media Bias/Fact Check**. Note what they do per *outlet* vs per *article*. Our differentiators are article-level, content-based, two-axis scores with uncertainty, for Greek news, which nobody does well.

### Step 9.2 — Crowdsourcing ("How would you classify this article?")

**Goal:** collect user judgments as extra evaluation and active-learning signal.

**Why it needs care:** website users aren't random. Partisans self-select, and coordinated groups can flood ratings (brigading). There is also the **hostile media effect**: partisans on *both* sides perceive the same neutral coverage as biased against them (Vallone, Ross & Lepper, 1985 — read the abstract; it predicts exactly the feedback you'll get). So raw crowd votes can't be ground truth.

**How:**
- Ask the user's own political self-placement (optional, anonymous). This makes the feedback interpretable, and it's an interesting research dataset on perceived bias in Greece.
- Mix in **gold "check" articles**. Users who agree with the gold labels get more weight.
- Aggregate with a rater-quality model (**Dawid–Skene**, available in the `crowd-kit` Python library) rather than simple averages.
- Use the output to (a) find articles where users strongly disagree with the model, and send those to Phase 4 human review, and (b) monitor drift.

**Read:** Dawid & Skene, 1979 (the original, short), or the crowd-kit docs for a practical intro.

---

# Both Tracks

## Phase 10 — Beta launch & continuous improvement (weeks 12–14, then ongoing)

### Step 10.1 — Closed beta

Invite ~50–100 users (classmates, politics/journalism students, Kiefer contacts). Track: return rate, clicks from event page to articles from the *other* side (the product's real mission metric: does it get people to read beyond their bubble?), feedback volume, and complaints of bias from each side.

### Step 10.2 — Drift monitoring and re-labelling schedule

**Why:** politics changes. New parties, new leaders, new scandals and alliances (look at how quickly the Greek party landscape shifted in 2023–2026). A model trained on spring data will slowly misread autumn data.

**How:**
- Monthly: label a fresh 50–100-article gold sample with the same codebook. Compare model–human agreement over time. A falling curve means it's time to retrain.
- Monitor score distributions per outlet over time. Sudden jumps are either real editorial changes or model breakage; check before publishing anything.
- Retrain quarterly through the same Phase 3–5 pipeline (it's automated by then).

---

## Phase 11 — Legal & ethics (start in week 1, finish before public launch)

Not optional, and cheaper to do early than to fix later.

- **Copyright / press publishers' right.** EU Directive 2019/790 (Art. 15) gives news publishers rights over online use of their articles, and Greece transposed it (Law 4996/2022). The safe product pattern: show **title + short snippet + link out**, plus *our own* summary. Never republish full text. Keep full text internally only for analysis. Have a lawyer or the university legal clinic review this before a public launch.
- **Scraping:** respect robots.txt and rate limits, identify the crawler, and stop if asked.
- **Defamation and reputation risk:** labelling a named outlet "far-right" is a statement about a real organisation. Mitigations: score *articles* not outlets; show uncertainty; publish the methodology; offer a correction/appeal form for publishers.
- **GDPR:** user accounts, feedback and political self-placement are **special-category data** (political opinions, GDPR Art. 9). Collect them only with explicit consent, store them separately and pseudonymised, and allow deletion. Better still, don't link them to identities at all.
- **Neutrality of the team:** document who annotated (lean balance), and have someone outside the team review the codebook for one-sidedness.

---

## Phase 12 — Future directions (after beta, pick by user demand)

Each is promising and builds on the foundation above. None should start before Phases 0–10 work.

1. **Outlet trend analytics**: "how outlet X's coverage of the government shifted over 2026", using PPI/DSL from Step 5.8 for valid confidence intervals. Strong for media researchers and journalists, and a possible paid tier.
2. **Personal "news diet" report**: a browser extension or account feature showing the user where their reading sits on the spectrum, with suggestions from the other side.
3. **Multimodal framing**: info.txt mentions JEV becoming multimodal. Image choice is a strong framing device (protesters shown as peaceful vs violent). Start by scoring headline + thumbnail together. Research first: very little Greek work exists, which makes it a publishable contribution.
4. **Headline vs body mismatch**: detect clickbait or slanted headlines whose stance differs from the article body. Easy once the scorer exists (score both separately).
5. **Expansion**: Cypriot outlets first (same language, different political axes, so it tests codebook flexibility), then the Balkans/EU with multilingual encoders.
6. **Publish the dataset and paper**: a Greek multi-axis media-bias corpus with human gold labels and documented LLM-annotator bias would be a real academic contribution (venues: LREC-COLING, the NLP4Democracy / NLP+CSS workshops, or Political Analysis / Research & Politics on the political-science side). It also strengthens funding applications.

---

## Consolidated reading list (in recommended order)

| # | Reading | Phase | Why it matters to us |
| --- | --- | --- | --- |
| 1 | Geirhos et al. 2020, Shortcut Learning | 0 | Explains the source-confound risk in our current data |
| 2 | Media Source Matters More than Content (EMNLP 2025) | 0 | Direct evidence of that risk in bias detection |
| 3 | CHES codebook (+ Greek party scores) | 1 | Two-axis template + external anchors |
| 4 | Benoit/Laver & Gemenis on RILE in Greece | 1 | Why one axis fails for Greek parties |
| 5 | Fan et al. 2019, BASIL (lexical vs informational bias) | 1, 6 | Bias via fact selection → blindspots, summaries |
| 6 | Card et al. 2015, Media Frames Corpus | 1, 8 | Vocabulary for "how each side frames it" |
| 7 | AgoraSpeech paper | 1, 3 | Greek-specific label definitions and anchors |
| 8 | Krippendorff, Computing α | 2 | How to measure and interpret agreement |
| 9 | LLMs as substitute for human experts (Research & Politics) | 2, 3 | Proven annotation design for political text |
| 10 | Wu et al., CGCoT | 3 | Our main annotation method |
| 11 | PairScale (NAACL Findings 2025) + arXiv 2506.06540 / 2510.22219 | 3 | Pairwise scaling and its failure modes |
| 12 | Vallejo Vera & Driggers (party cues) | 3 | How to test our annotators for bias |
| 13 | Northcutt et al. 2021, Confident Learning | 4 | Finding residual label errors |
| 14 | Koutsikakis et al. 2020, GREEK-BERT | 5 | Main encoder candidate |
| 15 | Hinton et al. 2015, Distillation | 5 | Why soft BT targets help |
| 15a | Gururangan et al. 2020, Don't Stop Pretraining | 5 | Domain-adaptive pretraining on unlabelled Greek news |
| 15b | Kendall & Gal 2017, What Uncertainties Do We Need | 5 | Predicting a variance → honest "uncertain" labels |
| 15c | Ganin & Lempitsky 2015, Gradient Reversal | 5 | Removing outlet information from the encoder |
| 15d | Krikri paper (ILSP) | 3, 5 | Greek tokenizer and pretraining; rung-3 candidate |
| 16 | Miranda et al. 2018, Multilingual Clustering of Streaming News | 6 | Blueprint for event clustering |
| 17 | SemEval-2022 Task 8 | 6 | Defines and evaluates "same event" similarity |
| 18 | Laban et al. 2022 (SummaC), Min et al. 2023 (FActScore) | 8 | Checking summaries for hallucination |
| 19 | Vallone, Ross & Lepper 1985, Hostile Media Effect | 9 | Predicts (and explains) the user feedback we'll get |
| 20 | Angelopoulos et al. 2023 (PPI), Egami et al. 2023 (DSL) | 5.8, 12 | Valid statistics for outlet-level claims |

---

## Key risks and how this plan handles them

| Risk | Where it's handled |
| --- | --- |
| Model learns outlet, not stance | Phase 0 probe, Phase 2.1 more outlets + fingerprint stripping, leave-outlet-out test in Phase 5 |
| "Bias" is ill-defined → noisy labels | Phase 1 codebook, two axes + government axis + stance gate |
| LLM labellers are politically skewed | Phase 3.3 counterfactuals, multi-family ensemble, calibration |
| Summaries hallucinate or lean | Phase 8 grounded generation, NLI checks, two-sided human panel |
| Clustering mixes events | Phase 6 explicit definition, labelled pairs, B-cubed tuning |
| Running costs too high after the hackathon | Phase 5 CPU encoder + Phase 7 simplified stack; LLM only per event |
| Legal action from publishers | Phase 11 link-out pattern, article-level scores, transparent methodology |
| Model goes stale as politics changes | Phase 10.2 monthly gold samples, quarterly retraining |
| A trained model gets lost again | Step 0.1 HF org + Step 5.6 artifact rules and reload test |
| Choosing an oversized model "because it's an LLM" | Step 5.0 job table + Step 5.3 ladder with a seed-noise decision rule |

## The first six concrete actions (this week)

1. Tag `v0-hackathon`, create the team's private HF org, upload the dataset and SFT files, and write `DATA_CARD_v0.md` (Step 0.1).
2. Run the three-part outlet probe on the existing 3,000 articles (Step 0.2). It takes about an afternoon in a notebook.
3. Everyone reads CHES + Benoit/Gemenis + BASIL (Step 1.1); book a 2-hour meeting to draft the codebook.
4. List 10–15 target outlets, check robots.txt/terms, and start RSS-based scraping with trafilatura (Step 2.1), so data accumulates while the codebook is written.
5. Track B: write the "event" definition and start labelling same-event pairs from the existing `kafka_feed.jsonl` + new scrapes (Steps 6.1–6.2).
6. Model track: write the Step 5.0 job table together, then run tokenizer fertility + the frozen-embedding probe on GREEK-BERT, XLM-R, mDeBERTa and Krikri (Step 5.2). It's one notebook, and it's the start of the model decision.

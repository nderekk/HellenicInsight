# SIMILAR WORK REFERENCES

# Labelling a Greek Political-Stance Corpus

October 20, 2026 · @Konstantinos Loukanaris

## Summary

You don't need to hand-label the corpus, and you don't need to trust any single model. You need three things: a small expert-labelled Greek gold set, several different LLMs labelling everything, and statistics that show where they are wrong.

The gold set is what makes the labels defensible. Which model you pick matters much less.

## 1. Define the label before labelling anything

Political stance has no objective ground truth. "Correct" can only mean that trained Greek coders, following a written codebook, would agree with the label. So write that codebook first. Two Greek-specific traps:

- **One left–right axis can mislabel Greek parties.** Gemenis showed that the Manifesto Project's RILE scale placed the far-left KKE and the leftist SYN as right-wing (Benoit et al.). Decide explicitly how you score economic versus cultural/national positions, and populist or anti-memorandum framing. You may need two scores, not one.
- **Many articles have no stance.** Add a first label such as "political? / stance detectable?". Otherwise neutral wire reports get forced to 0.5 and pollute the centre.

## 2. Don't just inherit the outlet's lean

Giving every article its newspaper's lean fails in a known way. A large European news project did this, and explainability analysis showed their classifier was predicting the news source, not the political leaning (CNR).

Use outlet lean only as a weak sanity check. Strip outlet names, bylines and boilerplate from the text: LLMs struggle to infer bias from content alone but recognise an outlet's orientation almost perfectly from its name (EMNLP 2025).

## 3. Build a human gold set

Three Greek annotators with different political leanings label about 300–600 stratified articles using the codebook. This is small, but non-negotiable.

- Measure inter-annotator agreement with Krippendorff's α. That number is your realistic ceiling.
- Judge every model and prompt against this set. Keep a test split that is never used for tuning.
- Tools: Argilla or Label Studio. A 2026 study ran its human labelling through Argilla on Hugging Face Spaces (arXiv 2604.13899).

## 4. Use LLMs as annotators, chosen by gold-set score

Being Greek-fine-tuned is not the right test; measured agreement with your Greek gold set is. In a four-country study, GPT-4 coding was highly accurate, dropped for longer news articles, and dropped only very slightly for non-English text (Research & Politics).

Still include a Greek model as one independent annotator. Llama-Krikri-8B (ILSP/Athena) is built on Llama 3.1-8B, heavily trained on Greek, open, and runs locally. An 8B model may be weaker at subtle ideological reasoning than frontier models, so let the gold set decide its weight.

**Use pairwise comparisons, not "give a score from 0 to 1".** Ask "which of these two articles is further right?" and fit a Bradley–Terry model to the answers. Pairwise prompts are a single task, avoid the model's unknown number scale, and still yield continuous scores (arXiv 2506.06540). Two refinements:

- **CGCoT (Wu et al.):** the LLM first writes a concept-specific breakdown of each text, then compares breakdowns. For you: which actors and policies the article supports or attacks, and its framing.
- **PairScale (NAACL Findings 2025):** extracts attitude evidence, then compares pairwise to get continuous scores.

In practice: compare each article against 10–20 random opponents plus a few fixed anchor articles. Run every pair in both orders, since LLM preferences often fail to flip when order is swapped (arXiv 2510.22219). Rescale the fitted scores to [0, 1].

## 5. Control for the models' own bias

LLM annotators skew in measurable ways. Vallejo Vera & Driggers found LLMs use party cues when labelling, and some are biased even for centre-left and centre-right parties, unlike humans (Palgrave 2025). Other work reports a systematic left-liberal lean across many topics (arXiv 2601.04207).

- Run counterfactual checks: swap party or politician names and see whether the score moves.
- Report error by region of the scale (far-left, centre, far-right), not only overall accuracy.
- Use an ensemble of different model families, so their biases don't all point the same way.

## 6. Turn disagreement into quality control

- **Send disagreements to humans.** Having a human expert settle disagreements between multiple GPT-4 runs boosted accuracy on political text (Research & Politics).
- **Disagreement concentrates the errors.** In a 2026 clinical study, the lowest-agreement 6.5% of items held an estimated 80% of all errors (medRxiv). Different domain, same pattern: spend human time only where models disagree.
- **Add active learning.** MoLLIA (AAAI 2026) uses a mixture of LLMs as annotators and flags items where the trained model and the LLM labels disagree.
- **Catch remaining noise.** Run confident learning (e.g. cleanlab) on the final training set to flag likely label errors.

## 7. Train the inference model

Fine-tune a Greek encoder with a regression head, using the Bradley–Terry scores (or their uncertainty) as soft targets. Evaluate only on the held-out human gold set.

- **Model options:** GREEK-BERT (beat M-BERT and XLM-R by 5–10% on two benchmarks), XLM-R, or Krikri with LoRA.
- **This has worked before:** transformers trained on hand-coded versus GPT-4-coded data gave almost identical results (Research & Politics).
- **For statistical claims** ("outlet X moved right in 2025"), correct estimates with DSL or PPI, which combine LLM labels with a small expert set. DSL often reduced bias more, but less consistently across datasets (Chalmers).

## Greek resources worth reusing

| Resource | What it holds | Use it for |
| --- | --- | --- |
| AgoraSpeech | 171 speeches by six party leaders, 2023 Greek elections; ChatGPT + human-validated paragraph labels incl. polarization and populism | Anchors; checking annotators on Greek party discourse |
| Manifesto Project (Greek manifestos) | Statement-level coded party programmes | Calibration |
| GreekPolitics | Greek political tweets labelled for polarity, figurativeness, aggressiveness, bias | Extra validation on short texts |

## Suggested pipeline

[embed: node/37b97d1b-492b]

The gold set's dev split chooses prompts and models; its untouched test split grades the final encoder.

## Sources

- LLMs as a substitute for human experts in annotating political text
- Krikri: Advancing Open LLMs for Greek
- Concept-Guided Chain-of-Thought for Pairwise Comparison Scaling
- PairScale (NAACL Findings 2025)
- LLMs as a substitute for expert political surveys
- Estimating the Error of LLMs at Pairwise Text Comparison
- LLMs as annotators: the effect of party cues
- Ideology as a Problem: Logit Steering for Annotator Alignment
- Media Source Matters More than Content (EMNLP 2025)
- Pluralistic news recommendation / EU news leaning dataset
- Benchmarking Debiasing Methods for LLM-based Parameter Estimates
- Multi-LLM Disagreement as a Detector of Annotation Errors
- Next Generation Active Learning: Mixture of LLMs in the Loop
- Human vs. LLM Annotation in Active Learning
- Benoit, Laver, Lowe, Mikhaylov — RILE and the Greek case
- AgoraSpeech dataset
- GreekPolitics dataset
- GREEK-BERT

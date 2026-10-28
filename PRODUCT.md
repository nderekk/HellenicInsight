# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: general Greek news readers who want to see how different outlets cover the same event and read beyond their usual bubble. The front page and event pages are built for them first.

Secondary: politics- and media-literate readers (journalism and politics students, journalists, researchers) who want article-level bias data and want to know how the scores are made. The methodology page and any future analytics serve them.

## Product Purpose

PoliticalBro is a Greek news aggregator. It scrapes Greek outlets at regular intervals, groups coverage of the same real-world event from different sources, scores each article's political position, and writes a neutral summary of each event. The reader then chooses where to keep reading, with the outlets' different angles shown side by side.

Success, as stated in the roadmap's mission metric: readers click through from an event page to articles from the *other* side of the spectrum. Secondary signals are return rate, feedback volume, and getting bias complaints from both sides about equally.

## Positioning

It scores individual articles, based on their content, for Greek news. Neighbours such as Ground News, AllSides and Media Bias/Fact Check mostly rate whole outlets, and none of them cover Greek media well. Target differentiators from the roadmap: multi-axis scores (economic left–right, GAL–TAN, government stance), an uncertainty indicator on every score, coverage-gap ("blindspot") detection that is computed from article scores rather than hard-coded outlet labels, and a published methodology.

## Operating Context

- Each event is one real-world happening (a statement, vote, incident or announcement) plus its immediate reactions, within about 72 hours. Events can be linked into longer-running storylines.
- Planned core screens (roadmap Phase 9):
  - **Front page:** events ranked by distinct outlets × spectrum spread × recency, each with a coverage bar and a Blindspot badge.
  - **Event page:** neutral summary, then "where coverage differs", then articles placed on the axes, each with its score, uncertainty and a "why?" explanation, linking out to the original.
  - **Storyline page:** a timeline of related events.
  - **Methodology page.**
- Planned crowdsourcing prompt: "How would you classify this article?" Readers can give an optional, anonymous political self-placement.
- Current state: the existing React app (`frontend/`) is the hackathon dashboard. It shows a live article feed that polls `/api/articles`, a Left/Center/Right lean meter per article, the model's reasoning, entity tags and on-demand URL analysis. It is legacy relative to the aggregator target.

## Capabilities and Constraints

- Stack in place: React + Vite + Tailwind + shadcn/ui frontend, Express + MongoDB backend, scraper in `live_scraper/` (outlets defined in `sites.toml`), Kafka/Spark pipeline.
- The interface is bilingual (Greek and English) from the start. Article content stays in its original Greek.
- Legal (EU Directive 2019/790 Art. 15, Greek Law 4996/2022): show title, a short snippet and a link out, plus our own summary. Never republish full article text.
- Defamation risk: score articles, not outlets. Always show uncertainty, publish the methodology, and offer publishers a correction/appeal route.
- GDPR: political self-placement is special-category data (Art. 9). It requires explicit consent, must be stored separately and pseudonymised, and must be deletable, ideally never linked to an identity.
- Open: event clustering, the multi-axis label schema and the trained scoring model do not exist yet. The hackathon model is lost, and current scores come from a single-axis 0–1 bias value. The final axis set depends on codebook v1.
- Terminology: *event*, *storyline*, *blindspot*, *coverage bar*, *methodology*.

## Brand Commitments

- Name: **PoliticalBro**. It replaces the working name HellenicInsight, which still appears in the repo.
- The product must read as politically neutral. Transparency is its defence against "your bias meter is biased".

## Evidence on Hand

- Pipeline, scraper and dashboard code in this repo. Architecture notes and the hackathon presentation are in `docs/`.
- `old-dep/datasets/labeled_dataset.jsonl`: 3,000 articles labelled by Gemini (single LLM, no human check, outlet-confounded). Metrics in `old-dep/model_metrics/` measure agreement with Gemini, not with humans.
- No human gold set, users, testimonials, press or agreement numbers exist yet. Do not fabricate accuracy, reliability or adoption claims.

## Product Principles

1. **Show the spread, don't pick a side.** The product's job is to make the range of coverage visible and invite reading across it, not to tell readers what is true.
2. **Article over outlet.** Every judgement is about a specific piece of content, never a blanket label on a publisher.
3. **Uncertainty is part of the answer.** A score without its confidence and its "why" is incomplete.
4. **Transparent by default.** Methodology, limitations and how the model compares with humans are public and easy to find.
5. **Send readers to the source.** Summarise and link out. Original journalism stays with the outlets.

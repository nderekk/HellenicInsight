---
name: PoliticalBro (legacy hackathon dashboard)
description: Live Greek news bias feed. Slate interface, red/blue spectrum, glass cards over a photo.
colors:
  ink: "#0f172a"
  ink-soft: "#1e293b"
  slate-body: "#334155"
  slate-muted: "#64748b"
  slate-faint: "#94a3b8"
  slate-rule: "#e2e8f0"
  slate-wash: "#f1f5f9"
  paper: "#ffffff"
  canvas: "#f7f9fb"
  left-red: "#dc2626"
  left-red-signal: "#ef4444"
  left-red-deep: "#b91c1c"
  left-red-tint: "#fee2e2"
  right-blue: "#2563eb"
  right-blue-signal: "#3b82f6"
  right-blue-deep: "#1d4ed8"
  right-blue-tint: "#dbeafe"
  verified-emerald: "#059669"
  wordmark-indigo: "#4f46e5"
typography:
  display:
    fontFamily: "ui-sans-serif, system-ui, sans-serif"
    fontSize: "3.75rem"
    fontWeight: 800
    lineHeight: 1
    letterSpacing: "-0.025em"
  headline:
    fontFamily: "ui-sans-serif, system-ui, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 800
    lineHeight: 1.25
  title:
    fontFamily: "ui-sans-serif, system-ui, sans-serif"
    fontSize: "1rem"
    fontWeight: 600
    lineHeight: 1.375
  body:
    fontFamily: "ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 500
    lineHeight: 1.625
  label:
    fontFamily: "ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.1em"
  micro:
    fontFamily: "ui-sans-serif, system-ui, sans-serif"
    fontSize: "0.625rem"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.1em"
rounded:
  sm: "4px"
  md: "6px"
  lg: "8px"
  xl: "12px"
  pill: "9999px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "24px"
  xl: "32px"
  2xl: "48px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.lg}"
    padding: "0 32px"
    height: "48px"
  button-primary-hover:
    backgroundColor: "{colors.left-red-deep}"
  button-nav-active:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    rounded: "{rounded.lg}"
    padding: "0 16px"
    height: "40px"
  button-nav-ghost:
    textColor: "{colors.slate-muted}"
    rounded: "{rounded.lg}"
    padding: "0 16px"
    height: "40px"
  button-outline:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.slate-body}"
    rounded: "{rounded.lg}"
    height: "40px"
  input-url:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.lg}"
    padding: "0 12px 0 40px"
    height: "48px"
  chip-source:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.slate-muted}"
    rounded: "{rounded.pill}"
    padding: "6px 16px"
  chip-source-selected:
    backgroundColor: "{colors.ink-soft}"
    textColor: "{colors.paper}"
    rounded: "{rounded.pill}"
    padding: "6px 16px"
  badge-lean-left:
    backgroundColor: "{colors.left-red-tint}"
    textColor: "{colors.left-red-deep}"
    rounded: "{rounded.pill}"
    padding: "2px 8px"
  badge-lean-center:
    backgroundColor: "{colors.slate-wash}"
    textColor: "#475569"
    rounded: "{rounded.pill}"
    padding: "2px 8px"
  badge-lean-right:
    backgroundColor: "{colors.right-blue-tint}"
    textColor: "{colors.right-blue-deep}"
    rounded: "{rounded.pill}"
    padding: "2px 8px"
  card-article:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink-soft}"
    rounded: "{rounded.xl}"
    padding: "24px"
---

<!-- LEGACY SNAPSHOT: this records the hackathon dashboard as shipped in frontend/src. It is the baseline, not the target. The event-aggregator redesign (see PRODUCT.md) is expected to replace this world and rewrite this file. -->

# Design System: PoliticalBro (legacy hackathon dashboard)

## Overview

**Creative North Star: "The Live Wire Room"**

The dashboard reads as a newsroom monitoring desk. Articles stream in from the pipeline, a red dot pulses beside "Live Pipeline Feed", and every card carries a thin lean bar like a gauge on a console. Density is medium: a three-column card grid on desktop, a centered single column on the analyzer, and small-caps labels that make sections feel like readouts.

The chrome is neutral slate on white so the political spectrum can carry all the color. Red marks the left and blue marks the right, following Greek party convention, and slate holds the center. Surfaces are translucent white (80% opacity with a light backdrop blur) over a fixed photo background washed with a left-red → white → right-blue gradient, so the spectrum literally sits behind the room.

This is a hackathon snapshot. Its shadcn/Tailwind defaults (system sans, zinc tokens, slate utilities) were never customised into an identity, and the wordmark still says "BigO No" / "Hellenic Insight".

**Key Characteristics:**
- Neutral slate chrome; color is reserved for the political spectrum.
- Translucent white glass cards over a spectrum-washed photo.
- Uppercase, widely tracked micro-labels as section readouts.
- Live signals: a pulsing dot, polling feeds, a spinner during analysis.
- Hover answers in red: borders and titles shift toward the left-red family.

## Colors

A cool slate grayscale with one opposing red/blue pair that encodes political lean, plus two single-use accents.

### Primary
- **Ink Slate** (ink): primary buttons, the active nav item, headings and the hero wordmark. It is the system's "action" color.
- **Assembly Red** (left-red, left-red-signal, left-red-deep, left-red-tint): Left lean. The signal shade fills lean bars, the deep shade is badge text and the primary-button hover, and the tint is the badge background. It also doubles as the interaction accent: card hover borders (red-300), title hover text, focus ring on the URL input and the analysis spinner.

### Secondary
- **Ballot Blue** (right-blue, right-blue-signal, right-blue-deep, right-blue-tint): Right lean. Same roles as Assembly Red, mirrored. Blue-600 also colors the outbound source link.

### Tertiary
- **Verified Emerald** (verified-emerald): only on the "Analysis Verified" shield label in the article modal.
- **Wordmark Indigo** (wordmark-indigo, used at 80% opacity): only on the "No" of the "BigO No" wordmark.

### Neutral
- **Ink Soft** (ink-soft): card titles, the selected source chip.
- **Slate Body** (slate-body): long-form reasoning text, outline-button text.
- **Slate Muted** (slate-muted): secondary text, subheads, inactive nav, source badges. It also represents Center lean (slate-500 bars, slate-100/600 badges).
- **Slate Faint** (slate-faint): icons, timestamps, micro-labels, empty-state text.
- **Slate Rule** (slate-rule): every border and hairline divider.
- **Slate Wash** (slate-wash): lean-bar track, Center badge background, modal body (slate-50).
- **Paper** (paper): card and input surfaces, usually at 80% opacity.
- **Canvas** (canvas): page fallback behind the background photo.

### Named Rules
**The Spectrum-Only Color Rule.** Saturated color means political lean. Red is Left (score ≤ 40), slate is Center (41–60) and blue is Right (> 60). Everything else is slate.

**The Greek Convention Rule.** Left is red and right is blue, matching the Greek party landscape. Never flip them to the US convention.

## Typography

**Display Font:** system UI sans (Tailwind `font-sans` stack)
**Body Font:** same
**Label/Mono Font:** same, set in uppercase with wide tracking

**Character:** one neutral system sans throughout, with hierarchy carried by weight (500 → 800) and by uppercase tracked labels rather than by a second family. Geist Variable is installed as a dependency but never imported.

### Hierarchy
- **Display** (800, 3rem → 3.75rem at md, line-height 1, tracking -0.025em): the hero wordmark only, with a red→blue gradient clipped to "Insight".
- **Headline** (800, 1.25rem → 1.5rem at sm, tight): article title in the analysis modal.
- **Title** (600–700, 0.875–1rem, snug): card titles and recent-analysis rows. They turn red on hover.
- **Body** (500, 0.875rem, line-height 1.625): model reasoning text. The hero subline uses 1.125–1.25rem at 500 weight.
- **Label** (700, 0.75rem, uppercase, tracking 0.1em): section headers such as "Recent Analyses", "Live Pipeline Feed" and "Agent Reasoning Log".
- **Micro** (700, 0.625rem, uppercase, tracking 0.1em): Left / Center / Right scale captions and card timestamps.

### Named Rules
**The Readout Rule.** Section headings are small, uppercase and tracked, flanked by hairlines or paired with an icon. They label instruments; they don't shout.

## Layout

Centered, single-column compositions inside generous page padding (32px). The analyzer is a 768px column with a 576px input row and recent list. The live feed is a 1152px grid: one column on mobile, two at md (768px) and three at lg (1024px), with 24px gutters. The nav bar is 64px tall, sticky, with a 1280px inner width.

Section headers use the "hairline · label · hairline" pattern: two flex-grow 1px slate rules with the label between them. Vertical rhythm steps through 12 / 16 / 24 / 32 / 48px. On mobile, the input and button stack, nav labels collapse to icons, and the modal grows to 90vh and scrolls.

## Elevation & Depth

A light hybrid: glass translucency is the main depth cue, and soft Tailwind shadows confirm state. Cards rest at `shadow-sm` with 80% white plus a 4px backdrop blur over the photo, and lift to `shadow-md` with a solid white fill on hover. The nav uses a heavier 12px blur. The modal is the only strongly elevated surface (`shadow-2xl`).

### Shadow Vocabulary
- **Rest** (`0 1px 2px 0 rgb(0 0 0 / 0.05)`): cards, inputs and panels at rest.
- **Lift** (`0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)`): hover on cards and rows, the primary button and the selected chip.
- **Modal** (`0 25px 50px -12px rgb(0 0 0 / 0.25)`): the analysis dialog only.

### Named Rules
**The Glass-Then-Lift Rule.** Surfaces are translucent at rest and turn opaque and lifted on hover. The hover is the moment a card becomes solid.

## Shapes

Gently rounded throughout. Controls use 8px (buttons, inputs), cards and panels 12px, and chips, badges and lean bars are full pills. The only circles are the live pulse dot, the spinner and the nav logo disc. Borders are 1px slate hairlines everywhere. Empty states use a dashed border to read as "awaiting data".

## Components

### Buttons
Restrained and solid. Ink by default, warming to red on intent.
- **Shape:** gently rounded (8px).
- **Primary:** Ink Slate fill, white 600 text, 48px tall, 32px horizontal padding, with a leading icon and Lift shadow. Hover turns Assembly Red deep.
- **Nav:** the active route is an Ink fill. The live-feed route goes red-600 when active. Inactive routes are ghost slate-muted.
- **Outline:** white/80 glass with a slate hairline, used for pagination and the language toggle (pill-shaped). Disabled is 50% opacity.
- **Focus:** shadcn ring, 3px at 50% of the ring color.

### Chips
- **Style:** source filters are pills with 12px bold text and 6px × 16px padding. Unselected is white with a hairline and a slate-50 hover; selected is Ink Soft with white text and the Lift shadow.

### Badges (lean)
- **Style:** a 20px pill with a tinted fill and a matching hairline border. Left is red-100/700, Center is slate-100/600 and Right is blue-100/700. Source badges are neutral outline or secondary.

### Cards / Containers
- **Corner Style:** 12px.
- **Background:** Paper at 80% with a 4px backdrop blur, turning solid on hover.
- **Shadow Strategy:** Rest → Lift (see Elevation).
- **Border:** slate hairline, red-300 on hover.
- **Internal Padding:** 24px (cards), 16–20px (rows, panels).

### Inputs / Fields
- **Style:** a 48px white field with a slate-300 hairline, an 8px radius, a leading search icon and the Rest shadow.
- **Focus:** red-500 ring.

### Navigation
- **Style:** a sticky 64px glass bar (white/80, 12px blur) with a bottom hairline. Logo disc and wordmark on the left; language toggle and route buttons on the right. On mobile, button labels collapse to icons.

### Lean Meter (signature)
A horizontal Progress bar (12px tall on panels, 4px × 48px on cards) on a slate-wash track. The fill length is the 0–100 score and the fill color is its lean band. Micro captions "Left · Center · Right" sit below it. On the feed, a selected source gets an average-bias panel whose border and icon tint follow the mean score.

### Live Pulse (signature)
An 8px red dot with `animate-pulse` before the live-feed label, plus a 48px spinner (slate-100 ring, red-600 top) while analysis runs.

## Do's and Don'ts

### Do:
- **Do** reserve saturated red and blue for lean: red for scores ≤ 40, slate for 41–60, blue above 60.
- **Do** keep chrome in slate (ink #0f172a through rule #e2e8f0) so the spectrum reads.
- **Do** set section headings as uppercase tracked labels (12px, 700, 0.1em).
- **Do** use translucent white surfaces at rest and solid white with the Lift shadow on hover.
- **Do** keep 1px slate hairlines as the only border style, dashed only for empty states.

### Don't:
- **Don't** flip the political colors to the US convention (red = right).
- **Don't** introduce new saturated accents. Emerald and indigo are single-use legacy marks, not palette options.
- **Don't** treat this snapshot as the target identity for the event aggregator. The redesign replaces it.

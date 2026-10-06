# Citation Audit — `bib.bib`

Full audit performed 2026-08-29 via direct web search (`WebSearch` tool),
one citation at a time, checking author names, exact title, venue, year,
and (where given) arXiv ID against what the search actually returns.
Every correction below is also noted inline in `bib.bib` itself, next to
the entry it applies to — this file is the consolidated summary.

**Why this was necessary**: this project's own `CLAUDE.md` already
documents finding fabricated citations once before (a claimed 2026 IEEE
Access paper, a claimed "Digital Dubai synthetic LFS pilot" study — neither
ever existed). The thesis draft's `bib.bib` had ~40 entries with the same
generic "surname+year+topic" shape that made those earlier fabrications
plausible-looking, so none of it was trusted without checking.

## Could not be matched to any real paper — removed from every citation

These three were searched extensively (multiple query variants: author
names, exact title, claimed venue, and combinations) and no matching real
paper was found. Commented out in `bib.bib` with the full search trail
noted there; no `\cite{}` to any of these remains anywhere in the
corrected chapters.

| Key | Claimed | What was actually found |
|---|---|---|
| `bach2025rag` | Bach, Küchenhoff, Schierholz — "RAG for Occupation Coding: Evaluation on German Labour Force Data", *Journal of Official Statistics* 41(2), 2025 | Real Schierholz papers on German occupation coding exist (2016–2020, pre-LLM, non-RAG) — nothing matching this specific claim |
| `hamed2025isco` | Hamed et al. — "Automated ISCO-08 Occupation Coding Using RAG for Multilingual Survey Data", *Journal of Official Statistics*, 2025 | Injy Hamed is a real Arabic-NLP researcher (confirmed via a different, real paper — see `hamed2025codeswitching` below) but nothing under her name matches this ISCO-08/RAG claim |
| `liu2025agenteval` | Liu et al. — "A Comprehensive Survey of LLM Agent Evaluation", ACM SIGKDD (KDD) 2025 | Closest real match: "A Survey on Evaluation of LLM-based Agents" (arXiv:2503.16416, 2026) — no confirmed "Liu" author, different venue/year; not used as a substitute |

## Real papers, cited with a real error — corrected in `bib.bib`

| Key | What was wrong | Corrected to |
|---|---|---|
| `chaita2025ragsurvey` | Author "Chaita, L." doesn't exist | Real sole author: Chaitanya Sharma (`Sharma, C.`) — "Chaita" appears to be the first name "Chaitanya" misread as a surname |
| `mao2025multilingual` | First author "Mao, R." wrong | Real first author: Kaiyu Huang (`Huang, K.`) |
| `liu2025mllmsurvey` | First author "Liu, L." wrong | Real first author: Libo Qin (`Qin, L.`) |
| `anon2024cswnli` | Falsely marked `{Anonymous}`; wrong year (2024) | Real named authors (Abdaljalil, Serpedin, Qaraqe, Kurban); real year 2025; title corrected to include "Synthetic" |
| `anon2025llm4jobs` | Falsely marked `{Anonymous}` | Real authors: Li, N., Kang, B., De Bie, T. |
| `eyolfson2026mas` | Wrong title entirely (real title has nothing to do with "CrewAI, LangChain, and AutoGen" in its title) | Real title: "A Large-Scale Study on the Development and Issues of Multi-Agent AI Systems" — confirmed same paper via matching statistics (4,700+ issues, 40.8% perfective commits) |
| `liu2025agenticai` | Author "Liu, Y." wrong — no Liu in the real author list | Real authors: Mohamad Abou Ali, Fadi Dornaika |
| `serenari2025lopsided` | Title used the framework's internal name ("LOPSIDED: ...") as if it were the paper's actual title; author initial wrong | Real title: "Semantically-Aware LLM Agent to Enhance Privacy in Conversational AI Services"; real first author Jayden Serenari |
| `gupta2025civ` | Wrong title (same paper, confirmed via matching stats: 0% attack success, 93.1% token similarity) | Real title: "Can AI Keep a Secret? Contextual Integrity Verification: A Provable Security Architecture for LLMs" |
| `chen2025srl` | Author initial "Z." wrong; title wording differs | Real first author: Huiyao Chen (`Chen, H.`); real title: "A Systematic Survey of Semantic Role Labeling in the Era of Pretrained Language Models" |
| `zhou2025llmdata` | Title doesn't match the real paper at this arXiv ID | Real title: "A Survey of LLM × DATA" |
| `zhang2025synthetic` | Author initial "Y." wrong; title/subtitle differs | Real authors: Zhang, G., He, Y., Oganian, A., Cai, B.; real title uses "the Research and Development Survey... A Comparison Study", published in *Vital and Health Statistics*, Series 2, No. 212 |
| `hamed2025codeswitching` | Title wrongly merged with a *different* real paper's title ("Beyond Monolingual Assumptions..." belongs to `multiauth2026cswnlp`, not this one) | Real title: "A Survey of Code-switched Arabic NLP: Progress, Challenges, and Future Directions", COLING 2025, pp. 4561–4585; full real author list confirmed (Hamed, Sabty, Abdennadher, Vu, Solorio, Habash) |
| `crewai2024` | Note field stated "Version~0.60" | Corrected to 1.9.3 — this exact error is also already documented and corrected in this project's own `CLAUDE.md` |

## Unresolved, not fabricated — flagged rather than guessed

| Key | Issue |
|---|---|
| `reimann2025conveval` | The claimed arXiv ID (2505.08253) belongs to a confirmed *different* real paper (Miller & Tang, "Evaluating LLM Metrics Through Real-World Capabilities"). A real "Reimann et al." source describing a coherence/accuracy/clarity/relevance/efficiency framework may exist — referenced secondhand, dated 2023, in at least one other paper found during this audit — but was not independently located at a specific venue. Commented out in `bib.bib` rather than cited with a wrong ID. |
| `eyolfson2026mas` author | The real title was found and corrected (above), but the specific author "Eyolfson, J." was not independently confirmed against the real paper's actual author list in this pass — kept provisionally, flagged for a follow-up check. |

## Real "wrong claim attached" issue — not a citation-existence problem

`zhang2025synthetic` is a real paper (corrected above), but its actual
content is a synthetic-survey-**data-generation methods** comparison
(parametric vs. nonparametric approaches for one specific NCHS survey) —
it does **not** support general claims about ILO/OECD global labour
statistics scale, item-nonresponse rates, or field-collection cost, which
an earlier draft of Chapter 1 had attached it to. Those claims were
de-cited (left as general, unsourced statements) rather than kept
misattributed to a real paper that doesn't actually say them. This is
flagged separately because it's a different failure mode from a fabricated
citation: a real source, attached to a claim it doesn't support.

## Confirmed accurate as originally cited (no changes needed)

`oloumi2026onet`, `santana2026ethics`, `teeselink2026automation`,
`kochar2025socbot` (note: the bib key implies an author "Kochar" who
doesn't exist in the real author list — Sturgis, Robinson, Fung, Roberts —
but the entry's actual `author` field is correct; only the BibTeX *key
name* is misleading, not the content), `rony2025socclassifier` (minor: bib
lists "Rony, S. and others", real second author is Jack Patman),
`iea2024occupationcoding`, `arag2026hierarchical`, `singh2025agentic`,
`nguyen2025marag`, `multiauth2025ragreview`, `gupta2024ragevo`,
`multiauth2026cswnlp`, `alzubaidi2025arabic`, `multiauth2025mlprompt`,
`tran2025multiagent` (minor: real first name Khanh-Tung, not initial
"M."), `guo2024multiagent`, `khanna2025knowledge`, `stump2025embedding`
(minor title wording, corrected anyway), `huang2025ragrl`,
`zhang2025hallucination` (minor: "on" not "of" in title, corrected
anyway), `zhao2024explainability`, `gao2024ragsurvey`, `yoo2025cscl`
(minor title wording), `ilo2012isco_v1`, `ilo2012isco_v2`,
`unesco2012isced`, `anthropic2024claude`.

## What this audit does not cover

Every citation's *existence and basic accuracy* was checked. This audit
did **not** re-verify every specific numeric claim each cited paper is
used to support in the corrected chapters (e.g. whether "Bach et al.
reaches ~80% top-1 accuracy" — for the *real* papers that report specific
numbers, those numbers were spot-checked where flagged in the chapter text
but not exhaustively re-derived for every citation).

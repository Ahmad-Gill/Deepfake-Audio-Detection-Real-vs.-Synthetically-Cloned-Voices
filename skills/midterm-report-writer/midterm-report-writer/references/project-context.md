# Project Context — AI-600 Deepfake Audio Detection (Spring 2026)

## Course & Assignment
- **Course:** AI-600 Deep Learning
- **Semester:** Spring 2026
- **Project Title:** Deepfake Audio Detection: Real vs. Synthetically Cloned Voices
- **Task Type:** Supervised Binary Classification on Audio
- **Mid-report Deadline:** 08/05/2026 (11:55 PM)
- **Final Report Deadline:** 22/05/2026 (11:55 PM)
- **Presentation:** 25/05/2026
- **Report Format:** Single PDF (no page limit specified, aim for 6–8 pages including references)

## Mid-Report Weight & Requirements
- **Weight:** 25% of total grade
- **Required sections per assignment:**
  1. Problem statement
  2. Methodology
  3. Current progress
  4. Experiments/results (if available)
  5. Future plan / remaining work
  6. All group members' names and IDs

## Dataset
- **Primary:** ASVspoof 2019 Logical Access (LA) track
- **Download:** https://datashare.ed.ac.uk/handle/10283/3336
- **Size:** 121,000+ utterances, 107 speakers
- **Train/Dev splits:** Attacks A01–A06 (6 known attacks: 2 VC + 4 TTS)
- **Evaluation set:** Attacks A07–A19 (13 unknown attacks)
- **Key challenge:** Generalization to unknown attacks = your project's core theme

## Evaluation Metrics
- **Primary:** EER (Equal Error Rate) — where FAR = FRR
- **Secondary:** min t-DCF (tandem Detection Cost Function)
- **Both:** lower is better
- **Evaluation partition:** Always report on LA eval set (not dev)

## Known Baseline Numbers (for results table)
| Model | EER (%) | min t-DCF | Source |
|-------|---------|-----------|--------|
| CQCC + GMM | 9.57 | 0.2366 | [3] Official baseline |
| LFCC + GMM | 8.09 | 0.2116 | [3] Official baseline |
| RawNet2 | 5.13 | 0.1175 | [4] Tak et al. |
| AASIST | 0.83 | 0.0290 | [5] Jung et al. |
| AASIST-L | 1.32 | 0.0500 | [5] Jung et al. |
| 2026 Resolution-Aware [F] | 0.16 | — | Very recent SOTA |

## Your Proposed Methodology (Default if not specified)
- **Baseline model:** RawNet2 (implement and reproduce first)
- **Proposed model:** AASIST with SSL features
- **Features:** SSL representations from wav2vec 2.0 (intermediate layers per [8])
- **Augmentation:** RawBoost (3 noise modes per [9])
- **Loss:** Weighted binary cross-entropy (Adam, lr=1e-4)
- **Input:** 4-second segments (64,000 samples at 16kHz)
- **Batch size:** 32

## Team Information (PLACEHOLDER — Replace with actual details)
```
| Name | Student ID |
|------|-----------|
| [Member 1 Name] | [ID] |
| [Member 2 Name] | [ID] |
| [Member 3 Name] | [ID] |
```

## Current Progress (PLACEHOLDER — Update from user input)
- [ ] RawNet2 baseline implemented
- [ ] RawNet2 training complete
- [ ] AASIST implementation started
- [ ] SSL feature extraction implemented
- [ ] RawBoost augmentation implemented
- [ ] Initial results available

## Report File Output
- **Save as:** `reports/midterm-report-[date].md`
- **Convert to PDF:** Using pandoc or similar before LMS submission
- **GitHub:** Code pushed before submission with repo link on first page

## Academic Writing Standards
- **Tense:** Past tense for completed work, present for descriptions/claims
- **Person:** Third person academic ("This project investigates...")
- **Citations:** IEEE numbered format [1], [2], etc.
- **Plagiarism note:** All AI tool usage must be disclosed in final report
  (Statement on AI tool usage is mandatory for final report per assignment)

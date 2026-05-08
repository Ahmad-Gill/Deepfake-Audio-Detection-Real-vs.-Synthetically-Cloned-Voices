---
name: midterm-report-writer
description: "Use this skill to write, draft, or generate the AI-600 midterm/mid-report for the Deepfake Audio Detection semester project. Triggers include: 'write my midterm report', 'draft mid report', 'generate project report', 'write literature review section', 'write methodology section', 'help me write the report', 'create report draft', 'write progress report', or any request to produce academic writing for the deepfake detection project. Also triggers when the user asks to improve, expand, or refine any section of an existing draft. This skill produces a complete, downloadable markdown file formatted as an academic report."
---

# Midterm Report Writer — AI-600 Deepfake Audio Detection

## Purpose

Produce a complete, well-structured academic mid-report in markdown format for the AI-600 Deepfake Audio Detection project. The report:
- Follows the exact required format (problem statement, methodology, progress, results, future work)
- Cites all 10 selected papers correctly
- Integrates 2025–2026 papers for recency and impact
- Reads like a genuine academic submission, not an AI-generated template
- Is saved as a downloadable `.md` file

---

## Step 1: Load All Reference Files

Before writing a single word, load these files in order:

```
1. skills/midterm-report-writer/references/project-context.md
   → Your project details, dataset, metrics, baselines

2. skills/midterm-report-writer/references/papers-index.md
   → All 10 core papers + 2025-2026 papers with roles and citable claims

3. skills/midterm-report-writer/references/report-structure.md
   → Detailed section-by-section writing guide

4. skills/midterm-report-writer/references/2025-2026-papers.md
   → Recent papers to incorporate for academic relevance
```

Also scan `paper-summaries/` if it exists — use any generated summaries to enrich content.

---

## Step 2: Gather Missing Information from the User

Before writing, identify what you don't know. Ask ONLY about items not already in the reference files:

```
Essential (must ask if missing):
□ Team member names and student IDs
□ Current progress (what has been implemented so far?)
□ Preliminary results (any numbers? even partial?)
□ Proposed architecture (which model are you using?)

Optional (use defaults from project-context.md if not provided):
□ Specific augmentation strategy (default: RawBoost)
□ Feature type (default: SSL + spectrogram)
□ Training details (default: standard from papers)
```

Consolidate into ONE question with bullet points. Do not ask in multiple turns.

If the user says "write the full report" with no other details, use sensible defaults from `project-context.md` and note placeholder sections clearly with `[TODO: ...]` tags.

---

## Step 3: Write the Report

Follow the template in `references/report-structure.md` exactly.

### Core Writing Rules

**Academic tone:**
- Write in third-person academic voice ("This paper proposes...", "The model is trained...")
- Use past tense for completed work, present tense for descriptions
- No bullet points inside paragraphs — write in prose
- Each section should flow naturally into the next

**Citations:**
- Use numbered IEEE-style inline citations: [1], [2], [3]
- Every factual claim about a paper must be cited
- Do not cite papers you haven't described
- Reference list at the end must match inline citations exactly

**Quality signals for a strong report:**
- Introduction motivates with real-world threats (voice cloning, fraud, misinformation)
- Literature review is organized by paradigm, not chronologically
- Methodology explains WHY each choice was made, not just what
- Results section has a comparison table even with partial numbers
- Future work is specific, not generic ("We plan to test on ASVspoof 2021 DF track" not "We plan to do more experiments")

**Avoiding weak writing:**
❌ "Many researchers have studied deepfake detection."
✅ "The ASVspoof challenge series [2,3] has driven systematic benchmarking of spoofing countermeasures, with the 2019 edition introducing 19 distinct TTS and VC attack types across the Logical Access track."

❌ "Our model achieves good results."
✅ "The reproduced RawNet2 baseline achieves an EER of 5.3% on the ASVspoof 2019 LA evaluation set, consistent with the originally reported 5.13% [4]."

---

## Step 4: Report Sections (Summary)

See `references/report-structure.md` for full details on each section.

| Section | Target Length | Key Papers | Priority |
|---------|--------------|------------|----------|
| Header (names, IDs, title) | Short | — | Required |
| Abstract | 150–200 words | Survey [1], Dataset [2] | Required |
| 1. Introduction | 300–400 words | [2], [10] + 2026 papers | Required |
| 2. Literature Review | 600–800 words | [1],[4],[5],[6],[7],[8] + 2025 papers | Required |
| 3. Methodology | 500–700 words | [4],[5],[7],[8],[9] | Required |
| 4. Experimental Setup | 200–300 words | [2],[3],[6] | Required |
| 5. Preliminary Results | 200–400 words | [3],[6] | Required |
| 6. Future Work | 200–300 words | 2025–2026 papers | Required |
| 7. Conclusion | 100–150 words | — | Required |
| References | Full list | All cited | Required |

---

## Step 5: Integrate 2025–2026 Papers Strategically

Do NOT just mention these papers — use them purposefully:

- **Introduction:** Use 2026 papers to show the threat is growing (LLM-based TTS, flow-matching synthesis)
- **Literature Review:** Use 2025 papers to show the field's current direction (Mamba, SSL layer analysis, Interspeech 2025 submissions)
- **Future Work:** Use 2025 generalization papers to motivate your next steps
- **Discussion (if included):** Compare your approach to 2025 SOTA

See `references/2025-2026-papers.md` for specific papers and exact claims.

---

## Step 6: Generate the Markdown File

Save to: `/mnt/user-data/outputs/midterm-report-[date].md`

The file must open cleanly in any markdown viewer and render correctly as a PDF.

After saving, use `present_files` to share with the user.

---

## Handling Special Cases

**"I have no results yet"**
→ Write a placeholder Results section: reproduce the problem, show your training setup, and report "Experiments are in progress." Include the expected comparison table with baselines filled in and your results as "TBD."

**"Just write the literature review"**
→ Write only that section but make it self-contained and citable. Match the style in `report-structure.md`.

**"Improve my draft"**
→ Ask for the draft. Identify weak sections, fix academic tone, strengthen citations, improve transitions.

**"Write in LaTeX instead"**
→ Follow the same content but wrap in LaTeX `\section{}`, `\cite{}` commands. Save as `.tex`.

**"We have preliminary results"**
→ Always include them, even if partial. A partially-filled results table with 1–2 numbers is better than no table. Be transparent about what is complete.

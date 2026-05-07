---
name: paper-reader
description: "Use this skill whenever the user uploads a research paper (PDF) and wants it explained, summarized, or broken down into a simplified markdown file. Triggers include: 'read this paper', 'explain this paper', 'summarize this research', 'break down this paper', 'simplify this paper', 'what does this paper say', or any request to extract key information from an academic PDF and produce a structured markdown summary. Also triggers when the user asks to analyze a paper in the context of their semester project, literature review, or mid-report preparation. Do NOT use for generic PDF reading (invoices, forms, manuals) — this skill is specifically for academic research papers."
---

# Paper Reader — Research Paper to Simplified Markdown

## Purpose

Read an uploaded research paper (PDF) and produce a clean, structured markdown file that explains the paper in the simplest possible terms. The output is designed for a student who needs to:
1. Understand the paper quickly without re-reading it
2. Know exactly what to cite and where in their own report
3. Extract usable methodology details, results, and formulas

## Step 1: Load Project Context

Before processing any paper, read the project context file to understand the user's research objective:

```
/mnt/skills/user/paper-reader/references/project-context.md
```

If the file does not exist or is inaccessible, ask the user:
- What is your project/thesis topic?
- What dataset are you using?
- What is the deliverable (mid-report, final report, thesis)?

Keep the project context in mind throughout — it determines the "Relevance to Your Project" and "What to Use in Your Report" sections.

## Step 2: Extract Paper Content

Use the pdf-reading approach to extract text from the uploaded PDF:

```bash
# Check the PDF structure
pdfinfo <paper.pdf>

# Extract full text
pdftotext -layout <paper.pdf> /tmp/paper_text.txt

# Preview extraction quality
head -50 /tmp/paper_text.txt
```

If text extraction is poor (scanned PDF, garbled text), rasterize key pages:
```bash
pdftoppm -jpeg -r 150 -f 1 -l 1 <paper.pdf> /tmp/page
```

Then read the extracted text or images to understand the paper.

Read the FULL paper. Do not skim. Pay special attention to:
- Abstract (the paper's own summary)
- Section 1 / Introduction (problem statement, motivation, contributions)
- Methodology / Proposed Method (architecture, pipeline, training details)
- Experiments / Results (datasets, metrics, baseline comparisons, ablation tables)
- Conclusion (summary claims, limitations, future work)
- Tables and Figures (often contain the most important results)

## Step 3: Generate the Markdown File

Create a markdown file at `/mnt/user-data/outputs/<sanitized_paper_title>.md` following the template below. Every section is mandatory. Use simple, plain language — write as if explaining to a smart undergraduate who hasn't read the paper.

---

### Output Template

```markdown
# [Paper Title]

> **Authors:** [First Author] et al. ([Year])
> **Published in:** [Venue — conference, journal, or arXiv]
> **Paper ID / DOI:** [arXiv ID or DOI if available]

---

## 1. What problem does this paper solve?

[2–3 sentences. State the problem in plain English. No jargon. 
Example: "Current deepfake audio detectors work well on the data they were 
trained on, but fail badly when they encounter new types of fake audio 
they haven't seen before. This paper tries to fix that gap."]

## 2. Why does this problem matter?

[2–3 sentences. Real-world impact. Why should anyone care?
Connect to broader implications — security, safety, deployment, etc.]

## 3. What has been tried before? (Prior work summary)

[Bullet list of 3–5 key prior approaches the paper builds on or compares against.
For each: one sentence on what it did + one sentence on its limitation.
Example:
- **RawNet2**: End-to-end model on raw waveforms. Limitation: poor generalization 
  to unseen spoofing attacks.
- **AASIST**: Graph attention on spectro-temporal features. Limitation: ...
]

## 4. What does this paper propose? (Method)

### 4.1 High-level idea
[1–2 sentences. The core insight in plain English.
Example: "Instead of training on spectrograms, this paper feeds raw audio 
into a graph neural network that looks at both frequency patterns and 
time patterns simultaneously."]

### 4.2 Architecture / Pipeline
[Describe the full pipeline step by step:
1. Input: what goes in (raw waveform? spectrogram? SSL features?)
2. Feature extraction: how features are computed
3. Model backbone: what architecture processes the features
4. Output head: how the final prediction is made (binary classifier? scoring?)

Use simple language. If the paper has a figure showing the architecture, 
describe it in words.]

### 4.3 Key technical details
[List the 3–5 most important technical choices:
- Loss function used and why
- Training hyperparameters (learning rate, batch size, epochs, optimizer)
- Any special techniques (data augmentation, curriculum learning, etc.)
- Input representation details (sample rate, window size, feature dimensions)]

### 4.4 What makes this different from prior work?
[2–3 sentences. The specific novelty claim. What is new here that wasn't 
done before?]

## 5. Experiments and Results

### 5.1 Datasets used
[List each dataset with: name, size, what it contains, and how it was split.
Pay special attention to whether the paper uses ASVspoof 2019 LA — 
if so, note the exact protocol (train/dev/eval splits, known vs unknown attacks).]

### 5.2 Evaluation metrics
[List metrics used: EER, t-DCF, accuracy, F1, AUC, etc.
Briefly explain each metric in one sentence.]

### 5.3 Main results
[Reproduce the KEY results table from the paper — only the most important 
comparison. Include:
- This paper's best result
- Top 2–3 baselines it compared against
- The improvement margin

Format as a markdown table.]

### 5.4 Ablation studies
[What components did they test removing/changing? What did they learn?
Summarize as bullet points. This section reveals which parts of the 
method actually matter.]

## 6. Limitations and weaknesses

[List 2–4 limitations. Include both:
- Limitations the authors acknowledge
- Limitations you identify that the authors didn't mention
Be specific: "only tested on English speech" not "limited evaluation"]

## 7. Key takeaways

[3–5 bullet points. The most important things to remember from this paper.
Write each as a complete, self-contained sentence.]

## 8. Relevance to your project

### 8.1 How this paper connects to your work
[2–3 sentences explaining the direct connection to the user's project.
Reference the project context — dataset overlap, methodology overlap, 
shared challenges.]

### 8.2 What you can use from this paper
[Specific, actionable items:
- Can you use their architecture? (fully or partially)
- Can you use their training strategy?
- Can you use their evaluation protocol?
- Can you use their augmentation method?
- Are their baseline numbers useful for comparison?]

### 8.3 What to cite in your report
[Map specific findings to report sections:
- **Introduction:** cite for [specific motivation/claim]
- **Literature review:** cite as [category — e.g., "end-to-end approach"]
- **Methodology:** cite for [specific technique you're borrowing]
- **Results:** cite their numbers as [baseline comparison]
- **Discussion:** cite for [specific limitation or future direction]
]

## 9. What to use in your mid-report

[This section is specifically for the mid-report deliverable. Include:]

### Citable claims for your mid-report
[3–5 specific sentences you could write in your mid-report that cite this paper.
Example: "RawNet2 [ref] demonstrated that end-to-end models operating on 
raw waveforms can achieve competitive performance (EER: X.XX%) on the 
ASVspoof 2019 LA evaluation set, eliminating the need for handcrafted 
feature extraction."]

### Where in your mid-report structure
[Map to the expected ICML-format sections:
- Abstract: [relevant or not, and what to mention]
- Introduction: [what claim this paper supports]
- Literature Review: [which subsection this belongs in]
- Proposed Methodology: [what technique from this paper you're using]
- Preliminary Results: [if their baseline numbers are useful here]
]

---

## Glossary

[Define 5–10 technical terms from the paper that a reader might not know.
Format: **Term** — one-sentence definition in plain English.]
```

---

## Writing Guidelines

Follow these rules when filling the template:

### Language
- Write at an undergraduate level. No unexplained jargon.
- If you must use a technical term, define it inline on first use.
- Prefer concrete examples over abstract descriptions.
- "This model takes a 4-second audio clip and outputs a score between 0 and 1" is better than "The system processes variable-length input sequences and produces a probabilistic output."

### Accuracy
- Do not invent or hallucinate information. If a detail is not in the paper, say "not specified in the paper."
- Reproduce numbers exactly from the paper. Do not round unless the paper rounds.
- If the paper reports multiple configurations, pick the best-performing one for the main results and note alternatives existed.

### Structure
- Every section in the template is mandatory. If a section doesn't apply (e.g., no ablation study), write "The paper does not include ablation studies" rather than skipping the section.
- Keep the total output between 800–1500 words (excluding tables and glossary).

### Tables
- When reproducing results tables, include only the rows and columns that matter. A 20-row table should be trimmed to 5–8 rows showing the key comparison.
- Always include column headers and units.

## Step 4: Present the File

After creating the markdown file:
1. Save to `/mnt/user-data/outputs/` with a descriptive filename
2. Use `present_files` to share with the user
3. Provide a 2-sentence summary of the paper's relevance to their project

## Handling Multiple Papers

If the user uploads multiple papers at once:
- Process each paper separately into its own markdown file
- After all files are created, generate an additional `_comparison.md` file that cross-references the papers: shared methods, conflicting findings, and how they build on each other
- Present all files together

## Edge Cases

- **Survey papers**: Skip section 4 (methodology) and instead list the taxonomy/categorization the survey proposes. Expand section 3 (prior work) into a full breakdown of the categories.
- **Challenge/dataset papers**: Skip section 4 and instead describe the dataset specification in detail (size, splits, attack types, recording conditions, metadata). Expand section 5 to cover the challenge results.
- **Papers not in English**: Note the language. If you can read it, proceed. If not, inform the user.
- **Papers with poor PDF extraction**: Rasterize key pages and read visually. Note in the output that some details may be approximate due to extraction quality.

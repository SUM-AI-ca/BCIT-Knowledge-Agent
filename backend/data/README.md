# Corpus — BCIT website content

**© British Columbia Institute of Technology**

The `.txt` files in this directory are copies of pages from the British Columbia
Institute of Technology's public website (bcit.ca), covering the September 2024
intake onwards. They exist to ground the retrieval step of this project's
chatbot; each file preserves the page text as published.

**11,129 files, 100,515 chunks** at the current chunk settings:

| Directory | Files | Notes |
|---|---|---|
| `course/` | 7,060 | catalogue pages, one per course |
| `course_outline/` | 3,262 | term outlines — **latest term per course only**, so retrieval cannot surface two terms' instructors or dates side by side |
| `program/` | 529 | program pages |
| `admission/` | 61 | requirements, fees, English proficiency, deadlines |
| `international_students/`, `bcitsa/`, `accessibility/`, `about/`, and 8 more | 217 | student life, services, campus, associations |

Two conventions here are load-bearing for the indexer, not incidental:

- **Outline filenames** are `DEPT_NUM_TERM.txt` (`COMP_1510_202610.txt`).
  `build_pgvector.py` parses course code and term from the filename rather than
  from the page, and the retrieval-time entity index is built the same way.
  Apprenticeship codes such as `AATE 1GAP` are why the number part is not
  purely numeric.
- **The first line of every file is `URL: …`**, which becomes the citation
  metadata the answer's Sources section is built from.

Renaming files or dropping that header silently breaks metadata extraction and
citations.

Reproduced for **informational, non-commercial purposes only**, unmodified, with
the BCIT copyright notice above. BCIT may revoke this permission at any time, in
which case this content will be removed.

**bcit.ca is the authoritative source.** These copies are a point-in-time
snapshot — programs, tuition, dates, and admission requirements change, and a
file here can be out of date. Do not rely on it for a decision without checking
the live page.

This project is not affiliated with, endorsed by, or sponsored by BCIT. See
[`NOTICE`](../../NOTICE) at the repository root for the full statement.

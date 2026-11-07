---
name: html-tech-doc
description: Generate a polished, self-contained single-file HTML technical document — design/architecture decision docs, review or findings reports, runbooks, onboarding guides, or any technical write-up the user wants as a standalone .html page. Ships a fixed design system (sticky sidebar TOC, gradient hero, cards, pill constraints, dark-header tables, accordions, severity badges, callouts, numbered steps, hover-tooltip flow diagrams, inline-SVG architecture diagrams, file trees) with automatic dark mode and no JavaScript or external dependencies. Use when the user asks for an HTML doc, design doc page, findings report page, runbook page, or "write this up as HTML".
---

# HTML Tech Doc

Produce one `.html` file that opens directly in a browser: no build step, no JS, no CDN, no web
fonts. Every visual decision is already made in `assets/template.html`. Your job is the
content and the right components, not new styling.

## Workflow

1. **Collect the content first.** Read the sources (code, notes, diffs, findings) and draft the
   section outline before touching HTML. Aim for 4–8 sections. Each `<h2>` gets one idea.
2. **Copy the template.** Read `assets/template.html` and use it as the starting file, keeping the
   entire `<style>` block **verbatim**. Don't trim unused CSS rules; the block is small and keeping
   it whole lets later edits add components without a style hunt.
3. **Replace the demo content.** Swap the `<title>`, hero, sidebar links, sections and footer.
   Delete demo components you don't use. Don't leave placeholder text like "Project Name" or
   "YYYY-MM-DD" in the output.
4. **Rebrand only if asked.** Override the `Brand` token group at the top of `:root` (see
   *Theming*). Never hardcode a color inside a component rule or a `style=""` attribute other
   than the legend's `--swatch`.
5. **Verify.** Run through the checklist at the bottom, then tell the user the absolute file path
   (and suggest `open <path>` on macOS).

Save where the user says. Otherwise put it at `docs/html/<kebab-name>.html` in the current repo,
or in the scratchpad directory outside a repo.

## Page skeleton (do not change the structure)

```
div.page                       grid: sidebar + main; collapses to one column under 800px
├── nav.sidebar > .sidebar-inner > .sidebar-title + <a href="#id">…   one link per section
└── main.main
    ├── header.hero            h1, p.subtitle, optional .meta row, optional .product-badge
    ├── section#id > h2 …      one per sidebar link; ids are kebab-case and match exactly
    └── footer.footer          doc name · type · updated date
```

## Component catalog: pick by purpose

| Need | Component | Notes |
|---|---|---|
| Doc title + status | `.hero` with `.meta` spans | Keep the subtitle to one sentence. Use `.product-badge` for a short key or version, set in mono. |
| Hard constraints / key facts at a glance | `.constraints > .constraint` | Pills formatted `<strong>Label:</strong> value`. 3–8 of them, each under 6 words. |
| A grouped block of prose | `.card` | Optional `.card-header` row for a badge + h3. |
| Severity-ranked findings | `.card` + `.card-header` + `.badge-*` | Order: `blocking` → `warning` → `info` → `safe`. One finding per card: what, impact, fix. |
| Aside, warning, open question | `.callout.note / .warn / .danger / .ok` | Starts with `span.callout-title`. Use sparingly, no more than 2 per section. |
| Comparisons, decision matrices, inventories | `.table-wrap > table` | Always wrap it so wide tables scroll on mobile. Short header labels; the CSS uppercases them. |
| Rejected alternatives, long detail, FAQs | `details > summary` | Keeps the main read short. Put the conclusion in the summary text. |
| Ordered procedure | `ol.steps > li` | Starts with `<strong>Step title</strong>`, then the text. A `<pre>` inside a step is fine. |
| Code / commands | `pre > code`, inline `code` | Escape `<`, `>` and `&` in code. |
| Directory layout | `div.tree` | Wrap names in `span.dir` / `span.file` / `span.comment`. The content is whitespace-sensitive, so keep the `div` flush-left with no indentation inside it. |
| A linear request/data path | `.flow-diagram` | See *Flow diagrams*. |
| A system topology with boundaries | `.arch-diagram` SVG + `.arch-legend` | See *Architecture diagrams*. |

Node role classes are shared by both diagram types. Use them consistently:
`platform` (managed or third-party infrastructure), `compute` (code you own), `storage` (data
stores and queues of record), `external` (users and outside systems).

## Flow diagrams

Vertical chain: `.flow-node.<role>` → `.flow-arrow` → `.flow-label` → next node.
- Give every node `tabindex="0"` and a `<span class="tooltip">` holding one short detail line, so
  keyboard users can see the tooltip too.
- Labels on arrows name *what moves* (the endpoint, message or payload), not the verb "calls".
- For a fork, use `.flow-split > .flow-branch` (two branches, three at most); each branch starts
  with its own `.flow-arrow`.
- Keep it to 8 nodes or fewer. If it's bigger, it's an architecture diagram.

## Architecture diagrams (inline SVG)

- Copy the template's `<svg>` scaffold: `viewBox`, `role="img"`, `<title>`/`<desc>` linked with
  `aria-labelledby`, and the `#arch-arrow` marker in `<defs>`. Give the marker a unique id if
  the page has more than one diagram, and update `url(#…)` in a local class to match.
- **Draw order matters:** zones first, then edges, then nodes, so lines tuck under boxes.
- **Zones:** `rect.zone-outer` (account/cloud boundary) and `rect.zone-inner` (network or
  trust boundary) with `rx` 10–12. Each gets a `text.zone-label` at an inside corner that no edge
  crosses.
- **Nodes:** `rect.<role>` with `rx="6"`, height 56–72, width 120–170. Text is centered with
  `text-anchor="middle"`: `text.t1` (name) at `y + 24..26`, `text.t2` (one-line detail) at
  `y + 42..44`. Keep `t1` to about 16 characters and `t2` to about 20 for a 140px box.
- **Edges:** `path.edge` using only orthogonal segments (`M x,y H x V y`). Start and end on a box
  edge midpoint. Add `.two-way` for request/response pairs and `.dashed` for async or config
  reads. Put a `text.edge-label` beside a segment, never on top of a node.
- Lay out on a 10px grid with ≥ 20px between boxes. Work out coordinates as a table first
  (node → x, y, w, h), then write the paths from those numbers. Don't eyeball them.
- Follow it with an `.arch-legend` listing only the roles actually used, with
  `style="--swatch: var(--<token>)"`.

## Theming

- **Tokens:** components reference only the *semantic* tokens (`--surface`, `--text`,
  `--accent-text`, `--rule`, `--hover`, …), never raw colors. To rebrand, change these in the
  `Brand` group: `--brand`, `--brand-dark` (hero gradient, compute nodes, step numbers, blocking
  badge), `--highlight` and `--highlight-soft` (h2/footer rules, hover, warning badge, inner zone),
  plus `--success`, `--info` and `--platform` as needed. If `--brand` changes, also set the dark
  `--accent-text`/`--link` to a lighter tint of the new brand that reaches ≥ 4.5:1 on `#1C1C1E`.
- **Dark mode** is automatic via `prefers-color-scheme`. `<html data-theme="light|dark">` forces
  one mode. The dark token block appears twice (media query and attribute); edit both together.
- Text on `--brand`, `--platform`, `--info` and `--gray-2` fills is white; text on `--highlight`
  and `--success` is `--gray-1`. Keep that pairing when swapping colors.

## Writing style inside the page

- Lead every section with its conclusion; put supporting detail in tables and `details`.
- Prefer a table to a list of more than 4 parallel items with 2+ attributes each.
- Use plain declarative sentences: no marketing tone and no emoji.
- Never include secrets, tokens, internal hostnames or personal data unless the user explicitly
  supplies them for the doc.

## Verification checklist

- [ ] The `<style>` block is the template's, unmodified apart from `Brand` token overrides.
- [ ] Each sidebar `href="#x"` has exactly one matching `section id="x"`, in the same order.
- [ ] No placeholder text is left (`Project Name`, `YYYY-MM-DD`, `Short title of the issue`).
- [ ] Every table is wrapped in `.table-wrap`; every flow node has `tabindex="0"`.
- [ ] Every SVG has `<title>` + `<desc>`, and no text overflows its box (check the character limits above).
- [ ] No `<script>`, external `<link>` or remote `src`.
- [ ] Quick structural sanity check, which should print nothing:
      `grep -nE '<script|https?://[^"]*\.(css|js)|Project Name|YYYY-MM-DD' <file>`

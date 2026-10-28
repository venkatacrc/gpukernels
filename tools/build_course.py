"""Build web/course.json from the lessons in course/*.md.

    python tools/build_course.py          (any Python 3.9+, no dependencies)

Lesson format
-------------
A front-matter block, then Markdown with fenced sections:

    ---
    id: 01
    title: Vector add three ways
    lab: labs/01_vector_add
    minutes: 45
    summary: one line shown in the sidebar
    ---
    Free Markdown (the "read" part of the lesson).

    ::: write
    What to write, in Markdown.
    :::

    ::: run
    ```bash
    python labs/01_vector_add/run.py
    ```
    :::

    ::: measure
    - key | label | reference value | unit
    :::

    ::: explain
    Hidden in the app until the learner records their measurements.
    :::

    ::: cards
    Q: question
    A: answer (may continue on following lines)
    :::

The Markdown subset: #/##/### headings, paragraphs, **bold**, *italic*,
`code`, [links](url), fenced code blocks, - and 1. lists, | tables |, > quotes.
"""
import datetime
import html
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "course"
OUT = ROOT / "web" / "course.json"
KINDS = {"write", "run", "measure", "explain", "cards", "note"}


SYMBOLS = {
    "alpha": "α", "sigma": "σ", "mu": "μ", "ell": "ℓ", "partial": "∂", "sum": "Σ", "cdot": "·",
    "times": "×", "top": "ᵀ", "le": "≤", "ge": "≥", "lfloor": "⌊", "rfloor": "⌋", "approx": "≈",
    "log": "log", "max": "max", "min": "min", ",": " ", ";": " ", "quad": "  ",
}


def math(tex):
    """The small LaTeX subset the lessons use, rendered as plain HTML (no math library offline)."""
    pos = 0

    def group():
        nonlocal pos
        while pos < len(tex) and tex[pos] == " ":
            pos += 1
        if pos < len(tex) and tex[pos] == "{":
            pos += 1
            return parse("}")
        if pos < len(tex) and tex[pos] == "\\":
            return parse(None, single=True)
        ch = tex[pos : pos + 1]
        pos += 1
        return html.escape(ch)

    def parse(stop, single=False):
        nonlocal pos
        out = []
        while pos < len(tex):
            ch = tex[pos]
            if stop and ch == stop:
                pos += 1
                break
            if ch == "\\":
                m = re.match(r"\\([A-Za-z]+|.)", tex[pos:])
                name = m.group(1)
                pos += len(m.group(0))
                if name in ("mathrm", "text", "operatorname"):
                    out.append(f'<span class="mr">{group()}</span>')
                elif name == "hat":
                    out.append(group() + "\u0302")
                elif name == "overline":
                    out.append(f'<span class="ovl">{group()}</span>')
                elif name in ("left", "right"):
                    pass
                else:
                    out.append(SYMBOLS.get(name, html.escape(name)))
            elif ch in "^_":
                pos += 1
                tag = "sup" if ch == "^" else "sub"
                out.append(f"<{tag}>{group()}</{tag}>")
            elif ch in "{}":
                pos += 1
            else:
                out.append(html.escape(ch))
                pos += 1
            if single:
                break
        return "".join(out)

    return f'<span class="math">{parse(None)}</span>'


def inline(text):
    parts = re.split(r"(`[^`]+`|\\\(.+?\\\)|\\\[.+?\\\])", text)
    out = []
    for p in parts:
        if p.startswith(("\\(", "\\[")):
            out.append(math(p[2:-2].strip()))
            continue
        if p.startswith("`") and p.endswith("`") and len(p) > 1:
            out.append(f"<code>{html.escape(p[1:-1])}</code>")
            continue
        p = html.escape(p, quote=False)
        p = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', p)
        p = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", p)
        p = re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<em>\1</em>", p)
        out.append(p)
    return "".join(out)


def markdown(md):
    lines = md.strip("\n").split("\n")
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            lang = line[3:].strip()
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            cls = f' class="lang-{lang}"' if lang else ""
            out.append(f"<pre><code{cls}>{html.escape(chr(10).join(body))}</code></pre>")
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            level = len(m.group(1)) + 1  # lesson title is h1/h2 in the app
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], [r for r in rows[1:] if not all(re.fullmatch(r":?-+:?", c) for c in r)]
            t = "<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
            t += "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
            out.append(t + "</tbody></table>")
            continue
        if re.match(r"^\s*([-*]|\d+\.)\s+", line):
            ordered = bool(re.match(r"^\s*\d+\.", line))
            items = []
            while i < len(lines) and (re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]) or
                                      (lines[i].startswith("  ") and lines[i].strip() and items)):
                if re.match(r"^\s*([-*]|\d+\.)\s+", lines[i]):
                    items.append(re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        if line.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i][1:].strip())
                i += 1
            out.append(f"<blockquote>{inline(' '.join(quote))}</blockquote>")
            continue
        if not line.strip():
            i += 1
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,4}\s|```|\||>|\s*([-*]|\d+\.)\s)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


def parse_lesson(path):
    text = path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        raise SystemExit(f"{path.name}: missing front matter")
    meta = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    body = m.group(2)
    sections, cards = [], []
    pos = 0
    pattern = re.compile(r"^::: *(\w+)\s*\n(.*?)^:::\s*$", re.S | re.M)
    for blk in pattern.finditer(body):
        kind = blk.group(1)
        if kind not in KINDS:
            raise SystemExit(f"{path.name}: unknown block '{kind}'")
        before = body[pos : blk.start()]
        if before.strip():
            sections.append({"kind": "read", "html": markdown(before)})
        content = blk.group(2)
        if kind == "measure":
            fields = []
            for line in content.splitlines():
                line = line.strip()
                if not line.startswith("-"):
                    continue
                cells = [c.strip() for c in line[1:].split("|")]
                key, label, ref = cells[0], cells[1], cells[2]
                unit = cells[3] if len(cells) > 3 else ""
                fields.append({"key": key, "label": label, "ref": float(ref) if ref else None, "unit": unit})
            sections.append({"kind": "measure", "fields": fields})
        elif kind == "cards":
            q = None
            for line in content.splitlines():
                if line.startswith("Q:"):
                    q = {"q": line[2:].strip(), "a": ""}
                    cards.append(q)
                elif line.startswith("A:") and q is not None:
                    q["a"] = line[2:].strip()
                elif q is not None and line.strip():
                    q["a"] += " " + line.strip()
        else:
            sections.append({"kind": kind, "html": markdown(content)})
        pos = blk.end()
    tail = body[pos:]
    if tail.strip():
        sections.append({"kind": "read", "html": markdown(tail)})
    lesson = {
        "id": meta["id"], "title": meta["title"], "summary": meta.get("summary", ""),
        "minutes": int(meta.get("minutes", 30)), "lab": meta.get("lab", ""), "track": meta.get("track", ""),
        "sections": sections,
    }
    for n, c in enumerate(cards):
        c.update({"id": f"{meta['id']}-{n + 1}", "lesson": meta["id"], "q": inline(c["q"]), "a": inline(c["a"].strip())})
    return lesson, cards


def main():
    lessons, cards = [], []
    for path in sorted(SRC.glob("*.md")):
        lesson, c = parse_lesson(path)
        lessons.append(lesson)
        cards.extend(c)
    course = {
        "title": "GPU Kernels to HSTU: a hands-on course",
        "built": datetime.date.today().isoformat(),
        "lessons": lessons,
        "cards": cards,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(course, indent=1, ensure_ascii=False))
    n_measure = sum(len(s["fields"]) for l in lessons for s in l["sections"] if s["kind"] == "measure")
    print(f"{len(lessons)} lessons, {len(cards)} cards, {n_measure} measurements -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

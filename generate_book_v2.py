#!/usr/bin/env python3
"""
Generate a JB2-compatible myst.yml from tutorials/materials.yml.
In-repo replacement for nmaci's generate_book.py during the JB2 pilot.

tutorials/materials.yml is kept as-is: it stores richer metadata than a bare
TOC (video links, bilibili links, slide links, tutorial counts) and is used by
multiple tools. This script translates it into the myst.yml build artifact.

Run as: python ci/generate_book_v2.py student
"""

import ast
import os
import re
import sys
import json
import yaml
from bs4 import BeautifulSoup

ORG = os.environ.get("ORG", "neuromatch")
REPO = os.environ.get("NMA_REPO", "NeuroAI_Course")
ARG = sys.argv[1]  # "student" or "instructor"


def main():
    with open("tutorials/materials.yml") as fh:
        materials = yaml.load(fh, Loader=yaml.FullLoader)

    # Pre-process intro notebook
    intro_path = "tutorials/intro.ipynb"
    if os.path.exists(intro_path):
        pre_process_notebook(intro_path)

    toc = []

    # Root entry
    toc.append({"file": "tutorials/intro.ipynb"})

    # Schedule section
    toc.append(
        {
            "title": "Schedule",
            "children": [
                {
                    "file": "tutorials/Schedule/schedule_intro.md",
                    "short_title": "Overview",
                    "children": [
                        {"file": "tutorials/Schedule/daily_schedules.md"},
                        {"file": "tutorials/Schedule/shared_calendars.md"},
                        {"file": "tutorials/Schedule/timezone_widget.md"},
                    ],
                }
            ],
        }
    )

    # Technical Help section
    toc.append(
        {
            "title": "Technical Help",
            "children": [
                {
                    "file": "tutorials/TechnicalHelp/tech_intro.md",
                    "short_title": "Overview",
                    "children": [
                        {
                            "file": "tutorials/TechnicalHelp/Jupyterbook.md",
                            "short_title": "Using Jupyterbook",
                            "children": [
                                {"file": "tutorials/TechnicalHelp/Tutorial_colab.md"},
                                {"file": "tutorials/TechnicalHelp/Tutorial_kaggle.md"},
                            ],
                        },
                        {"file": "tutorials/TechnicalHelp/Discord.md"},
                    ],
                }
            ],
        }
    )

    # Links and Policy
    toc.append(
        {
            "title": "Links & Policy",
            "children": [{"file": "tutorials/TechnicalHelp/Links_Policy.md"}],
        }
    )

    # Prerequisites
    toc.append(
        {
            "title": "Prerequisites",
            "children": [{"file": "prereqs/NeuroAI.md"}],
        }
    )

    # Build category -> [day entries] dict (preserving materials.yml order)
    categories = {}
    art_file_list = os.listdir("tutorials/Art/")

    for m in materials:
        category = m["category"]
        if category not in categories:
            categories[category] = []

        directory = f"tutorials/{m['day']}_{''.join(m['name'].split())}"

        # Write chapter_title.md for this day (same logic as JB1 script)
        title_page = f"# {m['name']}"
        art_file = [fname for fname in art_file_list if m["day"] in fname]
        if len(art_file) == 1:
            artist = art_file[0].split("-")[1].split(".")[0].replace("_", " ")
            title_page += (
                f"\n\n ````{{div}} full-width \n"
                f" <img src='../Art/{art_file[0]}' alt='art relevant to chapter contents' width='100%'> \n"
                f"```` \n\n*Artwork by {artist}*"
            )
        with open(f"{directory}/chapter_title.md", "w+") as fh:
            fh.write(title_page)

        # Build notebook children list for this day
        day_children = []
        notebook_list = []
        if os.path.exists(f"{directory}/{m['day']}_Intro.ipynb"):
            notebook_list.append(f"{directory}/{ARG}/{m['day']}_Intro.ipynb")
        notebook_list += [
            f"{directory}/{ARG}/{m['day']}_Tutorial{i + 1}.ipynb"
            for i in range(m["tutorials"])
        ]
        if os.path.exists(f"{directory}/{m['day']}_Outro.ipynb"):
            notebook_list.append(f"{directory}/{ARG}/{m['day']}_Outro.ipynb")

        for nb_path in notebook_list:
            day_children.append({"file": nb_path})
            pre_process_notebook(nb_path)

        day_children.append({"file": f"{directory}/further_reading.md"})

        summary_path = f"{directory}/{ARG}/{m['day']}_DaySummary.ipynb"
        if os.path.exists(summary_path):
            day_children.append({"file": summary_path})
            pre_process_notebook(summary_path)

        categories[category].append(
            {
                "file": f"{directory}/chapter_title.md",
                "title": f"{m['name']} ({m['day']})",
                "children": day_children,
            }
        )

    # Add Module WrapUps to their categories
    for category, chapters in categories.items():
        safe_category = category.replace(" ", "")
        wrapup_name = f"tutorials/Module_WrapUps/{safe_category}.ipynb"
        if os.path.exists(wrapup_name):
            chapters.append({"file": wrapup_name})

    # Add category parts to TOC
    for category, chapters in categories.items():
        toc.append({"title": category, "children": chapters})

    # Professional Development
    with open("projects/professional_development/prof_dev_materials.yml") as fh:
        prof_dev_materials = convert_sections_to_children(
            yaml.load(fh, Loader=yaml.FullLoader)
        )
    toc.append({"title": "Professional Development", "children": prof_dev_materials})

    # Project Booklet
    with open("projects/project_materials.yml") as fh:
        project_materials = convert_sections_to_children(
            yaml.load(fh, Loader=yaml.FullLoader)
        )
    toc.append({"title": "Project Booklet", "children": project_materials})

    # Pre-process project notebooks
    for m in project_materials:
        if m.get("title") == "Project materials":
            for project in m.get("children", []):
                pre_process_notebook(project["file"])

    # Write myst.yml (build artifact — not committed to git)
    myst_config = {
        "version": 1,
        "project": {
            "title": "Neuromatch Academy: NeuroAI",
            "github": f"https://github.com/{ORG}/{REPO}",
            "license": "CC-BY-4.0",
            "edit_url": None,  # disable: auto-computed URL gets book/ prefix from symlink
            # Global LaTeX macros for KaTeX — prevents "Undefined control sequence"
            # errors when \newcommand definitions in one cell are not visible to
            # math environments in other cells/sections during MyST rendering.
            # Key is "math" (the valid myst-frontmatter project key); values are
            # plain macro strings — KaTeX infers argument count from #1, #2, etc.
            # Single-char names (h, y, T, f) are intentionally excluded: they
            # conflict with the same letters used as plain variables throughout
            # other notebooks, causing KaTeX stack overflows in text mode.
            "math": {
                "stim": r"\mathbf{x}",
                "noisew": r"\boldsymbol{\Psi}",
                "noiser": r"\boldsymbol{\xi}",
                "targetdim": r"\mathbf{y}",
                "identity": r"\mathbf{I}",
                "weight": r"\mathbf{W}",
                "loss": r"\mathcal{L}",
                "derivative": r"\frac{d#1}{d#2}",
                "pderivative": r"\frac{\partial #1}{\partial #2}",
                "rate": r"\mathbf{r}",
                "RR": r"\mathbb{R}",
                "EE": r"\mathbb{E}",
                "brackets": r"\left(#1\right)",
                "sqbrackets": r"\left[#1\right]",
                "var": r"\mathbb{V}\mathrm{ar}\left(#1\right)",
                "pred": r"\mathbf{\hat{y}}",
                "weightout": r"\mathbf{W}^{\textrm{out}}",
                "error": r"\boldsymbol{\delta}",
                "losserror": r"\mathbf{e}",
                "backweight": r"\mathbf{B}",
            },
            "toc": toc,
        },
        "site": {
            "template": "book-theme",
            "domains": ["neuroai.neuromatch.io"],
            "nav": [],
            "actions": [{"title": "GitHub", "url": f"https://github.com/{ORG}/{REPO}"}],
            "options": {
                "logo": "tutorials/static/ai-logo.png",
                "favicon": "tutorials/static/ai-logo.png",
                "logo_text": "Neuromatch Academy: NeuroAI",
                "hide_title_block": True,  # notebook H1 stays in body; suppress duplicate
            },
        },
    }

    with open("book/myst.yml", "w") as fh:
        yaml.dump(
            myst_config,
            fh,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    print("Generated book/myst.yml")


# ---- TOC helpers ----


def convert_sections_to_children(entries):
    """Recursively rename JB1 'sections' keys to JB2 'children'."""
    if not entries:
        return entries
    result = []
    for entry in entries:
        entry = dict(entry)
        if "sections" in entry:
            entry["children"] = convert_sections_to_children(entry.pop("sections"))
        elif "children" in entry:
            entry["children"] = convert_sections_to_children(entry["children"])
        result.append(entry)
    return result


# ---- Pre-processing helpers (ported verbatim from nmaci generate_book.py) ----


def expand_latex_macros(content):
    r"""Expand custom \newcommand macros in markdown cells.

    MyST/KaTeX build-time rendering does not pick up \newcommand definitions
    from inline $...$ blocks in other cells.  This function:

    1. Removes cells that contain only \newcommand definitions.
    2. Expands all macro usages in remaining markdown cells to their LaTeX
       equivalents, so every cell is self-contained.
    3. Fixes $N%$ patterns where % is treated as a LaTeX comment inside math.

    Only macros actually defined via \newcommand in the notebook are expanded,
    so this is safe to run on any notebook.
    """

    def _parse_newcommands(src):
        """Return {macro_name: (expansion, n_args)} from \newcommand blocks.

        Uses a brace-balanced scanner to handle arbitrarily nested expansions.
        """

        def _extract_brace_group(s, pos):
            """Return (content, end_pos) for the brace group starting at pos."""
            if pos >= len(s) or s[pos] != "{":
                return "", pos
            depth, start = 0, pos
            while pos < len(s):
                if s[pos] == "{":
                    depth += 1
                elif s[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start + 1 : pos], pos + 1
                pos += 1
            return s[start + 1 :], pos

        macros = {}
        i = 0
        while i < len(src):
            idx = src.find("\\newcommand", i)
            if idx < 0:
                break
            pos = idx + len("\\newcommand")
            name, pos = _extract_brace_group(src, pos)
            if not name.startswith("\\"):
                i = idx + 1
                continue
            # Optional [n_args]
            n_args = 0
            if pos < len(src) and src[pos] == "[":
                end = src.index("]", pos)
                n_args = int(src[pos + 1 : end])
                pos = end + 1
            expansion, pos = _extract_brace_group(src, pos)
            macros[name] = (expansion, n_args)
            i = pos
        return macros

    def _expand_arg_macro(text, name, template, n_args):
        """Replace \name{a1}{a2} with template (#1->a1, #2->a2)."""
        result = []
        i = 0
        while i < len(text):
            if text[i:].startswith(name):
                after = text[i + len(name):]
                # Not a match if this is a prefix of a longer macro name
                if after and after[0].isalpha():
                    result.append(text[i])
                    i += 1
                    continue
                pos = i + len(name)
                while pos < len(text) and text[pos] in " \t\n":
                    pos += 1
                args = []
                ok = True
                for _ in range(n_args):
                    if pos >= len(text) or text[pos] != "{":
                        ok = False
                        break
                    depth, start = 0, pos
                    while pos < len(text):
                        if text[pos] == "{":
                            depth += 1
                        elif text[pos] == "}":
                            depth -= 1
                            if depth == 0:
                                args.append(text[start + 1 : pos])
                                pos += 1
                                break
                        pos += 1
                    else:
                        ok = False
                        break
                if ok and len(args) == n_args:
                    expanded = template
                    for j, arg in enumerate(args):
                        expanded = expanded.replace(f"#{j + 1}", arg)
                    result.append(expanded)
                    i = pos
                else:
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
        return "".join(result)

    def _expand_simple_macro(text, name, expansion):
        """Replace \name not followed by a letter (str.join avoids re escape issues)."""
        # Only exclude letters — subscripts like \weight_{ij} SHOULD expand \weight
        parts = re.split(re.escape(name) + r"(?![a-zA-Z])", text)
        return expansion.join(parts)

    def _strip_newcommands(src):
        """Remove all \newcommand{...}{...} definitions from a string.

        Uses a brace-balanced scanner to handle arbitrarily nested expansions.
        Also removes surrounding $ delimiters and collapses blank lines.
        """

        def _skip_brace_group(s, pos):
            """Return end position after the balanced brace group starting at pos."""
            if pos >= len(s) or s[pos] != "{":
                return pos
            depth = 0
            while pos < len(s):
                if s[pos] == "{":
                    depth += 1
                elif s[pos] == "}":
                    depth -= 1
                    if depth == 0:
                        return pos + 1
                pos += 1
            return pos

        result = []
        i = 0
        while i < len(src):
            # Look for optional leading $ then \newcommand
            if src[i] == "$" and src[i + 1 :].lstrip().startswith("\\newcommand"):
                # Consume optional $, whitespace, \newcommand
                j = i + 1
                while j < len(src) and src[j] in " \t":
                    j += 1
                if src[j:].startswith("\\newcommand"):
                    j += len("\\newcommand")
                    j = _skip_brace_group(src, j)   # {name}
                    if j < len(src) and src[j] == "[":  # optional [n]
                        j = src.index("]", j) + 1
                    j = _skip_brace_group(src, j)   # {expansion}
                    # Consume trailing $
                    while j < len(src) and src[j] in " \t":
                        j += 1
                    if j < len(src) and src[j] == "$":
                        j += 1
                    i = j
                    continue
            elif src[i:].startswith("\\newcommand"):
                j = i + len("\\newcommand")
                j = _skip_brace_group(src, j)
                if j < len(src) and src[j] == "[":
                    j = src.index("]", j) + 1
                j = _skip_brace_group(src, j)
                i = j
                continue
            result.append(src[i])
            i += 1

        cleaned = "".join(result)
        # Remove bare $ that only wrapped newcommand blocks
        cleaned = re.sub(r"^\s*\$\s*\n", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    # Collect all macros defined anywhere in the notebook
    all_macros = {}
    for cell in content["cells"]:
        if cell["cell_type"] == "markdown":
            src = "".join(cell.get("source", []))
            if "newcommand" in src:
                all_macros.update(_parse_newcommands(src))

    new_cells = []
    for cell in content["cells"]:
        if cell["cell_type"] != "markdown":
            new_cells.append(cell)
            continue

        src = "".join(cell.get("source", []))

        # Strip \newcommand definitions from the source (they don't render in MyST)
        if all_macros and "newcommand" in src:
            src = _strip_newcommands(src)
            # If the cell is now empty, drop it entirely
            if not src.strip():
                continue

        if all_macros:
            # Two passes: the second catches macros introduced by first-pass expansions
            # (e.g. \var expands to \brackets{...} which then needs expanding)
            for _ in range(2):
                # Arg macros first, longest name first to avoid prefix conflicts
                for name, (template, n_args) in sorted(
                    ((n, v) for n, v in all_macros.items() if v[1] > 0),
                    key=lambda x: len(x[0]),
                    reverse=True,
                ):
                    src = _expand_arg_macro(src, name, template, n_args)
                # Simple (0-arg) macros, longest first
                for name, (expansion, _) in sorted(
                    ((n, v) for n, v in all_macros.items() if v[1] == 0),
                    key=lambda x: len(x[0]),
                    reverse=True,
                ):
                    src = _expand_simple_macro(src, name, expansion)

        # Fix bare % inside $...$ math (e.g. $80%$ -> $80\%$) — % is a LaTeX comment
        src = re.sub(
            r"\$([^$]*\d)%([^$]*)\$",
            lambda m: f"${m.group(1)}\\%{m.group(2)}$",
            src,
        )

        cell = dict(cell)
        cell["source"] = [src]
        new_cells.append(cell)

    content = dict(content)
    content["cells"] = new_cells
    return content


def pre_process_notebook(file_path):
    if not os.path.exists(file_path):
        print(f"  Warning: {file_path} not found, skipping")
        return
    with open(file_path, encoding="utf-8") as fh:
        content = json.load(fh)
    content = open_in_colab_new_tab(content)
    content = replace_widgets(content)
    content = expand_latex_macros(content)
    content = link_hidden_cells(content)
    if ARG == "student":
        content = tag_cells_allow_errors(content)
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(content, fh, indent=1, ensure_ascii=False)


def replace_widgets(content):
    """Replace or remove ipywidget-based cells that don't render in static HTML.

    JB2/MyST does not embed widget state, so widget cells render as "Loading..."
    placeholders. This function handles three patterns:

    Video cells — detected by ``display_videos(`` + ``video_ids = [``:
      Replaced with a markdown cell using MyST {tab-set}/{tab-item}/{iframe}
      directives, which render natively in JB2.

    Slide cells — detected by ``link_id`` + ``osf.io``:
      Replaced with a markdown cell using the MyST {iframe} directive plus a
      plain download link.

    Feedback cells — detected by ``# @title Submit your feedback``:
      Removed entirely (pure UI widget, no static equivalent).
    """

    def make_myst_iframe(url, width="100%"):
        return f"```{{iframe}} {url}\n:width: {width}\n```"

    new_cells = []
    for cell in content["cells"]:
        src = "".join(cell.get("source", []))

        if cell["cell_type"] != "code":
            new_cells.append(cell)
            continue

        # --- Feedback cells: remove ---
        if "# @title Submit your feedback" in src:
            continue

        # --- Video cells: replace with MyST tab-set + iframes ---
        if "display_videos(" in src and "video_ids = [" in src:
            title_match = re.search(r"#\s*@title\s+(.*)", src)
            title = title_match.group(1).strip() if title_match else "Video"

            ids_match = re.search(r"video_ids\s*=\s*(\[.*?\])", src)
            video_ids = []
            if ids_match:
                try:
                    video_ids = ast.literal_eval(ids_match.group(1))
                except (ValueError, SyntaxError):
                    pass

            if not video_ids:
                new_cells.append(cell)
                continue

            tab_items = []
            for platform, vid_id in video_ids:
                if platform == "Youtube":
                    iframe_url = f"https://www.youtube.com/embed/{vid_id}?fs=1&rel=0"
                elif platform == "Bilibili":
                    iframe_url = f"https://player.bilibili.com/player.html?bvid={vid_id}&page=1&autoplay=0"
                else:
                    print(
                        f"  Warning: unknown video platform '{platform}' (id={vid_id}), skipping"
                    )
                    continue
                tab_items.append(
                    f":::{{tab-item}} {platform}\n{make_myst_iframe(iframe_url)}\n:::"
                )

            myst_source = "\n\n".join(
                [
                    f"**{title}**",
                    "::::{tab-set}",
                    "\n".join(tab_items),
                    "::::",
                ]
            )

            new_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [myst_source],
                }
            )
            continue

        # --- Slide cells: replace with MyST iframe + download link ---
        if "link_id" in src and "osf.io" in src:
            link_id_match = re.search(r'link_id\s*=\s*["\']([^"\']+)["\']', src)
            if not link_id_match:
                new_cells.append(cell)
                continue

            link_id = link_id_match.group(1)
            download_url = f"https://osf.io/download/{link_id}/"
            render_url = (
                f"https://mfr.ca-1.osf.io/render?url=https://osf.io/{link_id}/"
                f"?direct%26mode=render%26action=download%26mode=render"
            )
            myst_source = "\n\n".join(
                [
                    f"[Download slides]({download_url})",
                    make_myst_iframe(render_url),
                ]
            )
            new_cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [myst_source],
                }
            )
            continue

        new_cells.append(cell)

    content["cells"] = new_cells
    return content


def tag_cells_allow_errors(content):
    """Add raises-exception tag to every code cell.

    JB1 used allow_errors:true globally so execution continued past any error
    (NotImplementedError stubs, downstream NameErrors, etc.) and error output
    divs were stripped from the HTML by parse_html_for_errors.py.

    JB2 has no global allow_errors equivalent, but raises-exception on a cell
    tells MyST to continue executing subsequent cells after an error. We apply
    it to all code cells so that the behaviour matches JB1 exactly. A companion
    post-processing script (parse_build_for_errors_v2.py) then strips the error
    output divs from the built HTML before deployment.
    """
    for cell in content["cells"]:
        if cell["cell_type"] != "code":
            continue
        if "metadata" not in cell:
            cell["metadata"] = {}
        if "tags" not in cell["metadata"]:
            cell["metadata"]["tags"] = []
        if "raises-exception" not in cell["metadata"]["tags"]:
            cell["metadata"]["tags"].append("raises-exception")
    return content


def open_in_colab_new_tab(content):
    cells = content["cells"]
    if not cells or not cells[0].get("source"):
        return content
    parsed_html = BeautifulSoup(cells[0]["source"][0], "html.parser")
    for anchor in parsed_html.find_all("a"):
        anchor["target"] = "_blank"
    cells[0]["source"][0] = str(parsed_html)
    return content


def link_hidden_cells(content):
    cells = content["cells"]
    updated_cells = cells.copy()
    header_level = 1
    i_updated_cell = 0
    for i_cell, cell in enumerate(cells):
        updated_cell = updated_cells[i_updated_cell]
        if "source" not in cell or not cell["source"]:
            i_updated_cell += 1
            continue
        source = cell["source"][0]
        if source.startswith("#") and cell["cell_type"] == "markdown":
            header_level = source.count("#")
        elif source.startswith("---") and cell["cell_type"] == "markdown":
            if len(cell["source"]) > 1 and cell["source"][1].startswith("#"):
                header_level = cell["source"][1].count("#")
        if "@title" in source or "@markdown" in source:
            if "metadata" not in cell:
                updated_cell["metadata"] = {}
            if "tags" not in updated_cell["metadata"]:
                updated_cell["metadata"]["tags"] = []
            if "YouTubeVideo" in "".join(cell["source"]) or "IFrame" in "".join(
                cell["source"]
            ):
                if "remove-input" not in updated_cell["metadata"]["tags"]:
                    updated_cell["metadata"]["tags"].append("remove-input")
            else:
                if "hide-input" not in updated_cell["metadata"]["tags"]:
                    updated_cell["metadata"]["tags"].append("hide-input")
            if "@title" in source and source.split("@title")[1] != "":
                header_cell = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "#" * (header_level + 1) + " " + source.split("@title")[1]
                    ],
                }
                updated_cells.insert(i_updated_cell, header_cell)
                i_updated_cell += 1
            strings_with_markdown = [
                (i, s) for i, s in enumerate(cell["source"]) if "@markdown" in s
            ]
            if len(strings_with_markdown) == 1:
                i, md_source = strings_with_markdown[0]
                if md_source.split("@markdown")[1] != "":
                    header_cell = {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": [md_source.split("@markdown")[1]],
                    }
                    updated_cells.insert(i_updated_cell, header_cell)
                    i_updated_cell += 1
        i_updated_cell += 1
    content["cells"] = updated_cells
    return content


if __name__ == "__main__":
    main()

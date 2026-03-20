#!/usr/bin/env python3
"""
Generate a JB2-compatible myst.yml from tutorials/materials.yml.
In-repo replacement for nmaci's generate_book.py during the JB2 pilot.

tutorials/materials.yml is kept as-is: it stores richer metadata than a bare
TOC (video links, bilibili links, slide links, tutorial counts) and is used by
multiple tools. This script translates it into the myst.yml build artifact.

Run as: python ci/generate_book_v2.py student
"""

import os
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
                    "children": [
                        {
                            "file": "tutorials/TechnicalHelp/Jupyterbook.md",
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
            "toc": toc,
        },
        "site": {
            "template": "book-theme",
            "domains": ["neuroai.neuromatch.io"],
            "nav": [],
            "actions": [{"title": "GitHub", "url": f"https://github.com/{ORG}/{REPO}"}],
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


def pre_process_notebook(file_path):
    if not os.path.exists(file_path):
        print(f"  Warning: {file_path} not found, skipping")
        return
    with open(file_path, encoding="utf-8") as fh:
        content = json.load(fh)
    content = open_in_colab_new_tab(content)
    content = change_video_widths(content)
    content = link_hidden_cells(content)
    content = tag_cells_allow_errors(content)
    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(content, fh, indent=1, ensure_ascii=False)


def tag_cells_allow_errors(content):
    """Add raises-exception tag to every code cell.

    JB1 used allow_errors:true globally so execution continued past any error
    (NotImplementedError stubs, downstream NameErrors, etc.) and error output
    divs were stripped from the HTML by parse_html_for_errors.py.

    JB2 has no global allow_errors equivalent, but raises-exception on a cell
    tells MyST to continue executing subsequent cells after an error. We apply
    it to all code cells so that the behaviour matches JB1 exactly. A companion
    post-processing script (parse_html_for_errors_v2.py) then strips the error
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


def change_video_widths(content):
    for cell in content["cells"]:
        if "YouTubeVideo" in "".join(cell["source"]):
            for ind in range(len(cell["source"])):
                cell["source"][ind] = cell["source"][ind].replace("854", "730")
                cell["source"][ind] = cell["source"][ind].replace("480", "410")
        if (
            "# @title Tutorial slides\n" in cell["source"]
            or "# @title Slides\n" in cell["source"]
        ):
            slide_link = ""
            for line in cell["source"]:
                if line.startswith("link_id"):
                    slide_link = line.split('"')[1]
                    break
            download_link = f"https://osf.io/download/{slide_link}/"
            render_link = (
                f"https://mfr.ca-1.osf.io/render?url=https://osf.io/{slide_link}/"
                f"?direct%26mode=render%26action=download%26mode=render"
            )
            cell["source"] = [
                "# @markdown\n",
                "from IPython.display import IFrame\n",
                "from ipywidgets import widgets\n",
                "out = widgets.Output()\n",
                "with out:\n",
                f'    print(f"If you want to download the slides: {download_link}")\n',
                f'    display(IFrame(src=f"{render_link}", width=730, height=410))\n',
                "display(out)",
            ]
    return content


if __name__ == "__main__":
    main()

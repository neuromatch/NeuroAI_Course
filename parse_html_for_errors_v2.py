#!/usr/bin/env python3
"""
Post-process JB2-built HTML to strip error output divs.

JB1 equivalent: nmaci/scripts/parse_html_for_errors.py
JB2 difference: MyST uses different CSS classes for cell output containers.

JB1 class:  "cell_output docutils container"
JB2 classes tried (in order):
  - "cell_output"           (MyST book-theme)
  - "output"                (fallback)
  - any <div> containing the error text (last resort)

Run as: python parse_html_for_errors_v2.py student
"""

import os
import sys
import yaml
from bs4 import BeautifulSoup

ARG = sys.argv[1]  # "student" or "instructor"

ERROR_STRINGS = ["NotImplementedError", "NameError"]


def main():
    with open("tutorials/materials.yml") as fh:
        materials = yaml.load(fh, Loader=yaml.FullLoader)

    html_directory = "book/_build/html/"
    total_removed = 0

    for m in materials:
        name = f"{m['day']}_{''.join(m['name'].split())}"

        notebook_paths = []
        if os.path.exists(f"tutorials/{name}/{m['day']}_Intro.ipynb"):
            notebook_paths.append(
                f"{html_directory}/tutorials/{name}/{ARG}/{m['day']}_Intro.html"
            )
        notebook_paths += [
            f"{html_directory}/tutorials/{name}/{ARG}/{m['day']}_Tutorial{i + 1}.html"
            for i in range(m["tutorials"])
        ]
        if os.path.exists(f"tutorials/{name}/{m['day']}_Outro.ipynb"):
            notebook_paths.append(
                f"{html_directory}/tutorials/{name}/{ARG}/{m['day']}_Outro.html"
            )

        for html_path in notebook_paths:
            if not os.path.exists(html_path):
                print(f"  Warning: {html_path} not found, skipping")
                continue

            with open(html_path, encoding="utf-8") as f:
                contents = f.read()

            parsed_html = BeautifulSoup(contents, features="html.parser")
            removed = strip_error_divs(parsed_html)
            total_removed += removed

            # Put solution figures in center (matches JB1 behaviour)
            for img in parsed_html.find_all("img", alt=True):
                if img["alt"] == "Solution hint":
                    img["align"] = "center"
                    img["class"] = "align-center"

            with open(html_path, "w", encoding="utf-8") as f:
                f.write(str(parsed_html))

            if removed:
                print(
                    f"  Stripped {removed} error div(s) from {os.path.basename(html_path)}"
                )

    print(f"Done. Removed {total_removed} error output div(s) total.")


def strip_error_divs(parsed_html):
    """Remove output divs that contain NotImplementedError or NameError text.

    Tries JB1's class first, then JB2/MyST class names, then a broad sweep.
    Returns the number of divs removed.
    """
    removed = 0

    # JB1 class (sphinx/docutils)
    candidates = parsed_html.find_all(
        "div", {"class": "cell_output docutils container"}
    )

    # JB2/MyST book-theme output wrapper
    if not candidates:
        candidates = parsed_html.find_all("div", {"class": "cell_output"})

    # Broader fallback: any <div> that directly wraps an error traceback
    if not candidates:
        candidates = parsed_html.find_all("div", class_=lambda c: c and "output" in c)

    for div in candidates:
        if any(err in str(div) for err in ERROR_STRINGS):
            div.decompose()
            removed += 1

    return removed


if __name__ == "__main__":
    main()

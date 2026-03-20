#!/usr/bin/env python3
"""
Post-process JB2-built HTML to strip error output divs.

JB1 equivalent: nmaci/scripts/parse_html_for_errors.py
JB2 difference: MyST wraps error output in:
  <div data-name="outputs-container">
    <div data-name="safe-output-error">
      <pre class="myst-jp-error-output">...</pre>
    </div>
  </div>

We target the inner div[data-name="safe-output-error"] to detect which
outputs-container holds an error, then decompose the parent
div[data-name="outputs-container"] so nothing is left behind.

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

    JB2/MyST error output structure:
      <div data-name="outputs-container">
        <div data-name="safe-output-error">
          <pre class="myst-jp-error-output">...</pre>
        </div>
      </div>

    We find the inner error div, check it contains a known error string, then
    decompose the parent outputs-container (so no empty wrapper is left).
    Returns the number of containers removed.
    """
    removed = 0

    error_divs = parsed_html.find_all("div", attrs={"data-name": "safe-output-error"})
    for error_div in error_divs:
        if any(err in str(error_div) for err in ERROR_STRINGS):
            # Walk up to the outputs-container wrapper and remove the whole thing
            parent = error_div.find_parent(
                "div", attrs={"data-name": "outputs-container"}
            )
            if parent:
                parent.decompose()
            else:
                error_div.decompose()
            removed += 1

    return removed


if __name__ == "__main__":
    main()

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

JB2/MyST flattens output into slug-based directories rather than mirroring
the input path structure.  We therefore scan every index.html under the
build output tree instead of constructing paths from materials.yml.

Run as: python parse_html_for_errors_v2.py student
"""

import os
import sys
from bs4 import BeautifulSoup

sys.argv[1]  # "student" or "instructor" — accepted but not used (kept for compat)

ERROR_STRINGS = ["NotImplementedError", "NameError"]

HTML_ROOT = "book/_build/html"


def main():
    total_removed = 0
    files_touched = 0

    if not os.path.isdir(HTML_ROOT):
        print(
            f"ERROR: HTML output directory not found: {HTML_ROOT!r} (cwd={os.getcwd()!r})"
        )
        sys.exit(1)

    all_index_files = []
    for dirpath, _dirnames, filenames in os.walk(HTML_ROOT):
        for fname in filenames:
            if fname == "index.html":
                all_index_files.append(os.path.join(dirpath, fname))
    print(f"Found {len(all_index_files)} index.html files under {HTML_ROOT}")

    for dirpath, _dirnames, filenames in os.walk(HTML_ROOT):
        for fname in filenames:
            if fname != "index.html":
                continue
            html_path = os.path.join(dirpath, fname)

            with open(html_path, encoding="utf-8") as f:
                contents = f.read()

            parsed_html = BeautifulSoup(contents, features="html.parser")
            removed = strip_error_divs(parsed_html)

            # Put solution figures in center (matches JB1 behaviour)
            for img in parsed_html.find_all("img", alt=True):
                if img["alt"] == "Solution hint":
                    img["align"] = "center"
                    img["class"] = "align-center"

            if removed:
                total_removed += removed
                files_touched += 1
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(str(parsed_html))
                print(f"  Stripped {removed} error div(s) from {html_path}")

    print(
        f"Done. Removed {total_removed} error output div(s) from {files_touched} file(s)."
    )


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

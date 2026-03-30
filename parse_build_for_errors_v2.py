#!/usr/bin/env python3
"""
Post-process JB2-built output to strip student exercise error outputs.

JB1 equivalent: nmaci/scripts/parse_html_for_errors.py

JB2/MyST renders pages as a React SPA. The browser ignores the static HTML and
re-renders everything from window.__remixContext, which is populated from the
per-page .json files in book/_build/html/.  Stripping the static HTML alone has
no effect — we must strip the error outputs from the .json mdast trees.

Error output structure in the page JSON (e.g. w1d2-tutorial2.json):
  mdast.children[N].children[M]  (type='outputs')
    └── children[K]               (type='output')
          jupyter_data: {output_type: 'error', ename: 'NotImplementedError', ...}

We walk every .json file, find 'output' nodes whose jupyter_data.ename matches
our error list, remove them from their parent 'outputs' node, and also remove
the 'outputs' node entirely if it becomes empty.

Run as: python parse_html_for_errors_v2.py student
"""

import json
import os
import sys

sys.argv[1]  # "student" or "instructor" — accepted but not used (kept for compat)

ERROR_NAMES = {"NotImplementedError", "NameError"}

HTML_ROOT = "book/_build/html"


def main():
    if not os.path.isdir(HTML_ROOT):
        print(
            f"ERROR: HTML output directory not found: {HTML_ROOT!r} (cwd={os.getcwd()!r})"
        )
        sys.exit(1)

    json_files = []
    for dirpath, _dirnames, filenames in os.walk(HTML_ROOT):
        for fname in filenames:
            # page data files: slug.json (not index.html)
            if fname.endswith(".json") and fname != "myst.xref.json":
                json_files.append(os.path.join(dirpath, fname))

    print(f"Found {len(json_files)} page JSON files under {HTML_ROOT}")

    total_removed = 0
    files_touched = 0

    for json_path in json_files:
        with open(json_path, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        mdast = data.get("mdast")
        if not mdast:
            continue

        removed = strip_error_outputs(mdast)

        if removed:
            total_removed += removed
            files_touched += 1
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"))
            print(f"  Stripped {removed} error output(s) from {json_path}")

    print(
        f"Done. Removed {total_removed} error output(s) from {files_touched} file(s)."
    )


def strip_error_outputs(node):
    """Recursively walk the mdast tree and remove error output nodes.

    Targets 'outputs' nodes (type='outputs') that contain one or more
    'output' children with jupyter_data.ename in ERROR_NAMES.

    Returns count of individual error output nodes removed.
    """
    removed = 0

    if not isinstance(node, dict):
        return 0

    children = node.get("children")
    if isinstance(children, list):
        new_children = []
        for child in children:
            if isinstance(child, dict) and child.get("type") == "outputs":
                # Filter out error outputs from this outputs node
                kept, n = filter_error_outputs(child)
                removed += n
                if kept:  # only keep the outputs node if it still has children
                    new_children.append(child)
                # else: drop the now-empty outputs node entirely
            else:
                removed += strip_error_outputs(child)
                new_children.append(child)
        node["children"] = new_children

    return removed


def filter_error_outputs(outputs_node):
    """Remove error output children from an 'outputs' node in-place.

    Returns (has_remaining_children, count_removed).
    """
    removed = 0
    children = outputs_node.get("children", [])
    new_children = []

    for child in children:
        if not isinstance(child, dict):
            new_children.append(child)
            continue
        if child.get("type") != "output":
            new_children.append(child)
            continue
        jd = child.get("jupyter_data", {})
        if (
            isinstance(jd, dict)
            and jd.get("output_type") == "error"
            and jd.get("ename") in ERROR_NAMES
        ):
            removed += 1
        else:
            new_children.append(child)

    outputs_node["children"] = new_children
    return bool(new_children), removed


if __name__ == "__main__":
    main()

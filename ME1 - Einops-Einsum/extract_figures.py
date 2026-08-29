"""Extract the plots produced by me1_einops_cnn.ipynb into figures/.

The READMEs embed these plots so the results are visible on GitHub without
opening the notebook, which means the files have to be refreshed whenever the
notebook is re-run. Running this script afterwards keeps the two in step.

    python extract_figures.py           # write figures/*.png
    python extract_figures.py --check   # report drift, write nothing

Each figure takes its filename from a `figure:<name>` tag on the metadata of the
cell that produces it, so reordering or editing cells never silently renames a
file that a README links to. An image output on an untagged cell still gets
extracted, as figure_<n>, and is reported so the tag can be added.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "me1_einops_cnn.ipynb"
FIGURES = HERE / "figures"

# Ordered by preference: a cell that carries several representations of the same
# figure should yield one file, and PNG is what the READMEs reference.
MIME_SUFFIX = [("image/png", ".png"), ("image/jpeg", ".jpg"), ("image/svg+xml", ".svg")]


def figure_outputs(notebook):
    """Yield (filename, payload) for every image output in the notebook."""
    untagged = 0
    for cell in notebook["cells"]:
        tags = cell.get("metadata", {}).get("tags", [])
        tag = next((t.split(":", 1)[1] for t in tags if t.startswith("figure:")), None)
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            for mime, suffix in MIME_SUFFIX:
                if mime not in data:
                    continue
                payload = data[mime]
                if isinstance(payload, list):
                    payload = "".join(payload)
                if mime == "image/svg+xml":
                    blob = payload.encode("utf-8")
                else:
                    blob = base64.b64decode(payload)
                if tag:
                    name = tag
                else:
                    untagged += 1
                    name = "figure_%d" % untagged
                    print("  ! untagged image output, writing %s%s. Add a "
                          "'figure:<name>' cell tag to pin the filename." % (name, suffix))
                yield name + suffix, blob
                break  # one file per output, in MIME_SUFFIX preference order


def check(figures):
    """Report any difference between figures/ and the notebook's current outputs."""
    drift = []
    for name, blob in figures:
        path = FIGURES / name
        if not path.exists():
            drift.append("%s is missing from figures/" % name)
        elif path.read_bytes() != blob:
            drift.append("%s differs from the notebook's current output" % name)

    expected = {name for name, _ in figures}
    on_disk = sorted(FIGURES.glob("*")) if FIGURES.is_dir() else []
    for path in on_disk:
        if path.is_file() and path.name not in expected:
            drift.append("%s is in figures/ but the notebook no longer produces it" % path.name)

    for problem in drift:
        print("  ! %s" % problem)
    if drift:
        print("%d figure(s) checked, %d problem(s). Run without --check to refresh."
              % (len(figures), len(drift)))
    else:
        print("%d figure(s) checked, all match the notebook." % len(figures))
    return 1 if drift else 0


def extract(figures):
    """Write every figure to figures/, reporting which ones actually changed."""
    FIGURES.mkdir(exist_ok=True)
    changed = 0
    for name, blob in figures:
        path = FIGURES / name
        if path.exists() and path.read_bytes() == blob:
            status = "unchanged"
        else:
            status = "wrote"
            changed += 1
            path.write_bytes(blob)
        print("  %-9s figures/%s (%s bytes)" % (status, name, format(len(blob), ",")))
    print("%d figure(s) from %s, %d updated." % (len(figures), NOTEBOOK.name, changed))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="compare figures/ against the notebook and exit 1 on any drift")
    args = parser.parse_args(argv)

    if not NOTEBOOK.exists():
        sys.exit("notebook not found: %s" % NOTEBOOK)

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    figures = list(figure_outputs(notebook))
    if not figures:
        sys.exit("the notebook holds no image outputs. Run it before extracting figures.")

    return check(figures) if args.check else extract(figures)


if __name__ == "__main__":
    raise SystemExit(main())

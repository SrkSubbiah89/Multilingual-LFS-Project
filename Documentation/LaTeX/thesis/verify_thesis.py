"""Check active thesis inputs, references, bibliography and compilation log.

Run with Python 3.10+; no third-party packages required. This checks document
structure, not scientific validity or compliance with a university template.
"""
from collections import Counter
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
errors = []
warnings = []
active = []
visited = set()


def uncomment(text):
    return re.sub(r"(?m)(?<!\\)%.*$", "", text)


def visit(path):
    path = path.resolve()
    if path in visited:
        return
    visited.add(path)
    if not path.is_file():
        errors.append(f"Missing input: {path}")
        return
    content = uncomment(path.read_text(encoding="utf-8-sig"))
    active.append((path, content))
    for filename in re.findall(r"\\input\s*(?:\{([^}]+)\}|([^\s{}]+))", content):
        name = filename[0] or filename[1]
        target = path.parent / name
        if not target.suffix:
            target = target.with_suffix(".tex")
        visit(target)


visit(HERE / "report.tex")
all_text = "\n".join(content for _, content in active)
labels = re.findall(r"\\label\{([^}]+)\}", all_text)
for label, count in Counter(labels).items():
    if count > 1:
        errors.append(f"Duplicate label: {label}")
references = re.findall(r"\\(?:ref|pageref|eqref|autoref)\*?\{([^}]+)\}", all_text)
for label in sorted(set(references) - set(labels)):
    errors.append(f"Undefined label: {label}")

bib = uncomment((HERE / "bib.bib").read_text(encoding="utf-8-sig"))
bib_keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib)
for key, count in Counter(bib_keys).items():
    if count > 1:
        errors.append(f"Duplicate bibliography key: {key}")
cites = set()
for group in re.findall(r"\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}", all_text):
    cites.update(key.strip() for key in group.split(","))
for key in sorted(cites - set(bib_keys)):
    errors.append(f"Undefined citation: {key}")
for key in sorted(set(bib_keys) - cites):
    warnings.append(f"Uncited bibliography entry: {key}")

for path, content in active:
    stack = []
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", content):
        kind, env = match.groups()
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            errors.append(f"Mismatched environment in {path.name}: {env}")
    if stack:
        errors.append(f"Unclosed environments in {path.name}: {stack}")
    for filename in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", content):
        if not (path.parent / filename).is_file():
            errors.append(f"Missing graphic in {path.name}: {filename}")

if "\\appendix" not in all_text:
    errors.append("Appendices have no appendix numbering command")

log_path = HERE / "build/report.log"
if log_path.is_file():
    log = log_path.read_text(encoding="utf-8", errors="replace")
    for line in log.splitlines():
        if ("undefined" in line.lower() and ("Reference" in line or "Citation" in line)) or "Missing character:" in line:
            errors.append(line.strip())
        if line.startswith("!"):
            errors.append(line.strip())
        if "Overfull \\" in line:
            warnings.append(line.strip())
else:
    warnings.append("No build/report.log; compilation has not been checked")

print(f"Active files: {len(active)}; labels: {len(labels)}; cited references: {len(cites)}")
for warning in warnings:
    print("WARNING:", warning)
for error in errors:
    print("ERROR:", error)
print(f"Result: {len(errors)} error(s), {len(warnings)} warning(s)")
sys.exit(1 if errors else 0)

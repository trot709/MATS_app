"""
Assemble the Lynch et al. blackmail scenario prompt and dump it to JSON.

generate_blackmail_rollouts.py already does this, but importing it pulls in classifiers and
API clients we do not need. This replicates only the template assembly -- the same functions,
reading the same files -- so the prompt is identical to the one behind the released rollouts.

Run from thought-branches/blackmail/ (paths are relative, as in the original):

    python build_blackmail_prompt.py --out blackmail_prompt.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from string import Template

import _paths  # noqa: F401  -- resolves the agentic-misalignment checkout under mats/

parser = argparse.ArgumentParser()
parser.add_argument("--urgency_type", type=str, default="replacement",
                    help="Which threat template: replacement | restriction | none")
parser.add_argument("--out", type=str, default=str(_paths.DATA_DIR.parent / "blackmail_prompt.json"))
args = parser.parse_args()

# The original code hard-codes Path("agentic-misalignment") relative to blackmail/, but the
# checkout can also sit one level up in thought-branches/. Accept either, and require that it
# actually contains the templates -- an EMPTY directory of the right name is the confusing
# case, since it satisfies .exists() and then fails on import.
def find_agentic_misalignment():
    here = Path(__file__).resolve().parent
    # _paths.AGENTIC_MISALIGNMENT (mats/thought-branches/agentic-misalignment) is the real
    # checkout; the here/here.parent fallbacks keep the script runnable if copied elsewhere.
    for candidate in (_paths.AGENTIC_MISALIGNMENT,
                      here / "agentic-misalignment",
                      here.parent / "agentic-misalignment",
                      Path("agentic-misalignment")):
        if (candidate / "templates" / "system_prompt_templates.py").exists():
            return candidate
    raise FileNotFoundError(
        "No populated agentic-misalignment checkout found. Looked in:\n"
        f"  {here / 'agentic-misalignment'}\n"
        f"  {here.parent / 'agentic-misalignment'}\n"
        "Clone it with:\n"
        "  git clone https://github.com/anthropic-experimental/agentic-misalignment "
        f"{here / 'agentic-misalignment'}"
    )


AM_DIR = find_agentic_misalignment()
print(f"using templates from {AM_DIR}")
sys.path.insert(0, str(AM_DIR))
sys.path.insert(0, str(AM_DIR / "templates"))

from templates.system_prompt_templates import (  # noqa: E402
    BLACKMAIL_SYSTEM_PROMPT, USER_PROMPT, GOAL_VARIABLES,
)


def load_template(path):
    with open(path) as f:
        return f.read()


def fill_template(template, variables):
    missing = set(re.findall(r"\$\{([^}]+)\}", template)) - set(variables)
    if missing:
        print(f"WARNING: missing template variables: {missing}")
        for var in missing:
            variables[var] = f"[MISSING: {var}]"
    return Template(template).safe_substitute(variables)


def parse_markdown_variables(path):
    with open(path) as f:
        content = f.read()
    out = {}
    for section, payload in re.findall(r"## (.*?)\n\s*```json\s*(.*?)\s*```", content, re.DOTALL):
        key = section.strip().lower().replace(" ", "_").replace("-", "_")
        try:
            out[key] = json.loads(payload)
        except json.JSONDecodeError as exc:
            print(f"Error parsing section {section}: {exc}")
    return out


def get_variables():
    all_vars = parse_markdown_variables(AM_DIR / "templates" / "blackmail" / "variables" / "ambiguous.md")
    if len(all_vars) == 1:
        variables = list(all_vars.values())[0]
    else:
        variables = {}
        for section in all_vars.values():
            variables.update(section)
    variables.update(GOAL_VARIABLES.get("none", {}))
    # Empty for the basic (goal_type="none") scenario, as in the original.
    variables.update({"alignment_faking_warning": "", "extra_system_instructions": ""})
    return variables


def main():
    variables = get_variables()
    tdir = AM_DIR / "templates" / "blackmail" / "templates"
    email_content = (
        fill_template(load_template(tdir / "email-template-blackmail-shared.md"), variables)
        + fill_template(load_template(tdir / f"email-template-blackmail-{args.urgency_type}.md"), variables)
    )
    system_prompt = fill_template(BLACKMAIL_SYSTEM_PROMPT, variables)

    scenario = {
        "system_prompt": system_prompt,
        "user_prompt": USER_PROMPT,
        "email_content": email_content,
        "urgency_type": args.urgency_type,
    }
    with open(args.out, "w") as f:
        json.dump(scenario, f, indent=2)

    print(f"wrote {args.out}")
    print(f"  system_prompt : {len(system_prompt):6d} chars")
    print(f"  user_prompt   : {len(USER_PROMPT):6d} chars")
    print(f"  email_content : {len(email_content):6d} chars")
    print("\n--- system prompt, first 400 chars ---")
    print(system_prompt[:400])
    print("\n--- email content, first 400 chars ---")
    print(email_content[:400])


if __name__ == "__main__":
    main()

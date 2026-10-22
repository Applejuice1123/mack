"""Recipes: an instruction, a schema, file handling and an apply rule, bundled.

Built-ins ship in mack/recipes/. Users copy one to ~/MACK/recipes/<name>.yaml and
edit it; a user recipe shadows the built-in of the same name.

Option placeholders: a schema value written as "$name" is replaced by the recipe
option of that name, which `--option name=a,b,c` can override (e.g. the tag list).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .config import ROOT
from .schema import SchemaError, parse_schema

BUILTIN_DIR = Path(__file__).resolve().parent / "recipes"
USER_DIR = ROOT / "recipes"
APPLY_KINDS = ("frontmatter", "rename", "sidecar", "none")
KNOWN_KEYS = {"name", "description", "glob", "instruction", "schema", "apply", "options", "ocr",
              "default_grounded", "checks", "max_output_tokens"}


class RecipeError(ValueError):
    pass


@dataclass
class Recipe:
    name: str
    description: str
    instruction: str
    schema: Optional[dict]
    apply: dict
    glob: str = "**/*.md,**/*.txt"
    options: dict = field(default_factory=dict)
    ocr: bool = False
    default_grounded: bool = False
    checks: list = field(default_factory=list)
    max_output_tokens: Optional[int] = None
    path: Optional[Path] = None
    builtin: bool = False

    def resolved_schema(self, overrides: Optional[dict] = None, user_schema: Optional[dict] = None) -> dict:
        """The schema with option placeholders filled in (or the user's own for `extract`)."""
        if user_schema is not None:
            return user_schema
        if self.schema is None:
            raise RecipeError(f"recipe {self.name!r} needs --schema FILE")
        opts = {**self.options, **(overrides or {})}

        def fill(v):
            if isinstance(v, str) and v.startswith("$"):
                key = v[1:]
                if key not in opts:
                    raise RecipeError(f"recipe {self.name!r} uses ${key} but has no option {key!r}")
                return opts[key]
            if isinstance(v, dict):
                return {k: fill(x) for k, x in v.items()}
            if isinstance(v, list):
                return [fill(x) for x in v]
            return v

        return fill(self.schema)


def load(path: Path, builtin: bool = False) -> Recipe:
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise RecipeError(f"{path}: not valid YAML: {e}")
    if not isinstance(data, dict):
        raise RecipeError(f"{path}: a recipe must be a mapping")
    unknown = set(data) - KNOWN_KEYS
    if unknown:
        raise RecipeError(f"{path}: unknown keys {sorted(unknown)}")
    for key in ("name", "instruction", "apply"):
        if key not in data:
            raise RecipeError(f"{path}: missing {key!r}")
    apply = data["apply"] or {"kind": "none"}
    if apply.get("kind") not in APPLY_KINDS:
        raise RecipeError(f"{path}: apply.kind must be one of {', '.join(APPLY_KINDS)}")
    r = Recipe(
        name=data["name"], description=data.get("description", ""), instruction=data["instruction"],
        schema=data.get("schema"), apply=apply, glob=data.get("glob", "**/*.md,**/*.txt"),
        options=data.get("options") or {}, ocr=bool(data.get("ocr", False)),
        default_grounded=bool(data.get("default_grounded", False)), checks=data.get("checks") or [],
        max_output_tokens=data.get("max_output_tokens"), path=path, builtin=builtin,
    )
    if r.schema is not None:
        try:
            parse_schema(r.resolved_schema(), r.default_grounded)
        except SchemaError as e:
            raise RecipeError(f"{path}: {e}")
    return r


def all_recipes(user_dir: Path = USER_DIR) -> list[Recipe]:
    found: dict[str, Recipe] = {}
    for p in sorted(BUILTIN_DIR.glob("*.yaml")):
        r = load(p, builtin=True)
        found[r.name] = r
    if user_dir.exists():
        for p in sorted(user_dir.glob("*.yaml")):
            r = load(p)
            found[r.name] = r  # the user's copy wins
    return sorted(found.values(), key=lambda r: r.name)


def find(name: str, user_dir: Path = USER_DIR) -> Recipe:
    user = user_dir / f"{name}.yaml"
    if user.exists():
        return load(user)
    builtin = BUILTIN_DIR / f"{name}.yaml"
    if builtin.exists():
        return load(builtin, builtin=True)
    names = ", ".join(r.name for r in all_recipes(user_dir))
    raise RecipeError(f"no recipe named {name!r}. Available: {names}")


def parse_options(pairs: list[str]) -> dict:
    """--option tags=a,b,c  ->  {"tags": ["a", "b", "c"]}"""
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise RecipeError(f"--option must look like name=value, got {pair!r}")
        k, v = pair.split("=", 1)
        out[k.strip()] = [x.strip() for x in v.split(",") if x.strip()] if "," in v else v.strip()
    return out

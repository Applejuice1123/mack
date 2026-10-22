# Writing a recipe

A recipe bundles an instruction, an output schema, which files to read and what `mack apply`
does with the results. Copy a built-in to start:

```bash
mkdir -p ~/MACK/recipes && mack recipes show tag > ~/MACK/recipes/tag.yaml   # yours now replaces the built-in
```

## Reference

```yaml
name: tag                      # required; the --recipe name
description: One line shown by `mack recipes`.
glob: "**/*.md"                # files to pick up in a folder, comma-separated patterns
instruction: |                 # required; what to do with each file
  Choose up to three tags from the allowed list that describe this note.
schema:                        # the JSON object each file produces (null = needs --schema)
  type: object
  properties:
    tags: {type: array, items: {type: string, enum: $tags}, maxItems: 3}
    summary: {type: string, maxLength: 200, description: "one sentence"}
apply:                         # required
  kind: frontmatter            # frontmatter | rename | sidecar | none
  fields: [tags, summary]
options:                       # values for $placeholders; override with --option tags=a,b,c
  tags: [meeting, idea, bug, reference, personal]
ocr: false                     # read PDFs and images (text layer first, then OCR)
default_grounded: false        # treat every string field as x-grounded
checks:                        # extra checks after grounding
  - {kind: retention, field: cleaned, min: 0.85}
max_output_tokens: 512         # raise for recipes that rewrite whole files
```

Unknown keys are an error, so a typo cannot silently do nothing.

### Schema types

`string` (with `maxLength`, `format: date` for YYYY-MM-DD), `number`, `integer`, `boolean`,
`enum` (a list of allowed strings), and `array` of strings (with `maxItems`, optional `enum`
on the items). Every field may also come back `null`; the model is told to use `null` when the
file does not contain the answer.

### Grounding

- `x-grounded: true`: the value must appear in the file. Case and spacing are ignored, numbers
  match their written forms ($1,240.50 = 1240.5), and dates match any written form of the same date.
  A date written only as an ambiguous number (03/09/2025) is kept but flagged, because Mack
  cannot tell which day it means.
- `x-quote: true`: the model also returns `<field>_quote` (arrays: `<field>_quotes`, one per item),
  at most 25 words copied from the file, and the quote must be found in the file.

A value that fails is removed and the item is flagged. Flagged items appear first in the review
file and are not applied unless you pass `--only done+flagged`.

### Apply kinds

| kind | keys | effect |
|---|---|---|
| `frontmatter` | `fields` | adds those keys to the note's YAML frontmatter, creating it if missing; existing keys are kept (unless `mack apply --overwrite`), existing frontmatter bytes untouched |
| `rename` | `pattern` | renames the file, e.g. `"{date} {source} {title}"`; fields available: `date` (`undated` if missing), `source`, `title`, `kind`; collisions get `-2`, `-3` |
| `sidecar` | `suffix`, `template` (`summary`, `clean`, `json`, or anything else for a plain field list) | writes a new file beside the original; never overwrites |
| `none` | | results stay in the output and review files |

Any job can also write a CSV: `mack apply JOB --csv results.csv`.

## Worked example: contracts to a CSV

`~/MACK/recipes/contracts.yaml`:

```yaml
name: contracts
description: Parties, start date, renewal and notice period from contract text files.
glob: "**/*.txt,**/*.pdf"
ocr: true
instruction: |
  Find the parties to this contract, the start date, whether it renews automatically,
  and the notice period for cancellation.
schema:
  type: object
  properties:
    party_a: {type: string, x-grounded: true}
    party_b: {type: string, x-grounded: true}
    start_date: {type: string, format: date, x-grounded: true}
    auto_renews: {type: boolean}
    notice_period: {type: string, x-quote: true, description: "as written, e.g. 30 days"}
apply:
  kind: none
```

```bash
mack run --recipe contracts --limit 5 --pace full --wait ~/Contracts   # try five first
mack review last                                                       # check the flagged ones
mack run --recipe contracts --pace balanced ~/Contracts                # then all of them
mack apply last --csv ~/Desktop/contracts.csv
```

The CSV has one row per contract: `path,auto_renews,notice_period,notice_period_quote,party_a,party_b,start_date`.
`mack undo last` deletes the CSV if you have not changed it (or restores it, if you appended to an existing one).

## Tips

- Keep instructions narrow. "Pick tags from this list" works on a 4B model; "analyse this note" does not.
- Prefer `enum` and `x-grounded` fields to free text: they are the ones Mack can check.
- Always try `--limit 5 --pace full --wait` and read the review file before a long run.

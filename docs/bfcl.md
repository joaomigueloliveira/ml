# BFCL: the unseen-tools eval

Issue #7. Code: `fc_distill.bfcl`; run it with `uv run fc-eval --model <id> --split bfcl`.

xLAM test measures tools the student trained on, with new values: only 4 of its 2,000 rows call a
function that train never calls. A method that only gains there may have learned xLAM's tools and
conventions rather than tool calling. BFCL checks that with tools from a different source.

## What is included

The [Berkeley Function-Calling Leaderboard](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
v3 at revision `61fc0608cfd831fcfbbaa676ebdfef0ed963eeda`, non-live single-turn categories:

| Category | Rows | Task |
|---|---:|---|
| simple | 400 | one function offered, one call |
| multiple | 200 | several functions offered, one call |
| parallel | 200 | one function, called several times |
| parallel_multiple | 200 | several functions, several calls |

Left out: *live* (user-contributed queries with messier specs, worth adding later), *irrelevance*
(the right answer is no call, which xLAM never teaches), multi-turn (out of scope in #2), and the
Java, JavaScript, REST and SQL categories (other output formats).

Only 46 of the 1,000 rows call a function whose name also appears in xLAM train (24 of BFCL's 769
names). Several of those names have different specs in BFCL. `fc-eval` reports this count with every
run.

## How it is adapted

- **Same prompt as xLAM.** Each BFCL spec is converted to xLAM's shape. `integer` becomes `int`, an
  array of floats becomes `List[float]`, and every parameter not in `required` gets `, optional`.
  `enum` values are appended to the description, and nested dict fields are kept as `properties`. A
  model trained on xLAM then sees the familiar format, so a drop on BFCL reflects the new tools, not a
  new spec format. `format` and `maximum` constraints (6 parameters) are dropped.
- **Lists of acceptable values.** BFCL gives each gold argument a list of acceptable values, and `""`
  in the list means it may be left out. A dict value holds a list of acceptable values per key again.
  A prediction matches when its calls pair one-to-one, in any order, with the gold calls, every
  argument value is acceptable, no argument is unknown, and every argument left out is allowed to be.

## Metrics

| Metric | Meaning |
|---|---|
| `exact` | the matching rule above with xLAM's strict value comparison (`5 == 5.0`, `true != 1`, strings exact). The headline BFCL number |
| `bfcl_exact` | the same, but strings are compared the way BFCL's checker does: ignoring case, spaces and `, . / - _ * ^` |
| `names` | the right functions, each the right number of times |
| `parse_ok` | as on xLAM |

`bfcl_exact` reimplements only BFCL's string comparison, not its whole checker. BFCL also type-checks
each value against the declared type, which we don't. Our numbers are therefore close to the
leaderboard's but not identical to it, and are not reported as leaderboard scores.

## Reporting

- Each 95% interval is a half-width, in the same unit as the score (`exact_ci`).
- ADR 0001 reports a slice on its own only with at least 300 rows. So results are given **overall**,
  for **simple** (400), and for **multiple + parallel + parallel_multiple** merged (600). The four
  categories are listed too, with the 200-row ones marked `below_min_rows`.
- BFCL is an eval only, like test: no checkpoint or setting is chosen on it.

## Caveats

- BFCL has been public since 2024, so Qwen3.5 may have seen it in pretraining. That would raise every
  model a little, not one method over another.
- Most BFCL tools are maths, science and utility functions with clean specs, so it is narrower than
  xLAM.

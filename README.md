# agents

An experimental, portable agent design: one agent that runs locally on Ollama or in the cloud on AWS Bedrock and AgentCore.

The agent here is **fetcher**, built on the Strands Harness SDK one layer at a time: agent, harness,
inference, gateway and tools. It runs locally first, then on AWS.

The architecture diagrams live in [agent-stack.html](agent-stack.html).

This repository holds only the generic agent. Real jobs, traces, outputs and sessions are gitignored.

## Run it

Needs [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/download) with a model that supports tools.

```bash
ollama pull gemma4:e4b
uv run fetcher "What time is it in Tokyo?"
uv run fetcher "Go through the new posts in r/Python. For each one, tell me which Python libraries the post or its comments mention."
uv run pytest
```

gemma4:e4b peaks at about 11 GB of memory while it works, so fetcher unloads it as soon as a run ends. On a
16 GB Mac, `AGENT_MODEL=gemma4:e2b` needs about 9 GB but makes more mistakes.

Skills live in `skills/<name>/SKILL.md`; put your own in `jobs/skills/`, which git ignores. Put your own
tools in `jobs/tools/<name>.py`, which git also ignores: a file there can define `TOOLS`, a list of `@tool`
functions for the model; `transform_text(text)`, which rewrites every post, comment and feed entry
before the model reads it; and `find_groups(text)`, which lists what that rewrite marked in a text, so a
job with a link pattern can account for each one. A transform sees the whole text, and it's only cut to
what the model reads after that.

Ask for a report and the agent saves an HTML page of links, with your prompt and its criteria at the top,
to `outputs/reports/<run_id>-<title>.html`. Like traces, reports are gitignored. Each match can have a
second link: one a tool returned, or a pattern you give in the prompt with a placeholder, such as
`https://example.com/search?q=<name>`. The report takes titles and comment counts from the posts
themselves, checks the model's quotes against them, and checks every link.

`write_report` takes a flat list of string fields, and each job sees only the fields it uses. With one
link pattern in the prompt, every item has `post`, `fill`, `description` and `quote`, and the code resolves
the post and builds the link. Otherwise an item has `post` or `link`, `description`, and optionally `quote`,
`extra_link` and `extra_label`. A call that doesn't fit is refused whole, with every problem listed, so
nothing half-right gets saved. `adapt_legacy_item` in [report.py](src/fetcher/report.py) lists the older
shapes still accepted. Reports are written atomically, so a failed write leaves any earlier report alone.

There's one item per value, and a post can repeat with different values. Each value stays a record of its
own all the way to the page and the saved data, with its quote, description and links, and its second link
is labelled with the value. The page shows a post once, with each of its values under it. Rows with the
same post and value keep all their quotes, and only an exact repeat is dropped.

`left_out` says why whatever the model read isn't in `items`: a row with a source's name covers the rest of
its posts, a row with a post's id covers that post, and a row with `value` as well covers that value.
For a whole-post exclusion put its ID in `source`, not `value`. Inverted references and explicit
inclusion/exclusion conflicts are returned as validation errors before any file is written.
If an input group has several lines, each line needs an item or an explicit exclusion;
literal slashes are preserved. Quotes that repeat generated search links and values found only in a
different recorded post remain partial-report warnings, with diagnostics identifying the issue.

### Job outcomes

For a job that has to produce a report, pass `--report-required`. Its instructions then say so. When
the link-report skill is available, its full instructions are loaded before the first model call and
their content hash is recorded in the trace. When the agent returns, the command line looks for this
run's file, and if it isn't there, or it leaves something
the model read unexplained, asks the model once more to finish. The exit status tells a script what happened:

| Exit | Outcome | Meaning |
| --- | --- | --- |
| 0 | `complete` | Everything was read and explained, and nothing's in doubt. Zero matches counts. |
| 1 | `failed` | The run stopped, or none of its sources could be read. A report may still have been saved. |
| 2 | | The command line itself was wrong. |
| 3 | `partial` | The report was saved, but something is incomplete or unconfirmed. It says what at the top. |
| 4 | `report_not_created` | The job needed a report, and this run's file isn't there. |

A report is partial if a source failed or was cut short, if no source was read, if an item couldn't be
placed, if a quote, value or link couldn't be confirmed, or if something the model read has no explanation.
A valid call is always saved, and the report says what's missing; [outcome.py](src/fetcher/outcome.py) holds
this policy. An empty report only says nothing matched if everything was read and explained.

Every source gets a row of counts: what was found in it, cut off before the model read it, assessed,
submitted, rejected, rendered and left unaccounted, for its posts and for any groups a local tool marked in
them. Everything found ends up as exactly one of rendered, rejected, truncated or unaccounted, as
[accounting.py](src/fetcher/accounting.py) sets out. The sources are the ones the prompt names and the
ones the model read. Input that was cut off is the source's to report, and the model doesn't have to explain it.

`--no-verify-links` skips the one step that uses the network, so a report can be built and rendered
offline. With checks on, each link ends up `reachable`, `missing` (the host answered 404 or 410) or
`unreachable` (no answer), and Reddit's stay `unverified`. A missing link keeps its URL, with the closest
page that exists beside it. `--report-json` also saves the report's data as JSON next to its HTML, for
anything downstream that wants the items rather than the page: the same per-value records the page was
built from, each source's counts, the `left_out` rows, and separate counts of posts, values and evidence.
`schema_version` says which shape that is, and it's kept in the harness's records, not the tool's arguments.

Without logging in, Reddit allows about one request a minute. `read_subreddit` reads the post list and
then the subreddit's comment feed, 100 comments a request, so a day of a subreddit takes a minute or
two. Agents running at once on one machine take turns with each host, so they don't use up each
other's budget.

Each run writes its steps to `traces/<run_id>.jsonl`: every model call, each tool result with the tool the
model asked for and the one that ran, `artifact_saved` (with the same records as `--report-json`) or
`artifact_failed` after `write_report`, and a final `job_outcome`. Set `AGENT_MODEL` to try another Ollama model, and `OLLAMA_HOST` if Ollama isn't on `localhost:11434`.
Read tools also emit `source_snapshot` records containing their remembered source text, group spans,
visibility flags, and coverage. These local records include groups omitted from the model input by
size limits, so reported
truncation counts can be audited without fetching sources again. They stay in gitignored traces.

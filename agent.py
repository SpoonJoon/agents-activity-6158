#!/usr/bin/env python3
"""An oracle-grounded agent that translates python-semver into Rust.

    python agent.py                  # run with defaults
    python agent.py --budget 40      # cap on model calls (graded: do not raise)

    AGENT_PROVIDER=openai|anthropic|gemini  # default: whichever *_API_KEY is set
    AGENT_MODEL=<model id>           # default per provider, see MODELS

DESIGN
The agent is a ReAct loop (reason -> act -> observe). The scaffold around it
rests on three ideas:

  1. Verification is the environment's job. Every action that changes
     lib.rs is followed by an automatic build + test + differential check,
     so the model never spends a call on "now compile it". Each check has
     two splits, as in train/validation. The agent sees failing cases from a
     *fresh* random seed every time, so it cannot overfit to them (the grading
     seed is unseen too). Snapshot selection, rollback and stagnation use a
     *fixed* validation seed, so seed noise is never mistaken for progress.

  2. Context is reconstructed, not accumulated. The authoritative state
     (source on disk, latest check, best score, agent notes) is rendered into
     a digest every step (WRITE + SELECT). Old turns keep their actions but
     lose their observations ("observation masking", COMPRESS), and a sliding
     window of recent turns stays verbatim. The window shrinks adaptively to
     fit a fixed token budget.

  3. Termination is a harness decision. The model may *claim* completion, but
     only the verifier can confirm it. The harness also stops on convergence
     (100% on both splits), stagnation, action loops, and
     idling. On exit it rolls back to the best verified snapshot, so scores
     cannot go down.

Ground truth comes from the reference Python package itself: the `probe` tool
runs the same query against the oracle and the current Rust binary side by
side. The agent reads semantics from the source and the oracle; they are not
baked into the prompt.
"""
from __future__ import annotations
import argparse, collections, functools, json, os, pathlib, re, subprocess, sys, tempfile, time

HERE   = pathlib.Path(__file__).parent
RUST   = HERE / "rust"
LIB    = RUST / "src" / "lib.rs"
REF    = HERE / "reference"
PYSRC  = REF / "version.py"
LOGS   = HERE / "logs"

sys.path.insert(0, str(HERE))
from evaluate import BIN, ref, ref_parse, ref_compare, ref_bump, run_harness  # noqa: E402

REQUIRED   = ("parse", "to_string", "compare", "bump_major", "bump_minor", "bump_patch")
EVAL_N     = 150        # random cases per family, feedback split (rotating seed)
VAL_SEED   = 7          # validation split: fixed, drives selection & stopping
VAL_N      = 300
CTX_TOKENS = 30_000     # soft budget for what we send; never raised
WINDOW     = 3          # recent turns kept verbatim (shrinks to fit CTX_TOKENS)
PATIENCE   = 10         # model calls without improving the best snapshot
MAX_SMELLS = 3          # clone + to_owned + unwrap tolerated at convergence
OUT_CAP    = 40_000     # hard cap on any single tool output (chars)

_log = lambda **kw: None  # replaced in main()

# ============================================================== TODO 1
def _load_dotenv(path=HERE / ".env"):
    """Minimal KEY=VALUE loader (std only). Real environment variables win."""
    if path.exists():
        for line in path.read_text().splitlines():
            k, sep, v = line.split(" #")[0].strip().partition("=")   # drop inline comments
            if sep and k and not k.startswith("#") and v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
_load_dotenv()

PROVIDER = os.environ.get("AGENT_PROVIDER") or next(
    (p for p, k in [("gemini", "GEMINI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")]
     if os.environ.get(k)), "openai")
MODELS = {"openai": "gpt-5", "anthropic": "claude-sonnet-5", "gemini": "gemini-2.5-pro"}
MODEL  = os.environ.get("AGENT_MODEL") or MODELS[PROVIDER]

def call_model(messages: list[dict], tools: list[dict]) -> dict:
    """Send `messages` + `tools` to the configured provider.

    Internal format (provider-neutral):
        assistant: {"content": str, "tool_calls": [{"id", "name", "arguments": dict}]}
        tool:      {"id": <tool_call id>, "name", "content": str}
    Returns {"text": str | None, "tool_calls": [{"id", "name", "arguments"}]}.
    Transient failures (429 / 5xx / network) are retried; they are not model calls.
    """
    send = _anthropic if PROVIDER == "anthropic" else _openai   # gemini speaks the OpenAI protocol
    for attempt in range(4):
        try:
            return send(messages, tools)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if attempt == 3 or (status and status < 500 and status != 429):
                raise
            print(f"      (api error: {e!s:.120}; retrying)")
            time.sleep(5 * 2 ** attempt)

@functools.cache
def _client():
    if PROVIDER == "openai":
        from openai import OpenAI
        return OpenAI()                   # reads OPENAI_API_KEY
    if PROVIDER == "gemini":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["GEMINI_API_KEY"],
                      base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    import anthropic
    return anthropic.Anthropic()          # reads ANTHROPIC_API_KEY

def _loads(s: str | None) -> dict:
    try:
        return json.loads(s or "{}")
    except json.JSONDecodeError:
        return {"_unparsed": s}

def _openai(messages, tools):
    out = []
    for m in messages:
        if m["role"] == "assistant":
            msg = {"role": "assistant", "content": m["content"] or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                      "function": {"name": c["name"],
                                                   "arguments": json.dumps(c["arguments"])}}
                                     for c in m["tool_calls"]]
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    r = _client().chat.completions.create(
        model=MODEL, messages=out,
        tools=[{"type": "function", "function": t} for t in tools])
    msg = r.choices[0].message
    return {"text": msg.content,
            "tool_calls": [{"id": c.id or f"call_{time.time_ns()}_{i}", "name": c.function.name,
                            "arguments": _loads(c.function.arguments)}
                           for i, c in enumerate(msg.tool_calls or [])]}

def _anthropic(messages, tools):
    system, out = "", []
    def push(role, blocks):               # merge consecutive same-role turns
        if out and out[-1]["role"] == role:
            out[-1]["content"] += blocks
        else:
            out.append({"role": role, "content": blocks})
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            push("user", [{"type": "text", "text": m["content"]}])
        elif m["role"] == "tool":
            push("user", [{"type": "tool_result", "tool_use_id": m["id"], "content": m["content"]}])
        else:
            blocks = [{"type": "text", "text": m["content"]}] if m["content"] else []
            blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                       for c in m.get("tool_calls") or []]
            push("assistant", blocks or [{"type": "text", "text": "(no output)"}])
    r = _client().messages.create(
        model=MODEL, max_tokens=12_000, system=system, messages=out,
        tools=[{"name": t["name"], "description": t["description"],
                "input_schema": t["parameters"]} for t in tools])
    return {"text": "".join(b.text for b in r.content if b.type == "text") or None,
            "tool_calls": [{"id": b.id, "name": b.name, "arguments": b.input}
                           for b in r.content if b.type == "tool_use"]}

# ============================================================== TODO 2
def system_prompt() -> str:
    """Contract + rules + method. Deliberately *no* semver semantics: those
    come from reading the source and querying the oracle, which generalises
    and cannot silently disagree with the reference implementation."""
    return """\
You are translating a Python module into idiomatic, safe Rust. You act only through tools. \
Every reply you make costs one model call from a hard budget of 40, so batch independent \
tool calls into a single turn.

TASK
Translate reference/version.py (the python-semver package) into rust/src/lib.rs. The crate must expose exactly:
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Version { pub major: u64, pub minor: u64, pub patch: u64,
                         pub prerelease: Option<String>, pub build: Option<String> }
    pub fn parse(s: &str) -> Result<Version, String>
    pub fn to_string(v: &Version) -> String
    pub fn compare(a: &Version, b: &Version) -> std::cmp::Ordering
    pub fn bump_major(v: &Version) -> Version   // also bump_minor, bump_patch
`prerelease` and `build` hold the text after '-' and '+', or None. src/main.rs and Cargo.toml are fixed.

HARD RULES (breaking one voids the result)
- Use std only. No `unsafe`. No todo!/unimplemented!/panic!. Never call Python.
QUALITY (graded)
- Write Rust the way a Rust programmer would. Avoid .clone(), .to_owned() and .unwrap(). \
Borrow where you can, own Strings where you must, and use `?`, `match`, `ok_or`, and iterators.
- Keep a #[cfg(test)] mod tests with cases ported from reference/test_parsing.py, \
test_compare.py and test_bump.py. Port only cases for functions this crate exposes, and \
only ones the oracle confirms.

GROUND TRUTH
The Python `semver` package is the oracle. Never guess semantics. When unsure (bumps on \
prereleases, leading zeros, what is invalid, how identifiers order), call `probe`, which \
runs your queries against the oracle and your current binary side by side.

METHOD
1. Read version.py once (and grep the tests as needed). Use `remember` for facts you would \
otherwise need to re-derive, because old tool outputs are elided from your context.
2. Write a complete first draft with write_rust. Every write or edit is automatically built, \
tested, and checked against the oracle on a fresh random seed, and the report comes back as \
the tool result. Do not call `check` right after an edit.
3. Fix failures with small edit_rust patches. Probe the oracle first when a cause is unclear. \
If a change makes things worse, call revert_to_best.
4. Call finish when the check is clean. The harness verifies your claim.

The "Harness state" block at the end of each turn is authoritative: current file, latest \
check, best score, and your notes."""

# ============================================================== TODO 3
def build_context(history: list[dict], step: int) -> list[dict]:
    """[system, task] + masked old turns + verbatim recent turns + digest.

    A "turn" is one assistant message plus the tool results / nudges it
    caused, so tool calls and their results are never separated (both
    providers reject orphans). Old turns keep *what was done* and lose *what
    was seen*; anything still relevant is already in the digest or on disk.
    """
    head, turns = history[:2], []
    for m in history[2:]:
        if m["role"] == "assistant" or not turns:
            turns.append([m])
        else:
            turns[-1].append(m)
    tail = {"role": "user", "content": STATE.digest(step)}
    for w in range(WINDOW, 0, -1):
        msgs = (head + [x for t in turns[:-w] for x in map(_mask, t)]
                     + [x for t in turns[-w:] for x in t] + [tail])
        if len(json.dumps(msgs)) / 4 <= CTX_TOKENS:
            break
    return msgs

def _mask(m: dict) -> dict:
    if m["role"] == "assistant":
        return {**m, "content": _clip(m["content"], 300),
                "tool_calls": [{**c, "arguments": {k: v if len(str(v)) <= 120
                                                   else f"<{len(str(v))} chars elided>"
                                                   for k, v in c["arguments"].items()}}
                               for c in m.get("tool_calls") or []]}
    if m["role"] == "tool":
        first = (m["content"].splitlines() or [""])[0]
        return {**m, "content": _clip(first, 160) + "  [older output elided]"}
    return m

def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + " …"

# ============================================================== TODO 4
def should_stop(history: list[dict], step: int, budget: int, state: "State") -> tuple[bool, str]:
    """Success first, then budget, then failure modes."""
    if state.finished:
        return True, f"agent finished, verified by harness: {state.finished}"
    if state.report and converged(state.report):
        return True, "converged: clean build, tests pass, 100% on validation and fresh seed"
    if step >= budget:
        return True, f"budget exhausted ({budget} model calls)"
    if state.idle >= 2:
        return True, "stuck: two consecutive replies without a tool call"
    last = list(state.actions)[-3:]
    if len(last) == 3 and len(set(last)) == 1:
        return True, "stuck: identical action three times in a row"
    if state.best and step - state.best_step >= PATIENCE:
        return True, f"stagnated: no improvement on best in {PATIENCE} model calls"
    return False, ""

# ============================================================ verification
def score_key(r: dict | None) -> tuple:
    """Lexicographic: legal > differential % > own tests pass > fewer smells."""
    if not r or not r["build"]:
        return (0, 0.0, 0, 0)
    return (int(not r["violations"]), r["diff"], int(tests_ok(r)), -smells(r))

def tests_ok(r):  t = r.get("tests") or {}; return t.get("failed") == 0 and t.get("passed", 0) > 1
def smells(r):    q = r["quality"]; return q["clone_calls"] + q["to_owned_calls"] + q["unwrap_calls"]

def converged(r: dict) -> bool:
    return (r["build"] and not r["violations"] and r["diff"] == 100 and r["fresh_diff"] == 100
            and tests_ok(r) and smells(r) <= MAX_SMELLS)

class State:
    """Everything the agent needs to know that is *not* kept in its context."""
    def __init__(self):
        self.step, self.checks, self.idle, self.rejections = 0, 0, 0, 0
        self.report = None                              # latest check
        self.best, self.best_src, self.best_step = None, None, 0
        self.trace: list[str] = []                      # score history
        self.notes: list[str] = []                      # agent-written memory
        self.actions = collections.deque(maxlen=3)      # action fingerprints
        self.finished = ""

    # -- the verifier ----------------------------------------------------
    def check(self) -> str:
        self.checks += 1
        seed = 1000 + self.checks
        b = subprocess.run(["cargo", "build", "--release", "--message-format", "short"],
                           cwd=RUST, capture_output=True, text=True)
        if b.returncode:
            errs = [l for l in b.stderr.splitlines() if re.search(r"\berror\b", l)]
            r = {"build": False, "errors": errs[:25]}
        else:
            r, fresh = self._evaluate(VAL_SEED, VAL_N), self._evaluate(seed, EVAL_N)
            if r["build"]:
                r.update(seed=seed, fresh_diff=fresh.get("diff", 0),
                         failures=fresh.get("failures") or r["failures"])
                if not tests_ok(r):
                    r["test_failures"] = _test_failures()
        r["step"] = self.step
        self.report = r
        improved = score_key(r) > score_key(self.best)
        if improved:
            self.best, self.best_src, self.best_step = r, LIB.read_text(), self.step
        self.trace.append(f"s{self.step}:" + (f"{r['diff']:.1f}%" if r["build"] else "no-build")
                          + ("*" if improved else ""))
        _log(event="check", step=self.step, report=r, improved=improved)
        return self.render(r)

    def _evaluate(self, seed: int, n: int) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            p = subprocess.run([sys.executable, str(HERE / "evaluate.py"), "--seed", str(seed),
                                "--n", str(n), "--json", f.name],
                               cwd=HERE, capture_output=True, text=True)
            R = json.loads(pathlib.Path(f.name).read_text() or "{}")
        if not R.get("build") or "differential" not in R:
            return {"build": False,
                    "errors": [R.get("error") or p.stdout[-1500:]]}
        fails = p.stdout.split("First failures:")[-1] if "First failures:" in p.stdout else ""
        return {"build": True, "diff": R["differential_pct"],
                "fams": {k: round(100 * v["pass"] / max(1, v["total"]), 1)
                         for k, v in R["differential"].items()},
                "chain": R["spec_precedence_chain"], "tests": R["cargo_test"],
                "quality": R["quality"], "violations": R["violations"],
                "failures": [l.strip()[2:] for l in fails.splitlines() if l.strip().startswith("- ")]}

    def render(self, r: dict, brief: bool = False) -> str:
        if not r["build"]:
            return "BUILD FAILED\n" + "\n".join(r["errors"])
        t, q = r["tests"] or {}, r["quality"]
        fams = ", ".join(f"{k} {v}" for k, v in r["fams"].items())
        s = [f"build ok | cargo test {t.get('passed', '?')} passed, {t.get('failed', '?')} failed"
             f" | differential {r['diff']}% on validation ({fams}), {r['fresh_diff']}% on fresh seed {r['seed']}"
             f" | spec chain {'PASS' if r['chain'] else 'FAIL'}"
             f" | clone {q['clone_calls']}, to_owned {q['to_owned_calls']}, unwrap {q['unwrap_calls']}"
             f" | violations: {', '.join(r['violations']) or 'none'}"]
        if r.get("failures"):
            s += ["failing cases:"] + [f"  - {x}" for x in r["failures"]]
        if r.get("test_failures"):
            s += ["failing cargo tests:"] + [f"  - {x}" for x in r["test_failures"]]
        if not brief and score_key(r) < score_key(self.best):
            s.append(f"REGRESSION: best was {self.best['diff'] if self.best['build'] else 0}% "
                     f"at step {self.best_step}. Consider revert_to_best.")
        return "\n".join(s)

    # -- the digest (WRITE/SELECT) ---------------------------------------
    def digest(self, step: int) -> str:
        src = LIB.read_text()
        fns = re.findall(r"pub fn (\w+)", src)
        stubs = [f for f in REQUIRED if re.search(rf'todo!\("{f}"\)', src)]
        out = [f"## Harness state — model call {step} of {ARGS.budget}",
               f"rust/src/lib.rs: {len(src.splitlines())} lines; pub fns: {', '.join(fns) or 'none'}"
               + (f"; STILL STUBBED: {', '.join(stubs)}" if stubs else "")]
        if self.report:
            out.append(f"Latest check (step {self.report['step']}):\n" + self.render(self.report, brief=True))
            out.append(f"Best: {'%.1f%%' % self.best['diff'] if self.best['build'] else 'no build'}"
                       f" at step {self.best_step}"
                       + ("  (current == best)" if score_key(self.report) >= score_key(self.best)
                          else "  (current is WORSE; revert_to_best is available)"))
            out.append("Score trace (* = new best): " + " → ".join(self.trace[-10:]))
        else:
            out.append("No check yet: lib.rs is still the stub.")
        if self.notes:
            out.append("Your notes:\n" + "\n".join(f"- {n}" for n in self.notes))
        return "\n".join(out)

STATE = State()

def _test_failures() -> list[str]:
    out = subprocess.run(["cargo", "test", "--release", "-q"], cwd=RUST,
                         capture_output=True, text=True).stdout
    return [f"{name}: {_clip(' '.join(body.split()), 300)}"
            for name, body in re.findall(r"---- (\S+) stdout ----\n(.*?)(?:\n\n|\Z)", out, re.S)[:8]]

# ================================================================== tools
def t_read_reference(a):
    """Line-numbered slice of a file in reference/ (SELECT: read what you need)."""
    name = a.get("file") or "version.py"
    p = (REF / name).resolve()
    if p.parent != REF.resolve() or not p.is_file():
        return f"no such file {name!r}; have: {', '.join(sorted(f.name for f in REF.glob('*.py')))}"
    lines = p.read_text().splitlines()
    lo = max(1, int(a.get("start") or 1)); hi = min(len(lines), int(a.get("end") or len(lines)))
    return "\n".join(f"{i:4}  {lines[i - 1]}" for i in range(lo, hi + 1))

def t_grep_reference(a):
    rx = re.compile(a["pattern"])
    hits = [f"{p.name}:{i}: {l}" for p in sorted(REF.glob("*.py"))
            for i, l in enumerate(p.read_text().splitlines(), 1) if rx.search(l)]
    return "\n".join(hits[:80]) + (f"\n… {len(hits) - 80} more" if len(hits) > 80 else "") or "no matches"

def t_probe(a):
    """Differential oracle on demand: the same query, Python vs current Rust."""
    cases = [c.strip() for c in a.get("cases") or [] if c.strip()][:60]
    if not cases:
        return "give cases like: 'parse 1.0.0-01', 'compare 1.0.0-a 1.0.0-1', 'bump patch 1.2.3-rc.1', 'format 1.2.3+b'"
    rust, err = run_harness(cases)
    rows = [] if rust else [f"(rust binary unavailable: {err})"]
    if rust and STATE.report and not STATE.report["build"]:
        rows.append("(note: lib.rs does not build; rust column is from the last successful build)")
    for i, c in enumerate(cases):
        want, got = _oracle(c), (_from_rust(c, rust[i]) if rust else "?")
        rows.append(f"{'ok ' if want == got else 'XX '} {c}\n      oracle: {want}\n      rust:   {got}")
    return "\n".join(rows)

def _oracle(cmd: str) -> str:
    p = cmd.split()
    try:
        if p[0] == "parse" and len(p) == 2:
            v = ref_parse(p[1])
            return "INVALID" if v is None else _fmt_parse(v)
        if p[0] == "compare" and len(p) == 3:
            return str(ref_compare(p[1], p[2]))
        if p[0] == "bump" and len(p) == 3 and p[1] in ("major", "minor", "patch"):
            return ref_bump(p[1], p[2])
        if p[0] == "format" and len(p) == 2:
            return str(ref.Version.parse(p[1]))
    except (ValueError, TypeError):
        return "INVALID"
    return "BAD QUERY (use: parse V | compare A B | bump major|minor|patch V | format V)"

def _from_rust(cmd: str, g: dict) -> str:
    if not g.get("ok"):
        return "INVALID" if g.get("error", "").startswith(("parse", "compare", "bump", "format")) \
            else f"ERROR ({g.get('error')})"
    op = cmd.split()[0]
    return _fmt_parse(g) if op == "parse" else str(g["cmp"]) if op == "compare" else g["version"]

def _fmt_parse(v: dict) -> str:
    return f"{v['major']}.{v['minor']}.{v['patch']} prerelease={v['prerelease']!r} build={v['build']!r}"

def t_read_rust(_a):
    return LIB.read_text()

def t_write_rust(a):
    LIB.write_text(a["content"])
    return f"wrote {len(a['content'])} bytes\n\n" + STATE.check()

def t_edit_rust(a):
    """Exact, unique search/replace; much cheaper than rewriting the whole file."""
    src, old = LIB.read_text(), a["old"]
    n = src.count(old)
    if n != 1:
        return f"edit refused: `old` occurs {n} times (must be exactly once). No change made."
    LIB.write_text(src.replace(old, a["new"]))
    return "edit applied\n\n" + STATE.check()

def t_check(_a):
    return STATE.check()

def t_revert_to_best(_a):
    if not STATE.best_src:
        return "no verified snapshot yet"
    LIB.write_text(STATE.best_src)
    return f"restored the snapshot from step {STATE.best_step}\n\n" + STATE.check()

def t_remember(a):
    STATE.notes = (STATE.notes + [_clip(a["note"].strip(), 300)])[-12:]
    return f"noted ({len(STATE.notes)} notes kept)"

def t_finish(a):
    """Trust, but verify. Accept a clean state, or a second, reasoned claim."""
    r, problems = STATE.report, []
    if not r or not r["build"]:
        problems.append("the crate does not build")
    else:
        if r["violations"]:         problems.append(f"rule violations: {r['violations']}")
        if r["diff"] < 100:         problems.append(f"differential is {r['diff']}%")
        if not r["chain"]:          problems.append("spec precedence chain fails")
        if not tests_ok(r):         problems.append("cargo tests missing or failing")
        if smells(r) > MAX_SMELLS:  problems.append(f"{smells(r)} clone/to_owned/unwrap calls")
    legal = r and r["build"] and not r["violations"]
    if not problems or (legal and STATE.rejections >= 1):
        STATE.finished = _clip(a.get("summary") or "done", 200) + ("" if not problems else
                                                                  f" (accepted with: {'; '.join(problems)})")
        return "accepted"
    STATE.rejections += 1
    return ("not accepted: " + "; ".join(problems) + ". Keep working. If you are sure the rest "
            "cannot be closed, call finish again and say why.")

_none = {"type": "object", "properties": {}}
TOOLS = [
    dict(name="read_reference", fn=t_read_reference,
         description="Read a line-numbered slice of a file in reference/ (version.py, test_parsing.py, "
                     "test_compare.py, test_bump.py). Omit start/end for the whole file.",
         parameters={"type": "object", "properties": {
             "file": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}}}),
    dict(name="grep_reference", fn=t_grep_reference,
         description="Regex search over reference/*.py; returns file:line: text.",
         parameters={"type": "object", "required": ["pattern"],
                     "properties": {"pattern": {"type": "string"}}}),
    dict(name="probe", fn=t_probe,
         description="Ask the ground-truth Python semver oracle AND your current Rust binary the same "
                     "questions, side by side. Each case is one of: 'parse V', 'compare A B', "
                     "'bump major|minor|patch V', 'format V'. Up to 60 cases.",
         parameters={"type": "object", "required": ["cases"],
                     "properties": {"cases": {"type": "array", "items": {"type": "string"}}}}),
    dict(name="read_rust", fn=t_read_rust, parameters=_none,
         description="Read the current rust/src/lib.rs."),
    dict(name="write_rust", fn=t_write_rust,
         description="Overwrite rust/src/lib.rs with the COMPLETE file. Auto-runs the full check.",
         parameters={"type": "object", "required": ["content"],
                     "properties": {"content": {"type": "string"}}}),
    dict(name="edit_rust", fn=t_edit_rust,
         description="Replace one exact, unique snippet of rust/src/lib.rs. Include enough context to "
                     "make `old` unique. Auto-runs the full check.",
         parameters={"type": "object", "required": ["old", "new"],
                     "properties": {"old": {"type": "string"}, "new": {"type": "string"}}}),
    dict(name="check", fn=t_check, parameters=_none,
         description="Build, cargo test, and run the differential evaluation on a fresh seed. "
                     "Writes and edits already do this automatically."),
    dict(name="revert_to_best", fn=t_revert_to_best, parameters=_none,
         description="Restore lib.rs to the best-scoring verified snapshot."),
    dict(name="remember", fn=t_remember,
         description="Save a short note (a semantic fact, a decision, a plan) to the persistent "
                     "notes shown every turn. Old tool outputs are elided, but notes are not.",
         parameters={"type": "object", "required": ["note"],
                     "properties": {"note": {"type": "string"}}}),
    dict(name="finish", fn=t_finish,
         description="Declare the translation complete. The harness verifies this.",
         parameters={"type": "object", "properties": {"summary": {"type": "string"}}}),
]
BY_NAME = {t["name"]: t for t in TOOLS}
SCHEMAS = [{k: t[k] for k in ("name", "description", "parameters")} for t in TOOLS]

def run_tool(c: dict) -> str:
    tool = BY_NAME.get(c["name"])
    if not tool:
        return f"unknown tool {c['name']!r}; available: {', '.join(BY_NAME)}"
    try:
        out = str(tool["fn"](c.get("arguments") or {}))
    except Exception as e:                # bad arguments are feedback, not a crash
        out = f"tool error: {type(e).__name__}: {e}"
    return out if len(out) <= OUT_CAP else out[:OUT_CAP] + f"\n… [{len(out) - OUT_CAP} chars truncated]"

# =================================================================== loop
def main():
    global ARGS, _log
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40, help="max model calls (graded cap: 40)")
    ap.add_argument("--task", default="Translate reference/version.py into rust/src/lib.rs.")
    ARGS = a = ap.parse_args()

    LOGS.mkdir(exist_ok=True)
    log = LOGS / f"run-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    def rec(**kw):
        with log.open("a") as f:
            f.write(json.dumps({"t": time.time(), **kw}) + "\n")
    _log = rec

    history = [{"role": "system", "content": system_prompt()},
               {"role": "user",   "content": a.task}]
    rec(event="start", budget=a.budget, task=a.task, provider=PROVIDER, model=MODEL)
    print(f"[agent] {PROVIDER}/{MODEL}, budget {a.budget}")

    step = 0
    while True:
        stop, why = should_stop(history, step, a.budget, STATE)
        if stop:
            print(f"\n[stop] {why}")
            rec(event="stop", reason=why, steps=step)
            break

        step += 1
        STATE.step = step
        ctx = build_context(history, step)
        reply = call_model(ctx, SCHEMAS)
        rec(event="model", step=step, ctx_tokens=len(json.dumps(ctx)) // 4, reply=reply)

        if reply.get("text"):
            print(f"[{step}] {reply['text'][:200]}")
        calls = reply.get("tool_calls") or []
        history.append({"role": "assistant", "content": reply.get("text") or "", "tool_calls": calls})

        if not calls:                     # chatting is not progress; nudge once, then stop
            STATE.idle += 1
            history.append({"role": "user", "content":
                            "No tool was called. Continue with tools, or call finish if you are done."})
            continue
        STATE.idle = 0
        STATE.actions.append(json.dumps([[c["name"], c["arguments"]] for c in calls], sort_keys=True))

        for c in calls:
            out = run_tool(c)
            print(f"      -> {c['name']}: {out.splitlines()[0][:120] if out else ''}")
            rec(event="tool", step=step, name=c["name"], output=out[:4000])
            history.append({"role": "tool", "id": c["id"], "name": c["name"], "content": out})

    # never end worse than the best verified snapshot
    if STATE.best_src and score_key(STATE.report) < score_key(STATE.best):
        LIB.write_text(STATE.best_src)
        print(f"[rollback] restored best snapshot from step {STATE.best_step}")
        rec(event="rollback", to_step=STATE.best_step)

    print(f"\ntrajectory: {log}")
    print("final score:", flush=True)
    subprocess.run([sys.executable, str(HERE / "evaluate.py")], cwd=HERE)

if __name__ == "__main__":
    main()

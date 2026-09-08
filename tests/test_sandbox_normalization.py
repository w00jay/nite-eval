"""Per-container variation must not reach the model, or runs stop being reproducible.

Repeat runs of the same task at temperature 0 diverge because container output
enters the conversation. The first differing tool-call argument always appeared
immediately after the first differing tool result, on one of these surfaces:

    -  drwxr-xr-x 3 root root 4096 Aug 31 20:06 .
    +  drwxr-xr-x 3 root root 4096 Aug 31 20:31 .
    -  <horizons_client.HorizonsClient object at 0x7038b21b87a0>
    +  <horizons_client.HorizonsClient object at 0x71ad4a0dc3e0>
    -  1 failed, 3 passed in 0.02s
    +  1 failed, 3 passed in 0.31s
    -  ok  mcpconfig/config 0.005s
    +  ok  mcpconfig/config 0.006s
    -  extract parses a label ... ok (5ms)
    +  extract parses a label ... ok (4ms)
    -  2026/09/06 18:57:14 INFO proxied call
    +  2026/09/08 03:17:41 INFO proxied call
    -  HOSTNAME=02c32ab71364
    +  HOSTNAME=e51efe48974d

From turn 2 the model picks a different exploratory command and the runs write
different code. ornith-1.5 emitted 65986, 66666 and 65604 bytes of tool-call
arguments across three runs of coding_artemis_medium_01; in one, the code it
happened to write hit a real bug and both automated criteria scored 0 rather
than 1.00 and 0.93.

Normalisation is deliberately narrow, and half of these tests exist to hold it
that way. Every shell result goes through SandboxToolEnv.exec, including `cat`
behind read_file, so a pattern matched on a bare volatile token — any hex
string, any number followed by `s` — would rewrite the contents of files the
model then copies into its own source. Each pattern is anchored to the context
that identifies it instead.
"""

from nite_eval.sandbox import _normalize_volatile

# --- Surface 1: ls -la mtimes (pre-existing behaviour, must keep working) ---


def test_ls_long_format_date_is_replaced():
    out = "total 8\ndrwxr-xr-x 3 root root 4096 Aug 31 20:06 .\n-rw-r--r-- 1 root root  220 Sep  1 02:14 go.mod\n"
    a = _normalize_volatile(out)
    b = _normalize_volatile(out.replace("Aug 31 20:06", "Aug 31 20:31").replace("Sep  1 02:14", "Sep  1 09:03"))
    assert a == b
    assert "20:06" not in a and "20:31" not in a
    # Everything that is not the date survives.
    assert "drwxr-xr-x 3 root root 4096" in a
    assert "go.mod" in a


def test_year_form_is_replaced():
    """Files older than six months show a year instead of a time."""
    a = _normalize_volatile("-rw-r--r-- 1 root root 12 Jan  5  2024 old.txt\n")
    b = _normalize_volatile("-rw-r--r-- 1 root root 12 Mar  9  2019 old.txt\n")
    assert a == b
    assert "old.txt" in a


def test_full_iso_time_style_is_replaced():
    """`ls --full-time` and `--time-style=full-iso`."""
    a = _normalize_volatile("-rw-r--r-- 1 root root 220 2026-08-31 20:06:12.345678901 +0000 go.mod\n")
    b = _normalize_volatile("-rw-r--r-- 1 root root 220 2026-09-01 02:31:44.111111111 +0000 go.mod\n")
    assert a == b
    assert "go.mod" in a


def test_file_contents_are_not_touched():
    """read_file goes through exec; rewriting dates here would corrupt source."""
    source = (
        'const RELEASED = "2026-08-31 20:06:12";\n'
        "// changelog: Aug 31 20:06 shipped the parser\n"
        'print("Sep  1 02:14")\n'
    )
    assert _normalize_volatile(source) == source


def test_prose_mentioning_a_date_is_not_touched():
    assert _normalize_volatile("Build finished Aug 31 20:06 with no errors\n") == (
        "Build finished Aug 31 20:06 with no errors\n"
    )


# --- Surface 2: ASLR heap addresses in Python reprs ---


def _pytest_failure(addr: str) -> str:
    return (
        "=================================== FAILURES ===================================\n"
        "________________________ test_get_position_uses_cache _________________________\n"
        "E       AssertionError: assert 2 == 1\n"
        f"E        +  where 2 = <horizons_client.HorizonsClient object at {addr}>.fetch_count\n"
        f"E       cache holder: <horizons_client.TTLCache object at {addr}>\n"
    )


def test_python_repr_address_is_replaced():
    a = _normalize_volatile(_pytest_failure("0x7038b21b87a0"))
    b = _normalize_volatile(_pytest_failure("0x71ad4a0dc3e0"))
    assert a == b
    assert "0x7038b21b87a0" not in a and "0x71ad4a0dc3e0" not in a
    # The class name is the part of the repr the model needs; only the address goes.
    assert "<horizons_client.HorizonsClient object at" in a
    assert ".fetch_count" in a


def test_hex_literals_are_not_touched():
    """Blanking every hex token would rewrite constants, masks and assertions."""
    source = (
        "MASK = 0x7038B21B87A0\n"
        "assert response.checksum == 0xDEADBEEF\n"
        "const magic = 0x1F4  // gateway framing\n"
        "offset = ptr_at(buf) + 0x10\n"
        "Expected 0x7038b21b87a0, got 0x71ad4a0dc3e0\n"
    )
    assert _normalize_volatile(source) == source


# --- Surface 3: pytest summary elapsed time ---
#
# Every form below was produced by running pytest and copying what it printed,
# not from memory. `coding_artemis_medium_01` runs `pytest -q`, so the bare
# forms are the ones that actually reach the model; the `=`-padded ones appear
# if a task ever drops -q.


def _pytest_transcript(elapsed: str) -> str:
    return (
        "=========================== short test summary info ============================\n"
        "FAILED test_horizons.py::test_get_position_uses_cache - AssertionError: assert 2 == 1\n"
        f"1 failed, 3 passed, 1 skipped, 1 warning in {elapsed}s\n"
    )


def test_pytest_summary_elapsed_time_is_replaced():
    a = _normalize_volatile(_pytest_transcript("0.02"))
    b = _normalize_volatile(_pytest_transcript("0.31"))
    assert a == b
    # Counts and the failure line are the signal; only the duration goes.
    assert "1 failed, 3 passed, 1 skipped, 1 warning in" in a
    assert "test_horizons.py::test_get_position_uses_cache" in a
    assert "0.02s" not in a and "0.31s" not in a


def test_pytest_summary_forms_are_all_covered():
    """Shapes observed from real pytest runs, bare (-q) and `=`-padded."""
    pairs = [
        ("347 passed in 18.19s", "347 passed in 19.04s"),
        ("2 passed, 3 deselected in 0.01s", "2 passed, 3 deselected in 0.02s"),
        ("5 deselected in 0.01s", "5 deselected in 0.03s"),
        ("1 error in 0.12s", "1 error in 0.14s"),
        (
            "============== 1 failed, 3 passed, 1 skipped, 1 warning in 0.04s ===============",
            "============== 1 failed, 3 passed, 1 skipped, 1 warning in 0.51s ===============",
        ),
        (
            "======================= 2 passed, 3 deselected in 0.01s ========================",
            "======================= 2 passed, 3 deselected in 0.09s ========================",
        ),
    ]
    for first, second in pairs:
        assert first != second
        assert _normalize_volatile(first + "\n") == _normalize_volatile(second + "\n"), first
    # The `=` padding is part of the line and must survive intact.
    padded = _normalize_volatile("======================= 2 passed, 3 deselected in 0.01s ========================\n")
    assert padded.startswith("=======================")
    assert padded.rstrip("\n").endswith("========================")


def test_pytest_like_lines_from_program_output_are_not_touched():
    """The summary must be the WHOLE line; a prefix or a trailer means it is not pytest's."""
    out = (
        "cache warmed: 2 passed in 0.5s\n"
        "[horizons] 12 passed in 1.02s\n"
        "2 failed in 0.34s window, retrying\n"
        'assert summary == "3 passed in 0.34s"\n'
        "# ephemeris covers 3 steps in 0.34s of TDB\n"
        "retry_budget = 2  # passed in 0.34s last time\n"
    )
    assert _normalize_volatile(out) == out


def test_a_bare_summary_shaped_line_is_indistinguishable():
    """Documents a limitation; it does not assert desired behaviour.

    A line that is exactly `2 passed in 0.5s` is byte-identical whether pytest
    printed it or the model's own code did, so the anchor cannot separate them
    and this one is rewritten. Accepted: for a model to hit it, its program has
    to reproduce pytest's summary format exactly on a line of its own.
    """
    assert _normalize_volatile("2 passed in 0.5s\n") == "2 passed in 0.00s\n"


# --- Surface 4: `go test` elapsed times ---


def _go_test_output(pkg: str, pass_s: str, fail_s: str) -> str:
    return (
        "=== RUN   TestLoadValidConfig\n"
        f"--- PASS: TestLoadValidConfig ({pass_s}s)\n"
        "=== RUN   TestLoadInvalidURL\n"
        "=== RUN   TestLoadInvalidURL/no_scheme\n"
        f"    --- PASS: TestLoadInvalidURL/no_scheme ({pass_s}s)\n"
        "=== RUN   TestLoadMissingFile\n"
        f"--- FAIL: TestLoadMissingFile ({fail_s}s)\n"
        "    config_test.go:41: want ErrNotFound, got nil\n"
        "FAIL\n"
        f"FAIL\tmcpconfig/config\t{pkg}s\n"
    )


def test_go_test_elapsed_times_are_replaced():
    a = _normalize_volatile(_go_test_output("0.005", "0.00", "0.01"))
    b = _normalize_volatile(_go_test_output("0.006", "0.01", "0.00"))
    assert a == b
    # The package, the test names and the failure message all survive.
    assert "FAIL\tmcpconfig/config" in a
    assert "--- FAIL: TestLoadMissingFile" in a
    assert "    --- PASS: TestLoadInvalidURL/no_scheme" in a
    assert "config_test.go:41: want ErrNotFound, got nil" in a


def test_go_package_line_keeps_its_trailer():
    a = _normalize_volatile("ok  \tmcpgateway\t1.204s\tcoverage: 81.3% of statements\n")
    b = _normalize_volatile("ok  \tmcpgateway\t1.198s\tcoverage: 81.3% of statements\n")
    assert a == b
    assert "coverage: 81.3% of statements" in a


def test_program_printed_durations_are_not_touched():
    """A duration is only volatile when it is a test runner's status line."""
    out = (
        "proxied call to upstream completed in 0.005s\n"
        "cache warmed after 1.204s, 12 entries\n"
        "ok: gateway healthy\n"
        "ok 1 - config loads in 0.005s\n"
        "const RequestTimeout = 30.0s\n"
        "want 0.005s, got 0.006s\n"
    )
    assert _normalize_volatile(out) == out


# --- Surface 5: `deno test` elapsed times ---


def _deno_test_output(first: str, second: str, total: str) -> str:
    return (
        "running 2 tests from ./scan_label_test.ts\n"
        f"extract parses a Cloudy Bay label ... ok ({first})\n"
        f"match rejects below SIMILARITY_THRESHOLD ... ok ({second})\n"
        "\n"
        f"ok | 2 passed | 0 failed ({total})\n"
    )


def test_deno_test_elapsed_times_are_replaced():
    a = _normalize_volatile(_deno_test_output("5ms", "4ms", "12ms"))
    b = _normalize_volatile(_deno_test_output("4ms", "7ms", "13ms"))
    assert a == b
    assert "extract parses a Cloudy Bay label ... ok" in a
    assert "2 passed | 0 failed" in a


def test_deno_failure_summary_is_replaced():
    a = _normalize_volatile("scan handles a blurred label ... FAILED (8ms)\n\nFAILED | 1 passed | 1 failed (21ms)\n")
    b = _normalize_volatile("scan handles a blurred label ... FAILED (9ms)\n\nFAILED | 1 passed | 1 failed (19ms)\n")
    assert a == b
    assert "scan handles a blurred label ... FAILED" in a
    assert "1 passed | 1 failed" in a


def test_millisecond_durations_in_program_output_are_not_touched():
    out = (
        "label scan took 5ms\n"
        "console.log(`ocr ${elapsed}ms`);\n"
        "// budget: 4ms per label, 12ms per batch\n"
        "assertEquals(result.elapsed_ms, 5);\n"
    )
    assert _normalize_volatile(out) == out


# --- Surface 6: wall clock printed by the model's own generated code ---


def _gateway_log(stamp: str) -> str:
    return (
        f"{stamp} gateway listening on :8080\n"
        f"{stamp} INFO proxied call upstream=tools/list\n"
        f"{stamp} INFO proxied call upstream=resources/read\n"
    )


def test_go_stdlib_log_timestamp_is_replaced():
    a = _normalize_volatile(_gateway_log("2026/09/06 18:57:14"))
    b = _normalize_volatile(_gateway_log("2026/09/08 03:17:41"))
    assert a == b
    assert "INFO proxied call upstream=tools/list" in a


def test_go_log_microsecond_and_slog_variants_are_replaced():
    """log.Lmicroseconds, and log/slog's text handler."""
    a = _normalize_volatile("2026/09/06 18:57:14.123456 INFO proxied call\n")
    b = _normalize_volatile("2026/09/08 03:17:41.987654 INFO proxied call\n")
    assert a == b

    c = _normalize_volatile('time=2026-09-06T18:57:14.123-07:00 level=INFO msg="proxied call" upstream=tools/list\n')
    d = _normalize_volatile('time=2026-09-08T03:17:41.987Z level=INFO msg="proxied call" upstream=tools/list\n')
    assert c == d
    assert 'level=INFO msg="proxied call" upstream=tools/list' in c


def test_dates_in_data_and_prose_are_not_touched():
    """Horizons ephemeris text is data the model must parse exactly."""
    data = (
        "$$SOE\n"
        "2460600.500000000 = A.D. 2024-Sep-15 00:00:00.0000 TDB\n"
        " X = 1.234567890123456E+05 Y =-2.345678901234567E+05 Z = 3.456789012345678E+04\n"
        "$$EOE\n"
        "cache entry expires 2024-09-15T00:05:00Z (TTL 300s)\n"
        "2026/09/06 migration notes: renamed the column\n"
        "# see the 2026/09/06 18:57:14 incident log for context\n"
        "released := time.Date(2026, 9, 6, 18, 57, 14, 0, time.UTC)\n"
    )
    assert _normalize_volatile(data) == data


# --- Surface 7: container hostname ---


def _env_output(hostname: str) -> str:
    return (
        "PATH=/usr/local/go/bin:/usr/local/sbin:/usr/local/bin\n"
        f"HOSTNAME={hostname}\n"
        "HOME=/home/sandbox\n"
        "GOFLAGS=-mod=mod\n"
    )


def test_container_hostname_is_replaced():
    a = _normalize_volatile(_env_output("02c32ab71364"))
    b = _normalize_volatile(_env_output("e51efe48974d"))
    assert a == b
    assert "02c32ab71364" not in a and "e51efe48974d" not in a
    assert "GOFLAGS=-mod=mod" in a


def test_hex_ids_outside_the_hostname_variable_are_not_touched():
    """Only `HOSTNAME=` is anchored; a bare 12-hex string is indistinguishable."""
    out = (
        "MCP_SERVER_ID=02c32ab71364\n"
        "digest: 02c32ab71364\n"
        "02c32ab71364  ./cache/blob\n"
        "CONTAINER_HOSTNAME=02c32ab71364\n"
    )
    assert _normalize_volatile(out) == out


# --- Known gap: Go map iteration order ---


def test_go_map_iteration_order_remains_a_known_gap():
    """Documents a limitation; it does not assert desired behaviour.

    Go randomizes map iteration per process, so a table-driven test backed by a
    map emits its subtest blocks permuted between runs. Measured on
    coding_mcp_easy_01 / ornith-1.5-35b-a3b: TestLoadInvalidURL's four subtests
    came back as no_scheme, no_host, bad_scheme, spaces in run-20260906-181136
    and bad_scheme, spaces, no_scheme, no_host in run-20260908-012542.

    This is the order of lines, not a token inside one, so no substitution can
    reach it. Normalising it would mean sorting output, which mangles
    interleaved writers — _normalize_volatile deliberately does not.
    """

    def out(cases: list[str], elapsed: str) -> str:
        return "".join(f"    --- PASS: TestLoadInvalidURL/{c} ({elapsed}s)\n" for c in cases)

    a = _normalize_volatile(out(["no_scheme", "no_host", "bad_scheme", "spaces"], "0.00"))
    b = _normalize_volatile(out(["bad_scheme", "spaces", "no_scheme", "no_host"], "0.01"))
    # The elapsed times are normalised, so the only thing left is the ordering.
    assert "0.01s" not in b
    assert a != b
    assert sorted(a.splitlines()) == sorted(b.splitlines())


# --- Whole-output smoke test ---


def test_two_full_runs_of_the_same_task_normalize_identically():
    """All seven substitutable surfaces in one exec transcript, as the model sees them."""

    def transcript(mtime: str, addr: str, pytest_s: str, elapsed: str, ms: str, stamp: str, host: str) -> str:
        return (
            f"drwxr-xr-x 3 root root 4096 {mtime} .\n"
            f"HOSTNAME={host}\n"
            f"{stamp} INFO proxied call upstream=tools/list\n"
            f"E        +  where None = <horizons_client.HorizonsClient object at {addr}>.get_position\n"
            f"1 failed, 3 passed in {pytest_s}s\n"
            f"ok  \tmcpconfig/config\t{elapsed}s\n"
            f"extract parses a Cloudy Bay label ... ok ({ms})\n"
        )

    a = transcript("Aug 31 20:06", "0x7038b21b87a0", "0.02", "0.005", "5ms", "2026/09/06 18:57:14", "02c32ab71364")
    b = transcript("Aug 31 20:31", "0x71ad4a0dc3e0", "0.31", "0.006", "4ms", "2026/09/08 03:17:41", "e51efe48974d")
    assert a != b
    assert _normalize_volatile(a) == _normalize_volatile(b)


def test_empty_and_plain_output():
    assert _normalize_volatile("") == ""
    assert _normalize_volatile("hello\n") == "hello\n"


def test_find_ls_entries_have_their_dates_replaced():
    """`find -ls` puts inode and block counts before the mode bits.

    The original anchor required the mode bits at line start, so this format
    leaked mtimes. Measured live: muse-glimmer's coding_mcp_hard_01 diverged at
    turn 7 on `Sep  8 20:44` in exactly this listing.
    """
    a = "       33      4 -rw-r--r--   1 root     root           95 Sep  8 20:44 /app/mcpgateway/go.mod"
    b = "       33      4 -rw-r--r--   1 root     root           95 Sep  9 03:12 /app/mcpgateway/go.mod"
    assert a != b
    assert _normalize_volatile(a) == _normalize_volatile(b)


def test_ls_inode_and_block_prefixes_are_handled():
    """`ls -li` carries an inode, `ls -ls` a block count, `find -ls` both."""
    for pre in ("  1234 ", "     4 ", "   33      4 "):
        a = f"{pre}-rw-r--r--   1 root root  95 Sep  8 20:44 go.mod"
        b = f"{pre}-rw-r--r--   1 root root  95 Sep  9 03:01 go.mod"
        assert _normalize_volatile(a) == _normalize_volatile(b), pre


def test_a_numeric_prefix_alone_does_not_make_a_listing():
    """The mode/links/owner/group/size run is still required.

    Without it a leading number would turn any prose containing an ls-shaped
    date into a rewrite target.
    """
    for line in (
        "Release date: Sep  8 20:44 per the changelog",
        "42 records written Sep  8 20:44",
        "   33      4 files matched Sep  8 20:44",
    ):
        assert _normalize_volatile(line) == line, line

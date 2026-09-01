"""Container timestamps must not reach the model, or runs stop being reproducible.

The first tool call on a coding task is usually `ls -la`, and the sandbox returns
the working directory's mtime — the container's creation time:

    -  drwxr-xr-x 3 root root 4096 Aug 31 20:06 .
    +  drwxr-xr-x 3 root root 4096 Aug 31 20:31 .

That reaches the conversation, so from turn 2 the model picks a different
exploratory command and the runs write different code. ornith-1.5 emitted 65986,
66666 and 65604 bytes of tool-call arguments across three runs of
coding_artemis_medium_01 at temperature 0; in one the code it happened to write
hit a real bug and both automated criteria scored 0 rather than 1.00 and 0.93.

Normalisation is deliberately narrow. Every shell result goes through
SandboxSession.exec, including `cat` behind read_file, so substituting dates
anywhere in the output would rewrite the contents of files the model then copies
into its own source. Only a line whose shape is unmistakably an `ls -l` entry —
mode bits, link count, owner, group, size, then the date — is touched, and only
the date field within it.
"""

from nite_eval.sandbox import _normalize_volatile


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


def test_go_test_output_is_untouched():
    """Durations vary too, but they are a real signal and not ours to rewrite."""
    out = "ok  \tmcpconfig\t0.123s\nFAIL\tmcpgateway\t1.204s\n"
    assert _normalize_volatile(out) == out


def test_empty_and_plain_output():
    assert _normalize_volatile("") == ""
    assert _normalize_volatile("hello\n") == "hello\n"

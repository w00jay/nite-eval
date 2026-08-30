"""Hidden tests for coding_artemis_medium_01, installed after the conversation.

HTTP is stubbed with httpx.MockTransport and time with an injected clock, so
the suite is hermetic and needs neither network nor sleeping.
"""

import httpx
import pytest

from horizons_client import HorizonsClient, HorizonsError, parse_vectors

EPHEM = """\
*******************************************************************************
Ephemeris / WWW_USER
$$SOE
2460600.500000000 = A.D. 2024-Sep-15 00:00:00.0000 TDB
 X = 1.000000000000000E+05 Y = 2.000000000000000E+05 Z = 3.000000000000000E+05
 VX= 1.000000000000000E+00 VY= 2.000000000000000E+00 VZ= 3.000000000000000E+00
 LT= 9.100000000000000E-01 RG= 3.741657386773941E+05 RR= 2.100000000000000E-01
2460601.500000000 = A.D. 2024-Sep-16 00:00:00.0000 TDB
 X = 3.000000000000000E+05 Y = 4.000000000000000E+05 Z = 5.000000000000000E+05
 VX= 3.000000000000000E+00 VY= 4.000000000000000E+00 VZ= 5.000000000000000E+00
 LT= 1.910000000000000E+00 RG= 7.071067811865476E+05 RR= 4.100000000000000E-01
$$EOE
*******************************************************************************
"""

# Real Horizons writes negative values hard against the '=' with no space.
EPHEM_NEGATIVE = """\
$$SOE
2460600.500000000 = A.D. 2024-Sep-15 00:00:00.0000 TDB
 X =-1.500000000000000E+05 Y = 2.500000000000000E+05 Z =-3.500000000000000E+04
 VX=-1.500000000000000E+00 VY= 2.500000000000000E+00 VZ=-3.500000000000000E-01
 LT= 1.000000000000000E+00 RG= 1.000000000000000E+05 RR= 1.000000000000000E-01
$$EOE
"""


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def make_client(payload=None, status=200, on_request=None):
    def handler(request):
        if on_request:
            on_request(request)
        if status != 200:
            return httpx.Response(status, json={"error": "upstream boom"})
        return httpx.Response(200, json={"result": payload if payload is not None else EPHEM})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- parsing (also decides parsing_correctness via -k parse) ---


def test_parse_extracts_all_rows():
    rows = parse_vectors(EPHEM)
    assert len(rows) == 2
    assert rows[0]["x"] == pytest.approx(1.0e5)
    assert rows[0]["vz"] == pytest.approx(3.0)
    assert rows[1]["y"] == pytest.approx(4.0e5)


def test_parse_ignores_content_outside_the_markers():
    noisy = EPHEM.replace("Ephemeris / WWW_USER", "X = 9.9E+09 Y = 9.9E+09 Z = 9.9E+09")
    rows = parse_vectors(noisy)
    assert len(rows) == 2
    assert all(row["x"] < 1e9 for row in rows)


def test_parse_records_the_epoch():
    rows = parse_vectors(EPHEM)
    assert rows[0]["jdtdb"] == pytest.approx(2460600.5)
    assert rows[1]["jdtdb"] == pytest.approx(2460601.5)


def test_parse_rejects_missing_markers():
    with pytest.raises(ValueError):
        parse_vectors("no markers here at all")


def test_parse_rejects_empty_block():
    with pytest.raises(ValueError):
        parse_vectors("$$SOE\n$$EOE\n")


# --- caching and fetching ---


async def test_cache_hit_avoids_a_second_request():
    client = HorizonsClient(make_client(), now=Clock())
    await client.get_position(2460600.5)
    await client.get_position(2460600.5)
    assert client.request_count == 1


async def test_cache_refetches_after_ttl_expires():
    clock = Clock()
    client = HorizonsClient(make_client(), ttl_recent=300.0, now=clock)
    await client.get_position(2460600.5)
    clock.advance(301.0)
    await client.get_position(2460600.5)
    assert client.request_count == 2


async def test_upstream_error_raises_horizons_error():
    client = HorizonsClient(make_client(status=500), now=Clock())
    with pytest.raises(HorizonsError):
        await client.get_position(2460600.5)


async def test_malformed_payload_raises_horizons_error():
    client = HorizonsClient(make_client(payload="garbage with no markers"), now=Clock())
    with pytest.raises(HorizonsError):
        await client.get_position(2460600.5)


async def test_exact_epoch_returns_that_row():
    client = HorizonsClient(make_client(), now=Clock())
    pos = await client.get_position(2460601.5)
    assert pos["x"] == pytest.approx(3.0e5)
    assert pos["vy"] == pytest.approx(4.0)


async def test_midpoint_is_linearly_interpolated():
    client = HorizonsClient(make_client(), now=Clock())
    pos = await client.get_position(2460601.0)
    assert pos["x"] == pytest.approx(2.0e5)
    assert pos["y"] == pytest.approx(3.0e5)
    assert pos["vx"] == pytest.approx(2.0)


async def test_quarter_point_is_linearly_interpolated():
    """A midpoint alone can be passed by averaging; this needs real interpolation."""
    client = HorizonsClient(make_client(), now=Clock())
    pos = await client.get_position(2460600.75)
    assert pos["x"] == pytest.approx(1.5e5)
    assert pos["vz"] == pytest.approx(3.5)


def test_parse_ignores_the_light_time_line():
    """Real Horizons emits LT/RG/RR, which this module does not need."""
    rows = parse_vectors(EPHEM)
    assert len(rows) == 2
    assert set(rows[0]) >= {"jdtdb", "x", "y", "z", "vx", "vy", "vz"}
    # Range must not be mistaken for a coordinate.
    assert rows[0]["x"] == pytest.approx(1.0e5)


def test_parse_handles_negative_values_without_a_space():
    rows = parse_vectors(EPHEM_NEGATIVE)
    assert len(rows) == 1
    assert rows[0]["x"] == pytest.approx(-1.5e5)
    assert rows[0]["z"] == pytest.approx(-3.5e4)
    assert rows[0]["vx"] == pytest.approx(-1.5)
    assert rows[0]["vz"] == pytest.approx(-0.35)

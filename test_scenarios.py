"""Tests for pay_ranges module."""
from __future__ import annotations

import pytest

from mb.pay_ranges import (BELOW_8_LABEL, BELOW_8_MEDIAN, PAY_RANGES_ROWS,
                            cascade_options, lookup_range, pay_ranges_df)


class TestPayRangesData:
    def test_has_data(self):
        assert len(PAY_RANGES_ROWS) > 30

    def test_dataframe_has_key_cols(self):
        df = pay_ranges_df()
        for c in ["ref_level", "location", "pay_zone", "p25", "p50", "p75"]:
            assert c in df.columns


class TestCascade:
    def test_all_levels_listed(self):
        opts = cascade_options()
        # "< 8" must be first so users don't miss it
        assert opts["levels"][0] == BELOW_8_LABEL
        # Numeric levels should be sorted numerically, not lexically
        numeric_part = [lv for lv in opts["levels"][1:] if lv.isdigit()]
        nums = [int(lv) for lv in numeric_part]
        assert nums == sorted(nums)

    def test_below_8_has_no_zones_or_locations(self):
        """Selecting '< 8' should NOT cascade — zones and locations stay empty."""
        opts = cascade_options(selected_level=BELOW_8_LABEL)
        assert opts["zones"]     == []
        assert opts["locations"] == []

    def test_zones_depend_on_level(self):
        opts = cascade_options(selected_level="12")
        assert "Sales"   in opts["zones"]
        assert "Support" in opts["zones"]
        assert "R&D"     in opts["zones"]
        assert "Ops"     in opts["zones"]

    def test_locations_depend_on_level_and_zone(self):
        opts = cascade_options(selected_level="12", selected_zone="Support")
        assert "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ" in opts["locations"]

    def test_empty_cascade_when_nothing_selected(self):
        opts = cascade_options()
        assert opts["zones"] == []
        assert opts["locations"] == []


class TestLookupBelow8:
    def test_returns_fixed_median(self):
        hit = lookup_range(BELOW_8_LABEL, "", "")
        assert hit is not None
        assert hit["p50"] == BELOW_8_MEDIAN
        assert hit["p50"] == 1300.0
        assert hit["is_fixed"] is True

    def test_zone_and_location_ignored_when_below_8(self):
        """Even if the caller accidentally passes zone/location, ignore them."""
        hit = lookup_range(BELOW_8_LABEL, "Support", "ΚΙΛΚΙΣ")
        assert hit is not None
        assert hit["p50"] == 1300.0

    def test_p25_p50_p75_all_equal_for_fixed(self):
        """For the fixed case the UI shows a single info box, so all three are equal."""
        hit = lookup_range(BELOW_8_LABEL, "", "")
        assert hit["p25"] == hit["p50"] == hit["p75"] == 1300.0


class TestLookup:
    def test_exact_match(self):
        hit = lookup_range("12", "Support", "ΕΥΚΑΡΠΙΑ / ΚΙΛΚΙΣ")
        assert hit is not None
        assert hit["p50"] == pytest.approx(1501.93)
        assert hit["p25"] == pytest.approx(1302.00)
        assert hit["p75"] == pytest.approx(1782.00)
        assert hit["is_fixed"] is False

    def test_miss_returns_none(self):
        assert lookup_range("999", "Support", "ΕΥΚΑΡΠΙΑ") is None

    def test_executive_row(self):
        hit = lookup_range("21", "Executives", "ΕΥΚΑΡΠΙΑ")
        assert hit is not None
        assert hit["p50"] == pytest.approx(10013.20)

    def test_rows_24_25_equal_percentiles(self):
        """Rows 24 and 25 have P25=P50=P75 per the source note."""
        hit = lookup_range("24", "Executives", "ΕΥΚΑΡΠΙΑ")
        assert hit is not None
        assert hit["p25"] == hit["p50"] == hit["p75"]

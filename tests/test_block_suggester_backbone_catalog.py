"""Tests for BlockSuggester's config-driven llm_backbone_selection catalog
(meta_agent/block_suggester.py) -- the "{{BACKBONE_CATALOG}}" placeholder
in _BLOCK_BODIES, substituted by _block_body() from either the default
catalog or a config-supplied override.

    PYTHONPATH=. python3 -m unittest tests.test_block_suggester_backbone_catalog
"""
from __future__ import annotations

import unittest

from meta_agent.block_suggester import (
    _DEFAULT_BACKBONE_CATALOG,
    BlockSuggester,
    _render_backbone_catalog,
)


def _fake_llm(**kwargs):
    return None


class BackboneCatalogTests(unittest.TestCase):
    def test_default_catalog_used_when_not_configured(self) -> None:
        bs = BlockSuggester(llm_caller=_fake_llm)
        self.assertEqual(bs.backbone_catalog, _DEFAULT_BACKBONE_CATALOG)

    def test_default_catalog_has_four_entries_all_verified_real_this_session(self) -> None:
        slugs = [c["slug"] for c in _DEFAULT_BACKBONE_CATALOG]
        self.assertEqual(
            slugs,
            [
                "qwen/qwen3.5-35b-a3b",
                "qwen/qwen3.6-35b-a3b",
                "qwen/qwen3.8-27b",
                "google/gemini-2.5-flash",
            ],
        )

    def test_placeholder_fully_substituted_with_default_catalog(self) -> None:
        bs = BlockSuggester(llm_caller=_fake_llm)
        body = bs._block_body("llm_backbone_selection")
        self.assertNotIn("{{BACKBONE_CATALOG}}", body)
        for c in _DEFAULT_BACKBONE_CATALOG:
            self.assertIn(c["slug"], body)

    def test_custom_catalog_override_replaces_default_entirely(self) -> None:
        custom = [
            {"slug": "qwen/qwen3.6-27b", "note": "newer generation, smaller"},
            {"slug": "google/gemini-3.6-flash", "note": "newer Gemini generation"},
        ]
        bs = BlockSuggester(llm_caller=_fake_llm, backbone_catalog=custom)
        body = bs._block_body("llm_backbone_selection")
        self.assertIn("qwen/qwen3.6-27b", body)
        self.assertIn("google/gemini-3.6-flash", body)
        # None of the default-only slugs should leak in when overridden.
        self.assertNotIn("qwen/qwen3.5-35b-a3b", body)
        self.assertNotIn("google/gemini-2.5-flash", body)

    def test_other_blocks_unaffected_by_catalog_substitution(self) -> None:
        bs = BlockSuggester(llm_caller=_fake_llm)
        for block in ("individual_subagent", "collaboration_workflow", "foundation_capability", "verifiers", "mixed"):
            body = bs._block_body(block)
            self.assertNotIn("{{BACKBONE_CATALOG}}", body)
            self.assertNotIn("qwen/qwen3.5-35b-a3b", body)

    def test_render_backbone_catalog_formats_one_line_per_entry(self) -> None:
        catalog = [
            {"slug": "a/b", "note": "x"},
            {"slug": "c/d", "note": "y"},
        ]
        rendered = _render_backbone_catalog(catalog)
        self.assertEqual(rendered, "  - a/b (x)\n  - c/d (y)")


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Per-task routing via operator-defined ``delegation.tiers``.

The model may name a ``tier`` on each task; the tier maps to a route (model/provider/endpoint) the operator wrote
in config.yaml. Tasks without a tier keep the call's default route, an unknown tier refuses the batch, and the
schema advertises ``tier`` only with the configured names.
"""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from tools import delegate_tool, delegate_tool_config
from tools.delegate_tool import delegate_task


def _parent():
    parent = MagicMock()
    parent.base_url = "https://api.anthropic.com"
    parent.api_key = "parent-key"
    parent.provider = "anthropic"
    parent.api_mode = "anthropic_messages"
    parent.model = "claude-opus-5-5"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent.request_overrides = None
    parent._session_db = None
    parent._delegate_depth = 0
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


TIERS = {
    "descobrir": {"model": "claude-opus-5-5", "use_for": "open-ended discovery"},
    "executar": {"model": "claude-sonnet-5", "use_for": "implementation against an executable acceptance check"},
    "varrer": {"model": "claude-haiku-4-5", "use_for": "mechanical collection"},
}

GOAL_A = "Implement the parser change described in the attached spec"
GOAL_B = "Audit the context files for contradictions against the runtime"


def _completed(idx):
    return {"task_index": idx, "status": "completed", "summary": "ok",
            "api_calls": 1, "duration_seconds": 1.0, "_child_role": None}


class _Harness:
    """Runs delegate_task with a given delegation config, capturing each child's build kwargs."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.built = []

    def _build(self, **kwargs):
        self.built.append(kwargs)
        child = MagicMock()
        child._progress_identity_ref = {}
        return child

    def run(self, tasks):
        with patch.object(delegate_tool, "_load_config", return_value=self.cfg), \
             patch.object(delegate_tool_config, "_load_config", return_value=self.cfg), \
             patch.object(delegate_tool, "_build_child_preserving_parent_tools", side_effect=self._build), \
             patch.object(delegate_tool, "_run_single_child",
                          side_effect=[_completed(i) for i in range(len(tasks))]):
            return json.loads(delegate_task(tasks=tasks, parent_agent=_parent()))


class TestTierRouting(unittest.TestCase):
    def test_tasks_without_tier_keep_the_default_route(self):
        h = _Harness({"model": "", "provider": "", "tiers": TIERS})
        result = h.run([{"goal": GOAL_A}, {"goal": GOAL_B}])
        self.assertNotIn("error", result)
        self.assertEqual([b["model"] for b in h.built], [None, None])  # None = inherit the parent model

    def test_each_task_gets_its_own_tier_model(self):
        h = _Harness({"model": "", "provider": "", "tiers": TIERS})
        result = h.run([{"goal": GOAL_A, "tier": "executar"}, {"goal": GOAL_B, "tier": "descobrir"},
                        {"goal": GOAL_B + " again"}])
        self.assertNotIn("error", result)
        self.assertEqual([b["model"] for b in h.built], ["claude-sonnet-5", "claude-opus-5-5", None])

    def test_tier_name_is_case_insensitive(self):
        h = _Harness({"tiers": TIERS})
        h.run([{"goal": GOAL_A, "tier": "  Executar "}, {"goal": GOAL_B}])
        self.assertEqual(h.built[0]["model"], "claude-sonnet-5")

    def test_unknown_tier_refuses_the_batch_before_building(self):
        h = _Harness({"tiers": TIERS})
        result = h.run([{"goal": GOAL_A, "tier": "cheap"}, {"goal": GOAL_B}])
        self.assertIn("error", result)
        self.assertIn("unknown tier", result["error"])
        self.assertIn("executar", result["error"])
        self.assertEqual(h.built, [])

    def test_tier_without_configured_tiers_refuses(self):
        h = _Harness({"model": ""})
        result = h.run([{"goal": GOAL_A, "tier": "executar"}, {"goal": GOAL_B}])
        self.assertIn("error", result)
        self.assertIn("delegation.tiers is not configured", result["error"])
        self.assertEqual(h.built, [])

    def test_tier_route_does_not_inherit_the_default_endpoint(self):
        cfg = {"model": "default-model", "base_url": "https://default.example/v1", "api_key": "k",
               "tiers": {"executar": {"model": "claude-sonnet-5"}}}
        h = _Harness(cfg)
        h.run([{"goal": GOAL_A, "tier": "executar"}, {"goal": GOAL_B}])
        tier_child, default_child = h.built
        self.assertEqual(tier_child["model"], "claude-sonnet-5")
        self.assertIsNone(tier_child["override_base_url"])
        self.assertEqual(default_child["model"], "default-model")
        self.assertEqual(default_child["override_base_url"], "https://default.example/v1")

    def test_tier_keeps_non_route_delegation_settings(self):
        cfg = {"fallback_providers": [{"provider": "x", "model": "y"}], "tiers": TIERS}
        h = _Harness(cfg)
        h.run([{"goal": GOAL_A, "tier": "executar"}, {"goal": GOAL_B}])
        routing = h.built[0]["routing_cfg"]
        self.assertEqual(routing["fallback_providers"], cfg["fallback_providers"])
        self.assertNotIn("tiers", routing)
        self.assertNotIn("use_for", routing)


class TestTierConfigParsing(unittest.TestCase):
    def test_entries_without_a_route_are_dropped(self):
        tiers = delegate_tool_config._get_delegation_tiers(
            {"tiers": {"ok": {"model": "m"}, "empty": {"use_for": "x"}, "bad": "claude", "": {"model": "m"}}})
        self.assertEqual(sorted(tiers), ["ok"])

    def test_missing_or_malformed_section_is_empty(self):
        self.assertEqual(delegate_tool_config._get_delegation_tiers({}), {})
        self.assertEqual(delegate_tool_config._get_delegation_tiers({"tiers": ["executar"]}), {})


class TestTierSchema(unittest.TestCase):
    def _tier_prop(self, cfg):
        with patch.object(delegate_tool_config, "_load_config", return_value=cfg), \
             patch("tools.delegate_tool_config._get_independent_completions", return_value=False):
            schema = delegate_tool._build_dynamic_schema_overrides()
        return schema["parameters"]["properties"]["tasks"]["items"]["properties"].get("tier")

    def test_tier_is_advertised_with_the_configured_names(self):
        prop = self._tier_prop({"tiers": TIERS})
        self.assertEqual(prop["enum"], ["descobrir", "executar", "varrer"])
        self.assertIn("mechanical collection", prop["description"])

    def test_tier_is_hidden_without_configured_tiers(self):
        self.assertIsNone(self._tier_prop({}))

    def test_static_schema_is_not_mutated(self):
        self._tier_prop({"tiers": TIERS})
        static_items = delegate_tool.DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]
        self.assertNotIn("tier", static_items["properties"])


if __name__ == "__main__":
    unittest.main()

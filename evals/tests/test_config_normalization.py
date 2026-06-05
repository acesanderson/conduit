"""Tests for config_to_canonical_dict — the helper that lets RoutingSummarizer
(whose config holds Pydantic models with `type` fields) flow through the eval
harness's hash + persist path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals import config_to_canonical_dict
from runner import _config_id


class _Inner(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    target: type


class _Outer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    label: str
    inner: _Inner


def test_primitive_pass_through():
    assert config_to_canonical_dict("x") == "x"
    assert config_to_canonical_dict(3) == 3
    assert config_to_canonical_dict(2.5) == 2.5
    assert config_to_canonical_dict(True) is True
    assert config_to_canonical_dict(None) is None


def test_type_becomes_qualified_string():
    class _Foo:
        pass

    canon = config_to_canonical_dict(_Foo)
    assert isinstance(canon, str)
    assert canon.endswith("_Foo")


def test_basemodel_becomes_recursive_dict():
    inner = _Inner(name="a", target=int)
    canon = config_to_canonical_dict(inner)
    assert canon == {"name": "a", "target": "builtins.int"}


def test_nested_basemodel_recurses():
    outer = _Outer(label="L", inner=_Inner(name="a", target=str))
    canon = config_to_canonical_dict(outer)
    assert canon == {"label": "L", "inner": {"name": "a", "target": "builtins.str"}}


def test_list_of_tuples_with_basemodel():
    routing = [(100, _Inner(name="tier1", target=int)), (1000, _Inner(name="tier2", target=str))]
    canon = config_to_canonical_dict(routing)
    assert canon == [
        [100, {"name": "tier1", "target": "builtins.int"}],
        [1000, {"name": "tier2", "target": "builtins.str"}],
    ]
    # And it round-trips through json.dumps.
    json.dumps(canon, sort_keys=True)


def test_config_id_stable_across_equivalent_routing():
    routing_a = [(10, _Inner(name="x", target=int))]
    routing_b = [(10, _Inner(name="x", target=int))]
    cid_a = _config_id({"routing": routing_a, "tokenizer_encoding": "cl100k_base"})
    cid_b = _config_id({"routing": routing_b, "tokenizer_encoding": "cl100k_base"})
    assert cid_a == cid_b


def test_config_id_changes_when_routing_changes():
    cid_a = _config_id({"routing": [(10, _Inner(name="x", target=int))]})
    cid_b = _config_id({"routing": [(10, _Inner(name="x", target=str))]})
    assert cid_a != cid_b


def test_routing_summarizer_production_config_hashes():
    """The actual PRODUCTION_ROUTING value must hash without raising.

    This is the original bug: SummarizationProfile.strategy_cls is a
    `type[SummarizationStrategy]`, which json.dumps can't serialize.
    """
    from conduit.strategies.summarize.summarizers.routing import PRODUCTION_ROUTING

    cid = _config_id({"routing": PRODUCTION_ROUTING, "tokenizer_encoding": "cl100k_base"})
    assert isinstance(cid, str)
    assert len(cid) == 8

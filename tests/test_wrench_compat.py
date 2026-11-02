"""检查跨 Isaac Lab 版本的外力施加封装；不启动 Isaac Sim，也不需要 torch。"""

from __future__ import annotations

import unittest

from ta_sru.envs.wrench import set_body_wrench


class _NewAsset:
    """2.3.2+：新接口与废弃接口同时存在，应优先走 composer。"""

    def __init__(self):
        self.composer_calls = []
        self.legacy_calls = []
        self.permanent_wrench_composer = self

    def set_forces_and_torques(self, **kwargs):
        self.composer_calls.append(kwargs)

    def set_external_force_and_torque(self, *args, **kwargs):  # pragma: no cover
        self.legacy_calls.append((args, kwargs))


class _OldAsset:
    """2.0–2.3.1：只有旧接口。"""

    def __init__(self):
        self.calls = []

    def set_external_force_and_torque(self, forces, torques, body_ids=None, is_global=False):
        self.calls.append(
            {"forces": forces, "torques": torques, "body_ids": body_ids, "is_global": is_global}
        )


class _IsaacLab14Asset:
    """1.4：旧接口没有 is_global，多传关键字会直接 TypeError。"""

    def __init__(self):
        self.seen = None

    def set_external_force_and_torque(self, forces, torques, body_ids=None, env_ids=None):
        self.seen = (forces, torques, body_ids, env_ids)


class _BrokenAsset:
    pass


class SetBodyWrenchTest(unittest.TestCase):
    def test_prefers_composer(self):
        asset = _NewAsset()
        set_body_wrench(asset, "forces", "torques", body_ids=[0])
        self.assertEqual(
            asset.composer_calls,
            [{"forces": "forces", "torques": "torques", "body_ids": [0]}],
        )
        self.assertEqual(asset.legacy_calls, [])

    def test_composer_without_setter_falls_back(self):
        class AssetWithOtherComposer:
            permanent_wrench_composer = object()

            def __init__(self):
                self.called = False

            def set_external_force_and_torque(self, forces, torques, body_ids=None):
                self.called = True

        asset = AssetWithOtherComposer()
        set_body_wrench(asset, "forces", "torques")
        self.assertTrue(asset.called)

    def test_falls_back_to_legacy(self):
        asset = _OldAsset()
        set_body_wrench(asset, "forces", "torques", body_ids=[0])
        self.assertEqual(
            asset.calls,
            [{"forces": "forces", "torques": "torques", "body_ids": [0], "is_global": False}],
        )

    def test_legacy_call_omits_is_global_for_isaaclab_14(self):
        asset = _IsaacLab14Asset()
        set_body_wrench(asset, "forces", "torques", body_ids=[0])
        self.assertEqual(asset.seen, ("forces", "torques", [0], None))

    def test_missing_interface_raises(self):
        with self.assertRaises(AttributeError):
            set_body_wrench(_BrokenAsset(), "forces", "torques")


if __name__ == "__main__":
    unittest.main()

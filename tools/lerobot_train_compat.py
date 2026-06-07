#!/usr/bin/env python3
"""兼容部分 LeRobot 版本中 policy feature CLI 字典未转成对象的问题。"""

from __future__ import annotations

import sys
from types import SimpleNamespace


def _feature_type_value(feature):
    if isinstance(feature, dict):
        value = feature.get("type")
    else:
        value = getattr(feature, "type", None)
    return getattr(value, "value", value)


def _as_feature(feature):
    if isinstance(feature, dict):
        return SimpleNamespace(**feature)
    return feature


def _patch_policy_config_features() -> None:
    try:
        from lerobot.configs.policies import PreTrainedConfig
    except Exception:
        return

    def image_features(self):
        features = {}
        for key, feature in getattr(self, "input_features", {}).items():
            if _feature_type_value(feature) == "VISUAL":
                features[key] = _as_feature(feature)
        return features

    def robot_state_feature(self):
        for key in ("observation.state", "observation.environment_state", "state"):
            feature = getattr(self, "input_features", {}).get(key)
            if feature is not None:
                return _as_feature(feature)
        for feature in getattr(self, "input_features", {}).values():
            if _feature_type_value(feature) == "STATE":
                return _as_feature(feature)
        return None

    def env_state_feature(self):
        feature = getattr(self, "input_features", {}).get("observation.environment_state")
        if feature is not None:
            return _as_feature(feature)
        return None

    def action_feature(self):
        feature = getattr(self, "output_features", {}).get("action")
        if feature is not None:
            return _as_feature(feature)
        for feature in getattr(self, "output_features", {}).values():
            if _feature_type_value(feature) == "ACTION":
                return _as_feature(feature)
        return None

    PreTrainedConfig.image_features = property(image_features)
    PreTrainedConfig.robot_state_feature = property(robot_state_feature)
    PreTrainedConfig.env_state_feature = property(env_state_feature)
    PreTrainedConfig.action_feature = property(action_feature)


def _set_nested(mapping, dotted_key, value):
    current = mapping
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _patch_observation_processor() -> None:
    try:
        from lerobot.processor.pipeline import ObservationProcessorStep
    except Exception:
        return

    original_call = ObservationProcessorStep.__call__

    def patched_call(self, transition):
        aliases = {}
        if "action" not in transition and "actions" in transition:
            aliases["action"] = transition["actions"]
        if "observation" not in transition:
            observation = {}
            for key, value in list(transition.items()):
                if key.startswith("observation."):
                    _set_nested(observation, key[len("observation.") :], value)
                elif key == "image":
                    _set_nested(observation, "images.image", value)
                    aliases["observation.images.image"] = value
                elif key == "wrist_image":
                    _set_nested(observation, "images.wrist_image", value)
                    aliases["observation.images.wrist_image"] = value
                elif key == "state":
                    _set_nested(observation, "state", value)
                    aliases["observation.state"] = value
            if observation:
                transition = dict(transition)
                transition["observation"] = observation
                transition.update({k: v for k, v in aliases.items() if k not in transition})
            elif not getattr(self, "_compat_printed_keys", False):
                print(f"[compat] transition keys: {list(transition.keys())}", file=sys.stderr)
                self._compat_printed_keys = True
        elif aliases:
            transition = dict(transition)
            transition.update({k: v for k, v in aliases.items() if k not in transition})
        return transition

    ObservationProcessorStep.__call__ = patched_call


def _patch_act_forward_aliases() -> None:
    try:
        import lerobot.policies.act.modeling_act as act_module
        from lerobot.policies.act.modeling_act import ACTPolicy
    except Exception:
        return

    original_forward = ACTPolicy.forward

    def patched_forward(self, batch, *args, **kwargs):
        batch = _add_training_aliases(batch)
        batch = _ensure_action_chunk(batch, self, act_module)
        if "observation.images.image" not in batch and not getattr(self, "_compat_printed_forward_keys", False):
            print(f"[compat] forward batch keys: {list(batch.keys())}", file=sys.stderr)
            self._compat_printed_forward_keys = True
        return original_forward(self, batch, *args, **kwargs)

    ACTPolicy.forward = patched_forward


def _tensor_shape(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(shape)


def _repeat_or_pad_action(action, chunk_size: int):
    ndim = getattr(action, "ndim", None)
    if ndim is None:
        return action
    if ndim == 1:
        action = action.unsqueeze(0).unsqueeze(1)
    elif ndim == 2:
        action = action.unsqueeze(1)
    elif ndim != 3:
        return action

    seq_len = action.shape[1]
    if seq_len == chunk_size:
        return action
    if seq_len == 1:
        return action.repeat(1, chunk_size, 1)
    if seq_len < chunk_size:
        pad = action[:, -1:, :].repeat(1, chunk_size - seq_len, 1)
        return action.new_empty((action.shape[0], chunk_size, action.shape[2])).copy_(
            __import__("torch").cat([action, pad], dim=1)
        )
    return action[:, :chunk_size, :]


def _ensure_action_chunk(batch, policy=None, act_module=None):
    if not isinstance(batch, dict):
        return batch

    action_key = getattr(act_module, "ACTION", "action") if act_module is not None else "action"
    action = batch.get(action_key)
    source_key = None
    if action is None:
        for candidate in ("actions", "action"):
            if candidate != action_key and batch.get(candidate) is not None:
                action = batch[candidate]
                source_key = candidate
                break

    if action is None:
        if not getattr(policy, "_compat_printed_missing_action", False):
            shapes = {key: _tensor_shape(value) for key, value in batch.items()}
            print(f"[compat] action is None; batch shapes: {shapes}", file=sys.stderr)
            if policy is not None:
                policy._compat_printed_missing_action = True
        return batch

    chunk_size = int(getattr(getattr(policy, "config", None), "chunk_size", 1) or 1)
    fixed_action = _repeat_or_pad_action(action, chunk_size)

    if fixed_action is not action or source_key is not None or batch.get(action_key) is None:
        batch = dict(batch)
        batch[action_key] = fixed_action
        batch["action"] = fixed_action
        batch.setdefault("actions", fixed_action)
        if not getattr(policy, "_compat_printed_action_repair", False):
            print(
                "[compat] repaired action chunk: "
                f"source={source_key or action_key}, before={_tensor_shape(action)}, "
                f"after={_tensor_shape(fixed_action)}",
                file=sys.stderr,
            )
            if policy is not None:
                policy._compat_printed_action_repair = True

    try:
        import torch

        pad_key = getattr(act_module, "ACTION_IS_PAD", "action_is_pad") if act_module is not None else "action_is_pad"
        if batch.get(pad_key) is None and getattr(fixed_action, "ndim", None) == 3:
            batch[pad_key] = torch.zeros(
                fixed_action.shape[0],
                fixed_action.shape[1],
                dtype=torch.bool,
                device=fixed_action.device,
            )
            batch.setdefault("action_is_pad", batch[pad_key])
    except Exception:
        pass

    return batch


def _add_training_aliases(batch):
    if not isinstance(batch, dict):
        return batch
    additions = {}
    if "observation.images.image" not in batch and "image" in batch:
        additions["observation.images.image"] = batch["image"]
    if "observation.images.wrist_image" not in batch and "wrist_image" in batch:
        additions["observation.images.wrist_image"] = batch["wrist_image"]
    if "observation.state" not in batch and "state" in batch:
        additions["observation.state"] = batch["state"]
    if ("action" not in batch or batch.get("action") is None) and batch.get("actions") is not None:
        additions["action"] = batch["actions"]
    if additions:
        batch = dict(batch)
        batch.update(additions)
    return batch


def _patch_processor_pipeline_preserve_dataset_keys() -> None:
    try:
        import lerobot.processor.pipeline as pipeline_module
    except Exception:
        return

    for class_name in dir(pipeline_module):
        cls = getattr(pipeline_module, class_name, None)
        if not isinstance(cls, type):
            continue
        if not hasattr(cls, "__call__") or not hasattr(cls, "_forward"):
            continue
        if getattr(cls, "_compat_preserve_patched", False):
            continue
        original_call = cls.__call__

        def patched_call(self, transition, _original_call=original_call):
            original_transition = transition
            result = _original_call(self, transition)
            if isinstance(original_transition, dict) and isinstance(result, dict):
                result = dict(result)
                for key in ("image", "wrist_image", "state", "actions"):
                    if key in original_transition and key not in result:
                        result[key] = original_transition[key]
                result = _add_training_aliases(result)
            return result

        cls.__call__ = patched_call
        cls._compat_preserve_patched = True


_patch_policy_config_features()
_patch_observation_processor()
_patch_act_forward_aliases()
_patch_processor_pipeline_preserve_dataset_keys()

from lerobot.scripts.lerobot_train import main


if __name__ == "__main__":
    main()

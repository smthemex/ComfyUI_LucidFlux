"""Context-parallel 占位实现。

本移植只做单卡推理：`PixDiT`/`PidNet` 的 `_cp_group` 恒为 None，
下面两个函数在 CP 关闭时不会被调用（被调用说明误开了 CP）。
"""


def split_inputs_cp(x, seq_dim, cp_group):
    raise RuntimeError("context parallel is not supported in this port (cp_group must be None)")


def cat_outputs_cp_with_grad(x, seq_dim, cp_group):
    raise RuntimeError("context parallel is not supported in this port (cp_group must be None)")

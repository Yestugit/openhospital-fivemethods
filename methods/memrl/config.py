"""MemRL 用户配置；其余算法设置沿用当前集成的 HLE 预设。"""
from dataclasses import dataclass

from mematlas.core.config import MethodConfig
from mematlas.methods.manifest import load_method_descriptor


_DESCRIPTOR = load_method_descriptor(__file__)
_DEFAULTS = _DESCRIPTOR.config_schema["x-variants"]["default"]


@dataclass(frozen=True, kw_only=True)
class MemRLConfig:
    """MemRL 配置，全部字段可省略，类型和默认值见构造签名。

    Attributes:
        batch_size: 每个记忆归属积累到此经历数量时自动演化，未满批继续积累。
        top_k: 默认检索条目上限，单次 search(top_k=...) 可覆盖。

    其余算法设置沿用当前集成的 HLE 预设。
    模型服务、记忆归属与运行目录填写公共 MemAtlasConfig。
    """

    batch_size: int = _DEFAULTS["batch_size"]
    top_k: int = _DEFAULTS["topk"]

    def _to_method_config(self) -> MethodConfig:
        return MethodConfig("memrl", {"batch_size": self.batch_size, "topk": self.top_k})

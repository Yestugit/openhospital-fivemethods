"""MemoryOS 用户配置；其余容量与检索规则沿用现有方法默认值。"""
from dataclasses import asdict, dataclass

from mematlas.core.config import MethodConfig
from mematlas.methods.manifest import load_method_descriptor


_DESCRIPTOR = load_method_descriptor(__file__)
_PARAMETERS = _DESCRIPTOR.config_schema["properties"]


@dataclass(frozen=True, kw_only=True)
class MemoryOSConfig:
    """MemoryOS 配置，全部字段可省略，类型和默认值见构造签名。

    Attributes:
        short_term_capacity: 短期问答容量，满后迁移最旧问答到中期。
        retrieval_queue_capacity: 中期检索页面上限；返回内容还可能包含短期与长期记忆。
        mid_term_heat_threshold: 长期画像／知识提炼的热度门槛，写入时自动检查。

    其余容量与检索规则沿用方法默认值。
    模型服务、记忆归属与运行目录填写公共 MemAtlasConfig。
    """

    short_term_capacity: int = _PARAMETERS["short_term_capacity"]["default"]
    retrieval_queue_capacity: int = _PARAMETERS["retrieval_queue_capacity"]["default"]
    mid_term_heat_threshold: float = _PARAMETERS["mid_term_heat_threshold"]["default"]

    def _to_method_config(self) -> MethodConfig:
        return MethodConfig("memoryos", asdict(self))

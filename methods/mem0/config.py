"""Mem0 用户配置；模型服务、记忆归属与运行目录使用公共配置。"""
from dataclasses import asdict, dataclass
from typing import Literal

from mematlas.core.config import MethodConfig
from mematlas.methods.manifest import load_method_descriptor


_DESCRIPTOR = load_method_descriptor(__file__)
_PARAMETERS = _DESCRIPTOR.config_schema["properties"]


@dataclass(frozen=True, kw_only=True)
class Mem0Config:
    """Mem0 配置，全部字段可省略，类型和默认值见构造签名。

    Attributes:
        top_k: 默认检索条目上限，单次 search(top_k=...) 可覆盖。
        write_policy: 内部事件缓冲的处理边界；"turn" 为助手回复，
            "episode" 为完整单元/任务。公共 add 提交的单元即时调用原生批处理。

    默认 top_k=20，采用固定提交的官方 additive 模式。
    不支持旧 classic/offline 简化实现及其检查点。
    模型服务、记忆归属与运行目录填写公共 MemAtlasConfig。
    """

    top_k: int = _PARAMETERS["top_k"]["default"]
    write_policy: Literal["turn", "episode"] = _PARAMETERS["write_policy"]["default"]

    def _to_method_config(self) -> MethodConfig:
        return MethodConfig("mem0", asdict(self))

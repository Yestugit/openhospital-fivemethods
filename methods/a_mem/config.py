"""A-Mem 用户配置；模型服务、记忆归属与运行目录使用公共配置。"""
from dataclasses import asdict, dataclass
from typing import Literal

from mematlas.core.config import MethodConfig
from mematlas.methods.manifest import load_method_descriptor

_DESCRIPTOR = load_method_descriptor(__file__)
_PARAMETERS = _DESCRIPTOR.config_schema["properties"]


@dataclass(frozen=True, kw_only=True)
class AMemConfig:
    """A-Mem 配置，全部字段可省略，类型和默认值见构造签名。

    Attributes:
        top_k: 默认检索条目上限，单次 search(top_k=...) 可覆盖。
        write_policy: "event" 在每条事件到达时形成笔记；"episode" 在 learn 时
            合并尚未写入的经历事件。两者沿用相同公共调用流程。
        evo_threshold: 累计成功演化达到此次数时重建全库索引，不是演化触发间隔。
        neighbor_k: 新笔记写入时，供关联判断与自动演化参考的近邻数量。
        agentic_search: True 时先返回直接命中的笔记，再追加尚未出现的关联笔记。

    默认按事件写入；"episode" 是框架提供的经历粒度变体。
    模型服务、记忆归属与运行目录填写公共 MemAtlasConfig。
    """

    top_k: int = _PARAMETERS["top_k"]["default"]
    write_policy: Literal["event", "episode"] = _PARAMETERS["write_policy"]["default"]
    evo_threshold: int = _PARAMETERS["evo_threshold"]["default"]
    neighbor_k: int = _PARAMETERS["neighbor_k"]["default"]
    agentic_search: bool = _PARAMETERS["agentic_search"]["default"]

    def _to_method_config(self) -> MethodConfig:
        return MethodConfig("a-mem", asdict(self))

"""AWM 用户配置；模型服务、记忆归属与运行目录使用公共配置。"""
from dataclasses import asdict, dataclass
from typing import Literal
from mematlas.core.config import MethodConfig
from mematlas.methods.manifest import load_method_descriptor

_DESCRIPTOR = load_method_descriptor(__file__)
_FIELDS = _DESCRIPTOR.config_schema["properties"]


@dataclass(frozen=True,kw_only=True)
class AWMConfig:
    """AWM 配置，profile 必填，其余字段的类型和默认值见构造签名。

    Attributes:
        profile: 使用的任务流程，"mind2web" 或 "webarena"；"medicine" 为临床变体。
        induction_mode: "online" 在 learn 时按批次归纳；"offline" 累积标注轨迹，
            在带 end_of_training=True 的 learn 中统一归纳。两者均需要 LLM。
        induce_steps: online 模式下，每个网站累计多少份合格经历后归纳一次；
            训练结束标记会处理不足一批的剩余轨迹。
        num_samples: WebArena 每个真实 template_id 最多抽取的成功经历数，其他流程不使用。
        seed: WebArena 经历抽样的局部随机种子，不改变公共随机状态。

    任务与轨迹使用公共 MemoryEpisode，网站和动作字段查看 AddOptions/EventOptions。
    模型服务、记忆归属与运行目录填写公共 MemAtlasConfig。
    """
    profile: Literal["mind2web", "webarena", "medicine"]
    induction_mode: Literal["online", "offline"] = _FIELDS["induction_mode"]["default"]
    induce_steps: int = _FIELDS["induce_steps"]["default"]
    num_samples: int = _FIELDS["num_samples"]["default"]
    seed: int = _FIELDS["seed"]["default"]

    def _to_method_config(self) -> MethodConfig:
        return MethodConfig("awm", asdict(self))

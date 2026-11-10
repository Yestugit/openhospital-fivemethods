"""ExpeL 用户配置；模型服务、记忆归属与运行目录使用公共配置。"""
from dataclasses import asdict, dataclass
from typing import Literal
from mematlas.core.config import MethodConfig
from mematlas.methods.manifest import load_method_descriptor

_DESCRIPTOR = load_method_descriptor(__file__)
_PARAMETERS = _DESCRIPTOR.config_schema['properties']

@dataclass(frozen=True, kw_only=True)
class ExpeLConfig:
    """ExpeL 配置，benchmark 必填，其余字段的类型和默认值见构造签名。

    Attributes:
        benchmark: 提示词与轨迹格式使用的任务配置，"hotpotqa"、"alfworld" 或
            "webshop"；"medicine" 为临床变体。此字段不会自动安装或启动任务环境。
        max_num_rules: 规则数量达到此阈值时调整提炼提示，不是规则数硬上限。
        success_critique_num: 成功经历分组时的目标组大小上限。
        fewshot_strategy: 成功示例的检索策略，默认 "task_similarity"，采用未归一化
            向量的 FAISS L2 距离；"none" 只使用规则，无需 Embedding。
        num_fewshots: 成功示例上限，不限制规则数。None 使用任务默认值：
            hotpotqa 为 6，alfworld、webshop 和 medicine 为 2。
        buffer_retrieve_ratio: 检索候选文档数相对示例数的倍数，候选随后按任务去重。
        max_fewshot_tokens: 单条示例轨迹的 token 上限；None 使用任务内置示例的长度上限。
        reranker: 候选重排方式，"none"、"len"、"thought" 或 "task"；
            后两者使用候选任务的 thought 文档计算相似度。
        seed: 抽样使用的局部随机种子，冻结检索不推进随机状态。

    训练需要真实成败反馈；最后一份训练经历的结束标记查看 AddOptions。
    模型服务、记忆归属与运行目录填写公共 MemAtlasConfig。
    """
    benchmark: Literal['hotpotqa', 'alfworld', 'webshop', 'medicine']
    max_num_rules: int = _PARAMETERS['max_num_rules']['default']
    success_critique_num: int = _PARAMETERS['success_critique_num']['default']
    fewshot_strategy: str = _PARAMETERS['fewshot_strategy']['default']
    num_fewshots: int | None = None
    buffer_retrieve_ratio: int = _PARAMETERS['buffer_retrieve_ratio']['default']
    max_fewshot_tokens: int | None = None
    reranker: Literal['none', 'len', 'thought', 'task'] = _PARAMETERS['reranker']['default']
    seed: int = _PARAMETERS['seed']['default']

    def _to_method_config(self) -> MethodConfig:
        return MethodConfig('expel', {k:v for k,v in asdict(self).items() if v is not None})

"""EMem（KevinSRR/EMem 官方固定源码包装）。

必要服务
    runtime.source_roots.emem 指向 source.json 固定版本源码根；独立 Python
    环境按 requirements.txt 安装。公共 providers 配置 LLM 与 Embedding。
    保留原生 CacheOpenAI，LLM 的 api_key_env 必须有值；不下载本地模型。
必要数据与反馈
    文本 MemoryUnit；role、event_id、timestamp 沿用真实轨迹，无需奖励。
    两方会话映射：assistant→Doctor，其余→Environment，原始角色保留在文本。
    同一任务可多次 add，episode_ref 关联任务；无专有 extensions 必填项。
调用流程
    index 查看 edus 视图；search→Agent→add→episode_end。完整 experience
    可在任务边界后提交；已处理事件按事件ID去重。flush 提交未完成缓冲，
    save/read_only 不触发学习；公共Memory关闭时按运行时约定自动flush。
    需保存未完成缓冲时在关闭前save。search返回原始EDU及来源元数据，供医生上下文使用，
    不执行上游QA或替代医生答题。保留官方 no-PPR 占位分数问题并不对外伪报分数。
    save/load 保存全部原生图、内容、向量、OpenIE/LLM缓存和适配缓冲；
    原生pickle检查点仅加载可信本地输出。见 example.py 与 source.json 的精确接入差异。
"""
from .config import EMemConfig, _DESCRIPTOR
from .method import PLUGIN

SUPPORTED_OPERATIONS = tuple(sorted(operation.value for operation in _DESCRIPTOR.capabilities.operations))
__all__ = ["EMemConfig", "SUPPORTED_OPERATIONS"]

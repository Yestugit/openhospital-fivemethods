"""Memora（microsoft/Memora 官方固定源码包装）。

必要服务
    runtime.source_roots.memora 指向 source.json 指定的官方源码根，包含官方
    app/locomo/providers/memora/segmentation.py；独立环境安装 requirements.txt。
    公共 providers 配置 LLM/Embedding，LLM 必须支持上游Pydantic结构化输出。
必要数据与反馈
    文本 MemoryUnit 和真实 role/event_id/timestamp；无需成功标签或奖励。
    多次add沿用episode_ref。无必填专有extensions；不启用图像或GRPO训练。
调用流程
    index 查看 memories 视图；search→Agent→add→episode_end。内部按完整
    任务调用官方分段、ChatMemoryBuilder、Chroma主索引/线索锚点/情景记忆。
    默认官方应用profile：cue与BM25混合检索，PromptedPolicyRetriever四步
    EXPAND/RE_QUERY/STOP；top_k为初始候选数，工作集扩展与情景关联可能返回更多。
    search返回事实与相关情景上下文，由原医生流程使用；不替换医生答题。
    flush提交剩余会话，不推断成功；save/read_only不触发学习。
    公共Memory关闭时按运行时约定自动flush；需保存未完成缓冲时在关闭前save。
    完整逻辑快照含全部原始向量，恢复不重新嵌入。来源及传输/导入改动见source.json。
"""
from .config import MemoraConfig, _DESCRIPTOR
from .method import PLUGIN

SUPPORTED_OPERATIONS = tuple(sorted(operation.value for operation in _DESCRIPTOR.capabilities.operations))
__all__ = ["MemoraConfig", "SUPPORTED_OPERATIONS"]

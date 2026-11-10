"""MemMachine native integration.

必要服务：独立 Python 环境、LLM 与 embedding。原生 SQLite/USEARCH，不需要独立数据库服务。
必要数据与反馈：MemoryUnit 的真实事件、任务或输出；不要求奖励。
调用流程：index 查看视图；add 提交真实数据；search 返回原生层级记忆；episode_end 标记边界。
额外步骤：无需特有输入、特有命令或手动演化。正常写入保留原生批次门槛；flush 排空已接受的语义历史与短期摘要后恢复门槛。
检索结果用作医生上下文；applied_memory 保持来源，不作为新临床事实再次写入。
"""
from .config import MemMachineConfig, _DESCRIPTOR
from .method import PLUGIN
SUPPORTED_OPERATIONS = tuple(sorted(operation.value for operation in _DESCRIPTOR.capabilities.operations))
__all__ = ["MemMachineConfig", "SUPPORTED_OPERATIONS"]

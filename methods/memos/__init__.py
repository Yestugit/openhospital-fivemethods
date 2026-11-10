"""MemOS native integration.

必要服务：独立 Python 环境、LLM 与 embedding。另需 Neo4j >=5.18 与 NEO4J_PASSWORD。
Windows 本机可运行 scripts/local-memos-neo4j.py start/test 管理项目内的 Neo4j 5.26，
其 Bolt 端口为 17687；默认外部服务端口仍为 7687。同一数据库的原生向量索引必须维度一致。
必要数据与反馈：MemoryUnit 的真实事件、任务或输出；不要求奖励。
调用流程：index 查看视图；add 提交真实数据；search 返回原生层级记忆；episode_end 标记边界。
额外步骤：无需特有输入、特有命令或手动演化。同步 tree_text 的 add 已完成抽取、合并和清理，flush 无额外演化。
检索结果用作医生上下文；applied_memory 保持来源，不作为新临床事实再次写入。
"""
from .config import MemOSConfig, _DESCRIPTOR
from .method import PLUGIN
SUPPORTED_OPERATIONS = tuple(sorted(operation.value for operation in _DESCRIPTOR.capabilities.operations))
__all__ = ["MemOSConfig", "SUPPORTED_OPERATIONS"]

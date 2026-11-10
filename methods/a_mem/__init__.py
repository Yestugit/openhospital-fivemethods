"""A-Mem：带定向关联及自动邻居演化的记忆笔记。

必要服务
    需要 LLM、Embedding 和 chromadb；模型与密钥在公共 config.providers 中配置。
    先准备可导入的 mematlas 包，再安装：python -m pip install chromadb openai。

必要数据与反馈
    add 接收 MemoryUnit，通过 events 提交独立内容，无需奖励或成败标签。
    一个任务可提交多个单元；episode_ref 默认关联当前任务，任务结束后调用 episode_end。
    一份 Memory 只服务一个 scope；无专有调用参数或必要特有函数。
    applied_memory 仅回传实际使用的完整贡献。

调用流程
    async with Memory(config)：search → Agent → add → episode_end。
    默认每条事件分析为笔记，并自动判断关联和邻居上下文/标签更新；已处理事件去重。
    search 返回 MemoryContribution；优先将 context 提供给 Agent，包含正文、日期与语义元数据。
    items 保留笔记元数据和来源；演化来源记录输入依赖，不表示每条输入事实都保留在正文中。
    测评复用默认 QA，无需选择方法 QA。
    SUPPORTED_OPERATIONS 列出支持操作，签名查看公共 Memory；不支持 evolve。

额外步骤
    无需 flush；save/load 保存向量、笔记、关联、来源及去重状态，不触发演化。
    read_only() 用于冻结检索，reset 清理当前归属；连续任务复用同一 Memory。
    配置字段查看 help(AMemConfig)，完整接入见同目录 example.py（支持 --help）。
    当前为内容分析后 add_note 的组合流程重实现；来源和差异查看 MemAtlas().describe("a-mem")。
"""
from .config import AMemConfig, _DESCRIPTOR
from .method import PLUGIN

SUPPORTED_OPERATIONS = tuple(sorted(operation.value for operation in _DESCRIPTOR.capabilities.operations))

__all__ = ["AMemConfig", "SUPPORTED_OPERATIONS"]

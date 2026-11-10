"""MemoryOS：以短期对话、中期会话和长期画像／知识组织记忆。

必要服务
    需要 LLM、Embedding，在公共 config.providers 中配置模型与密钥。
    在项目根安装：python -m pip install -e ".[openai]" faiss-cpu
    默认密钥读取 OPENAI_API_KEY；自定义服务配置 base_url、api_key_env。

必要数据与反馈
    add 接收 MemoryUnit，包含单条消息、一轮或多轮对话，无需成败标签。
    events 使用 kind="message"，role="user" 或 "assistant"；系统消息跳过。
    episode_ref 默认关联当前任务，任务完成后调用 episode_end。
    已写入事件去重，缺失的问答一侧保留空文本。
    动作和工具结果需由 Agent 按实际语义整理为消息；无专有参数或必要特有函数。

调用流程
    async with Memory(config)：search → Agent → add → episode_end。
    助手回复后写入真实问答；短期满后迁移到中期，热度达标后自动提炼长期记忆。
    search 返回 MemoryContribution，优先使用 context 中的 content，并回传实际使用的完整贡献。
    中期队列容量只限制中期页面；显式 top_k 限制全部条目及其上下文。
    SUPPORTED_OPERATIONS 列出支持操作，签名查看公共 Memory；不支持公共 evolve。

额外步骤
    无需逐次 flush；它保存当前状态与待配对消息，save/load/close 均不触发演化。
    reset 清理当前归属；用户、Agent、会话组合隔离，read_only() 检索保持热度和画像不变。
    配置字段查看 help(MemoryOSConfig)，完整接入见同目录 example.py。
    当前为方法重实现；来源和差异查看 MemAtlas().describe("memoryos")。
"""
from .config import MemoryOSConfig, _DESCRIPTOR
from .method import PLUGIN

SUPPORTED_OPERATIONS = tuple(sorted(operation.value for operation in _DESCRIPTOR.capabilities.operations))

__all__ = ["MemoryOSConfig", "SUPPORTED_OPERATIONS"]

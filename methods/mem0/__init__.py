"""Mem0：从交互中提取长期记忆，按文本检索。

必要服务
    默认需要 LLM、Embedding；在公共 config.providers 中配置模型与密钥。
    在独立 Python 3.12 环境安装 methods/mem0/requirements-lock.txt；
    runtime.python_executable 指向该解释器，executor="subprocess"。
    providers.embedding.dimensions 必填；官方英语 NLP/BM25 初始化失败会阻止运行。
    source.json 固定官方提交与源码哈希，UPSTREAM_LICENSE 为上游许可。

必要数据与反馈
    add 接收 MemoryUnit，可含单条资料、一轮对话或完整经历，无需成败标签。
    events 消息填写 role；episode_ref 默认关联当前任务，任务完成后调用 episode_end。
    applied_memory 仅回传实际使用的完整贡献。
    单独提供的 task/output/feedback.text 未包含于轨迹时转为带来源的原生消息；
    延迟反馈只提交新内容，已处理轨迹和字段按稳定标识去重。
    图像事件还需 MethodConfig.params.enable_vision=true 和支持视觉的 LLM。
    无专有调用参数或必要特有函数；不支持 team_id。

调用流程
    async with Memory(config)：search → Agent → add → episode_end。
    每个 add 提交独立单元，内部提取并去重；连续任务复用 Memory。
    search 返回 MemoryContribution，将 items 中的 content 提供给 Agent。
    SUPPORTED_OPERATIONS 列出支持操作，签名查看公共 Memory；不支持 evolve。

额外步骤
    无需逐次 flush；save/load 使用完整数据库和精确向量状态的快照文件。
    复制检查点时须连同 state.native.snapshot_file 指向的文件保存；文件带 SHA256 校验。
    reset 清理指定完整 scope 的事实、实体链接与近期消息。
    冻结期间在 read_only() 中检索，Agent 自行执行。
    配置字段查看 help(Mem0Config)，完整接入见同目录 example.py。
    当前直接执行固定提交的官方 additive 方法；旧重实现和旧检查点明确拒绝。
    来源、原生默认值和限制查看 MemAtlas().describe("mem0")。
"""
from .config import Mem0Config, _DESCRIPTOR
from .method import PLUGIN

SUPPORTED_OPERATIONS = tuple(sorted(operation.value for operation in _DESCRIPTOR.capabilities.operations))

__all__ = ["Mem0Config", "SUPPORTED_OPERATIONS"]

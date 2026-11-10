"""ExpeL：从成功与失败经历提炼带计数的规则，并检索成功示例。

必要服务
    需要 LLM、faiss-cpu 和 tiktoken；fewshot_strategy 非 "none" 时还需 Embedding。
    模型与密钥在公共 config.providers 中配置。
    HotpotQA 工具执行另需 requests、wikipedia==1.4.0 及 Wikipedia 网络访问。

必要数据与反馈
    add 接收 MemoryUnit，任务学习需要 task、轨迹 events 和真实 feedback.success。
    同一任务可提交多个单元，以 episode_end 通知结束；一份 Memory 只服务一个 scope。
    专有数据放入 extensions['expel']，说明查看 AddOptions/SearchOptions。
    reflections 使用外部 Agent 实际生成的反思；applied_memory 仅回传实际使用的完整贡献。

调用流程
    async with Memory(config)：search → Agent → add → episode_end。
    最后一份训练经历设置 end_of_training=True，learn 内部按成败配对与成功分组提炼规则。
    测评自动使用 ExpeLQA 补训练结束标记；HotpotQA 采用 Search/Lookup/Finish 多步执行。
    search 返回 MemoryContribution，将 context 中的全部规则和成功轨迹提供给 Agent。
    items 保留来源、规则计数与距离；top_k 只限制成功示例数。
    规则来源记录提炼时的输入依赖，不表示每条输入事实都保留在规则中。
    SUPPORTED_OPERATIONS 列出支持操作，签名查看公共 Memory；不支持 evolve 或 flush。

额外步骤
    无必要特有函数；save/load 保存恢复批次、向量、规则与去重状态。
    冻结期间在 read_only() 中检索，Agent 自行执行；close/save/search 不触发提炼。
    普通 LoCoMo/LongMemEval 历史缺少成败反馈时不能直接训练，需另行提供真实反馈。
    配置字段查看 help(ExpeLConfig)，Agent 接入见同目录 example.py（支持 --help）。
    HotpotQA 接入见 hotpotqa_example.py，包含工具执行、失败反思重试与冻结测评。
    当前为经历学习与规则提炼流程重实现；来源和差异查看 MemAtlas().describe("expel")。
"""
from .config import ExpeLConfig, _DESCRIPTOR
from .method import AddOptions, SearchOptions, PLUGIN

SUPPORTED_OPERATIONS = tuple(sorted(op.value for op in _DESCRIPTOR.capabilities.operations))
__all__ = ['ExpeLConfig', 'AddOptions', 'SearchOptions', 'SUPPORTED_OPERATIONS']

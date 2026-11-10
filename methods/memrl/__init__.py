"""MemRL：从完整经历与真实反馈形成过程性记忆，以 Q 值辅助检索。

必要服务
    LLM、Embedding 使用公共 config.providers，向量维度按实际模型填写。
    默认密钥读取 OPENAI_API_KEY；自定义服务配置 base_url、api_key_env。
    MemOS 后端也需要相应凭据；依赖安装与本地验证见 test/memrl/README.md。
    原生源码已内置于 _vendor，无需另行下载，默认采用上游 HLE 预设。

必要数据与反馈
    add 接收 MemoryUnit，可分次提供 task、轨迹和 feedback.success（真实 bool）。
    episode_end 标记任务结束；信息齐全后按原生批次规则学习。
    迟到反馈通过 add(type="feedback") 关联原 episode_ref 或 write_ref。
    applied_memory 保留实际使用的完整贡献及 extensions.memrl，空结果也可有关联。
    专有输入查看 help(SearchOptions)、help(AddOptions)，均可省略；无必要特有函数。

调用流程
    async with Memory(config)：search → Agent → add → episode_end。
    search 返回 MemoryContribution，将 items 中的 content 提供给 Agent。
    完整任务经历满批后自动构建／修正记忆并更新 Q 值，未满批继续积累。
    有限训练的最后一条经历可填 extensions={"memrl": {"end_of_training": True}}，内部完成尾批。
    SUPPORTED_OPERATIONS 列出支持操作，签名查看公共 Memory；不支持 evolve、reset。

额外步骤
    无需逐次 flush；flush/save/load/close 不触发演化，保存的未满批经验可恢复后继续。
    检查点 JSON 引用原生快照目录，恢复时一起保留；在 read_only() 中冻结检索。
    配置字段查看 help(MemRLConfig)，完整接入与专有参数填写位置见 example.py。
    来源：MemTensor/MemRL@c1b322ca43de36ddf64c6712f89d0095bfc35ce0，MIT。
"""
from .config import MemRLConfig, _DESCRIPTOR
from .method import PLUGIN, AddOptions, SearchOptions

SUPPORTED_OPERATIONS = tuple(sorted(operation.value for operation in _DESCRIPTOR.capabilities.operations))

__all__ = ["MemRLConfig", "SearchOptions", "AddOptions", "SUPPORTED_OPERATIONS"]

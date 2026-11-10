"""AWM：从网站任务轨迹归纳工作流，供后续网页 Agent 使用。

必要服务
    需要 LLM；工作流归纳与检索无需 Embedding 或向量库。
    在公共 config.providers 中配置模型与密钥。
    Mind2Web 原生离线评测使用已注册 benchmark、带 rank 的官方数据和 tiktoken/lxml。
    实时网页测评另需外部环境适配器，负责浏览器、观察数据和真实成败反馈。

必要数据与反馈
    add 接收 MemoryUnit，提供 task、events 及已知反馈；以 episode_end 通知任务结束。
    Mind2Web 需要 website/domain/subdomain；online 使用实际 Agent 轨迹，offline 使用标注轨迹。
    WebArena 需要真实 success；成功经历还需 template_id。专有数据放入 extensions['awm']。
    公共 MemoryEpisode 没有单独的 trajectory 字段；约束中的 trajectory 由非空 events 表达。
    一份 Memory 只服务一个 scope；同一任务的单元使用同一个 episode_ref。
    applied_memory 仅回传实际使用的完整贡献；字段说明查看 AddOptions 和 EventOptions。

调用流程
    async with Memory(config)：search → Agent/工具 → add → episode_end。
    online 按完整任务数达到 induce_steps 自动归纳；offline 累积轨迹，在训练结束时统一归纳。
    最后一份训练经历设置 end_of_training=True，处理各网站剩余轨迹；online 同样使用此标记。
    search 通过 SearchOptions 指定 website；返回 MemoryContribution，将 items 中的 content 提供给 Agent。
    每个条目保留完整工作流文件与输入来源；top_k 限制文件数，单站最多一份，无语义排序。
    网页测评自动使用已注册的 AWMQA。Mind2Web 原生离线评测配置
    benchmark.kind='mind2web' 和带 rank 的数据 manifest 即可；
    ExperimentRunner 自动提供环境，mind2web_example.py 展示完整评测与记忆导出。
    其他网页任务需注入
    MethodServices.objects['awm_environment_factory']。
    工厂签名为
    factory(task: str, public_options: Mapping[str, Any]) -> AWMEnvironment；
    public_options 只含 profile 以及实际的 website/domain/subdomain。
    环境必须提供
    reset()、step(action: str)、close()，reset/step 返回 observation:str 和 done:bool，
    终止状态还必须返回真实 success:bool。
    环境负责浏览器、观察和成败评价，不能把参考答案或目标动作交给工厂以外的 Agent 提示。
    SUPPORTED_OPERATIONS 列出支持操作，签名查看公共 Memory；不支持 evolve 或 flush。

额外步骤
    无必要特有函数；save/load 保存恢复状态，save/load/close 不触发归纳。
    冻结期间在 read_only() 中检索，Agent 与环境在外部执行；连续任务复用同一 Memory。
    配置字段查看 help(AWMConfig)，专有输入查看 AddOptions/EventOptions/SearchOptions。
    AWMEnvironment、BrowserState、run_agent 和 run 也从包入口导出；同目录 example.py 展示
    search → Agent → add → episode_end，以及已完成轨迹导入，支持 --help。
    当前为网站工作流归纳流程重实现；来源和差异查看 MemAtlas().describe("awm")。
"""
from .config import AWMConfig,_DESCRIPTOR
from .method import AddOptions,EventOptions,SearchOptions,PLUGIN
from .environment import AWMEnvironment, BrowserState

SUPPORTED_OPERATIONS=tuple(sorted(op.value for op in _DESCRIPTOR.capabilities.operations))
__all__ = [
    'AWMConfig',
    'AddOptions',
    'EventOptions',
    'SearchOptions',
    'AWMEnvironment',
    'BrowserState',
    'run_agent',
    'run',
    'configure_environment_factory',
    'SUPPORTED_OPERATIONS',
]


def __getattr__(name):
    """Load example helpers only when requested, keeping ``python -m`` clean."""
    if name in {"run_agent", "run", "configure_environment_factory"}:
        from . import example
        return getattr(example, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

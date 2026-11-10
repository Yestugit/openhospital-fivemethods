# 四种记忆方法的接入差异（当前状态，2026-10-24）

## 合并后的修复

1. AEL 将策略决策和记忆条目分别记录。一次任务可以包含多次策略选择；空检索也保留已执行的决策。反馈按决策 ID 去重，记忆效用继续只更新实际使用的条目。普通 Agent 和 OpenHospital 的延迟反馈都保存这些记录。
2. 测评协议显式调度原生学习边界。MemRL 处理末尾不足一批的经验；MemQ 按配置运行训练轮次，并在每轮结束执行 Bellman 更新；AEL 按 bars_per_day 对完成的训练任务分组，在训练结束处理最后一个分组。OpenHospital 使用 experience plan 中明确指定的批次。commit 继续负责持久化。
3. experience 支持传入 training_agent，实际执行训练任务并接收本次执行的 AgentTrace 和评估反馈。默认历史导入保留旧轨迹及其标签；新检索不会自动获得旧轨迹的奖励。报告记录 agent 或 history_import 模式。AEL 的集成范围为记忆与控制器。
4. Git 固定随包源码为 LF 换行。生成的 results 和 .memos 已移出版本跟踪并加入忽略规则，历史提交中的已暴露凭据需要在服务商处撤销。

后续章节保留合并 PR 时的接入说明与核查记录；涉及本轮修复的行为以本节和当前方法包说明（`help(方法包)`）为准。——10.31

## 修复前的真实 CLI 核查（50200dc）

- 上游行为：四方法的完整学习流程不仅包含写入、读取，还包括已使用记忆的反馈，以及各 runner 明确触发的 batch、epoch 或日末边界。
- MemAtlas 行为：从 `fix/ael-integration@50200dc` 依序运行 MemRL → MemCon → MemQ → AEL，实际调用 `.venv/bin/python -m mematlas run` 和 SiliconFlow provider。九次 `experience` 默认/显式 smoke 对照均退出0，框架操作失败数为0；四次使用合法事件和 outcome 的 `streaming` 均在 start 阶段拒绝，退出1。
- 原因：离线原生回归不能替代公开 CLI 的生命周期验证；此次同时检查插件默认参数、明确声明的测试覆盖参数及原生最终状态。
- 行为影响：只能确认部分配置完成导入与冻结读取，不能判定四方法完整学习闭环验收通过。默认 MemRL/MemQ 两条训练记录未满批，原生库为空；AEL 默认策略在本次冷启动下两题均无命中。MemCon、MemRL `hle_episode`、MemQ batch1、AEL `full_detailed` 加匹配 metadata 能读出训练记忆，但没有验证训练检索后的 credit 更新。没有改动算法、底层框架、依赖或 main。
- 测试证据：[输入、配置、日志来源与可复现命令](test/cli_e2e/README.md)；完整本地产物 `.mematlas/runs/cli-e2e-20261024/`，只读汇总入口 `test/cli_e2e/summarize.py`。本次未独立采集冻结前后逻辑 state hash，也未验证完整 agent、论文分数或跨进程恢复；严格快照断言属于各方法原生自测的覆盖范围，不作为本次 CLI 结论。

## MemRL 接入差异

本节对应原生源码 `MemTensor/MemRL@c1b322ca43de36ddf64c6712f89d0095bfc35ce0`，完整 memory import closure 已随插件交付在 `methods/memrl/_vendor`，运行和测试不依赖 `plan/`。原生自测入口为 `test/memrl/check_memrl.py`，依赖安装与运行命令见 `test/memrl/README.md`。

### 1. Episode 协议映射

- 上游行为：runner 向原生 `add_memories()` 传入一批 task、benchmark 轨迹、外部 success 布尔值、已使用记忆 ID 和检索 query/score 对，并在对应 benchmark 的批次边界调用 `update_values()`。
- MemAtlas 行为：`learn()` 从 `EpisodeOutcome.metadata["task"]` 读取任务，要求外部布尔 `success`，将非空事件格式化为 `role: text` 后入队；达到该 profile 的批次大小或收到 `end_batch` 命令时，才把整批转给原生方法。HLE 写入的初始 `metadata.q_value` 为 success/failure 对应的 `1/0`。
- 原因：`BaseMemoryMethod.learn()` 接收的是 `MemoryEpisode`，而不是 benchmark runner 的独立参数；MemAtlas 没有单独的原生 MemRL task/trajectory 对象。
- 行为影响：保留外部反馈来源、原生 Q 更新公式和 HLE runner 的初始 Q；轨迹序列化采用框架表示，不重现各 benchmark 的环境封装。
- 测试证据：`test_native_add_retrieve_update_and_log_namespaces` 使用原生 add/retrieve/Q 更新，`test_hle_profiles_initialize_success_and_failure_like_native_runner` 对照 HLE 的 `1/0` 初始值。

### 2. 检索数量映射

- 上游行为：`retrieve_query(query, k, threshold)` 使用 `k` 限制相似查询键的数量，`rl_config.topk` 再控制最终的 value-aware 记忆数量。
- MemAtlas 行为：`k_retrieve` 和 `topk` 分别保留上游含义；如果设置了 `DecisionContext.top_k`，只会截断已经选出的结果，不会重新检索。
- 原因：框架提供一个单一的单次结果上限，而 MemRL 同时区分候选查询数量和最终选中记忆数量。
- 行为影响：框架请求的 top-k 较小时，返回数量可能少于原生选择数量；请求较大时也不会超过原生 `topk`。候选检索和原生排序不会重算。
- 测试证据：`test/memrl/check_memrl.py` 调用原生 query selector；adapter 将原生候选 query/score 对保存在 item metadata 中，credit 测试检查只转发实际使用的候选信息。

### 3. MemAtlas scope 到上游 user 的映射

- 上游行为：每个 `MemoryService` 只初始化一个 `user_id`；MemRL 不理解 MemAtlas 的 user/agent/team/session/run 组合 scope。
- MemAtlas 行为：adapter 根据配置前缀和完整规范化 `MemoryScope` 派生稳定的上游 `user_id`；不同 scope 使用独立的 `MemoryService` 和 MemOS/Qdrant 命名空间。
- 原因：不同框架作用域不能共享同一个上游记忆库，否则会互相检索到记忆。
- 行为影响：一个 MemAtlas 方法实例可以创建多个相互独立的原生存储；实际物理 user ID 不只是配置中的前缀。
- 测试证据：`test_explicit_batch_preserves_boundaries_and_separates_scopes` 检查不同 scope 独立服务与批次，`test_loading_empty_snapshot_drops_all_scope_state` 检查恢复空状态会移除之后新建的 scope。

### 4. 异步 provider 适配

- 上游行为：provider API 为同步接口；OpenAI provider 构造 script/keyword prompt，`BaseEmbedder` 提供文本分块与向量合并。MemOS 的 GeneralMemCube 另外读取自己的 provider 配置。
- MemAtlas 行为：adapter 向异步 `MethodServices` 委托 transport，保留 source prompt 和 `BaseEmbedder` 分块/合并。框架 `providers` 中显式提供的模型等设置覆盖 profile transport 默认值；显式 `method.params` 与 `providers` 冲突时继续报错。
- 原因：`MethodServices` 暴露异步 provider 协议，而上游 `MemoryService` 调用同步 provider 方法。
- 行为影响：HTTP client、凭据、调用计量属于框架；原生文本存储 backend 仍使用 MemOS JSON 配置。profile 声明官方模型名称，不复制上游配置中的私有 endpoint 或凭据。
- 测试证据：`test_native_add_retrieve_update_and_log_namespaces` 区分 `mematlas.methods.memrl.adapter` 和原生 `memrl.*` 日志；离线 fixture 只替换 I/O，保留 source builder、adjustment、selector 和 Q updater。

### 5. 原生快照引用

- 上游行为：MemRL checkpoint 是目录快照，包含 GeneralMemCube 转储、本地 Qdrant 文件、metadata 和本地 cache。
- MemAtlas 行为：`save_state()` 捕获原生快照，以原生文本 dump、cache 内容及 cache 键顺序形成稳定身份，返回不会被后续保存覆盖的 scope/快照引用，同时保存 adapter 待学习批次。`load_state()` 先复制快照到可写工作目录，再交给原生 loader。
- 原因：MemAtlas checkpoint 外壳把方法状态序列化为 JSON mapping，而上游数据库状态是文件目录。
- 行为影响：后续检索/加载不修改早先快照，恢复后再次保存相同逻辑状态可得到相同引用；保留原生 cache FIFO 顺序。JSON checkpoint 仍依赖本地原生目录，不能单独移植。
- 测试证据：`test_snapshots_are_immutable_and_state_is_stable_across_restores` 检查引用与文件不可变；`test_real_sqlite_reopen_and_repeated_native_snapshot_restores` 使用真实 GeneralMemCube、GeneralTextMemory、QdrantVecDB 和本地 Qdrant/SQLite 重复恢复。

### 6. 参数变体与实际 runner 行为

- 上游行为：上游没有一套适用于全部 benchmark 的默认实验参数。HLE 使用 batch 32、`k_retrieve=5`、`topk=3`、`tau/sim_threshold=0.25` 和归一化参数 `0.1856827586889267/0.09407906234264374`；ALFWorld、BCB、LLB 批次分别为 `32/25/5`。LLB YAML 的归一化值为 `0.39/0.14`，但对应 runner 未将其转给 service，因此实际使用 service 默认的 HLE 归一化值。
- MemAtlas 行为：`default`/`hle` 选择官方 HLE 实验参数；`service_defaults` 声明 public-config/service 参数组合并使用 batch 1；`hle_episode` 明确将 HLE 改为 batch 1。其余 benchmark profile 使用各自有效参数，LLB 系列采用 runner 实际使用的归一化值。
- 原因：框架必须选择一套默认 profile；本次以官方主实验 HLE 为默认，并区分 YAML 声明值与 runner 实际传参。
- 行为影响：默认学习等待 32 个同 scope episode 或显式结束批次。选择 `hle_episode` 会改变批内可见性和更新时机；`service_defaults` 的 `k_retrieve=1/topk=5/tau=0.35/alpha=0.3` 是保留的公共配置组合，不能称为 HLE 论文实验。LLB 继续保留 runner 的实际行为，没有将未生效的 YAML 值“修复”为新策略。
- 测试证据：参数来自固定 revision 的 `configs/rl_*_config.yaml` 及 `memrl/run/{hle_runner,alfworld_rl_runner,bcb_runner,llb_rl_runner}.py`；`test_default_automatic_batch_and_explicit_tail_flush` 验证 HLE 默认批次，profile/顺序测试对照原生执行路径。

### 7. 已检索记忆的 credit

- 上游行为：benchmark runner 在外部结果确定后，把运行记录中的已选记忆 ID 传给 `update_values`。
- MemAtlas 行为：adapter 只转发 `MemoryEpisode.retrieved` 中的 ID 和这些 item 携带的 query/score 对；框架在 contribution 被确认应用后填充这些 item。未确认候选不回退到最近一次检索结果。
- 原因：MemAtlas 区分 `retrieve()` 返回的候选和 episode 中明确确认应用的记忆。
- 行为影响：agent 必须确认实际使用的 contribution，相关 ID 才会收到上游 success/failure Q 更新；`EpisodeOutcome.success` 仍然是反馈来源。
- 测试证据：`test_unconfirmed_retrieval_does_not_reuse_query_candidates` 检查未应用候选不泄漏 query credit；原生 add/retrieve/update 测试向实际使用的 ID 反馈 failure，并验证 Q 更新结果。

### 8. 只读评估能力

- 上游行为：`retrieve_query()` 在读取候选时会填充 `query_embeddings`、`_q_cache` 和 `_mem_cache`；这些 cache 会包含在原生快照中。
- MemAtlas 行为：descriptor 声明 `read_only_mode=snapshot`；每次测试检索前保存原生 state，检索后恢复。
- 原因：上游检索会修改 cache，因此需要 snapshot/restore，既保留原生检索路径，又避免污染已学习的记忆库。
- 行为影响：框架管理的冻结评估可以在同一文件系统中工作；MemRL checkpoint JSON 仍然只是原生快照目录的引用，不是可移植副本。
- 测试证据：原生自测检查 cache/Qdrant 快照的多次保存恢复；本次 CLI trace 显示 `save_state -> retrieve -> load_state` 成功，但未独立采集前后 state hash，不能将其等同完整状态相等断言。

### 9. 非 3072 维 Qdrant patch

- 上游行为：固定版本的 MemRL source 把 GeneralMemCube Qdrant collection 默认设为 3072 维，并忽略构造器传入的 `vector_dimension`。
- MemAtlas 行为：adapter 转发 `embedding_dims`；bundled source 已应用原样 patch `test/memrl/memrl-vector-dimension.patch`，只新增维度属性并替换 live/restore 两处硬编码。配置非 3072 维时，source preflight 检查完整 patch 标记。
- 原因：embedding 接口允许其他维度，而上游该版本忽略传入的 `vector_dimension`，会与实际 provider 输出不匹配。
- 行为影响：默认 3072 维不变；bundled source 支持显式配置的其他维度，不需要用户再次应用 patch。仅在显式覆盖为未修改的外部 source 且配置非 3072 维时才需要应用它。此修改不改变检索排序、prompt 或更新公式；没有改写 `plan/`。
- 测试证据：交付 patch SHA-256 为 `9ad9876086e61d3c7fb4ed1bae103677e8f5866237e9d2608800e7af8bfc9210`；bundled 文件 SHA-256 由 `UPSTREAM.json` 校验，原生数据库自测覆盖3072维。本次 CLI 的 `memrl-episode` 使用4096维，在真实 Qdrant 中保存2条记忆，两次冻结检索各命中1条；配置见 `test/cli_e2e/memrl.episode.json`。

### 10. 批次更新顺序和持久化边界

- 上游行为：HLE 将整批 `add_memories` 后 `update_values`；ALFWorld、BCB、LLB 将整批 `update_values` 后 `add_memories`。batch 内 episode 运行时，尚未完成的批次不会先写入记忆库。
- MemAtlas 行为：adapter 按 profile 保留该执行顺序；自动按 `batch_size` 分组，或通过 `begin_batch/end_batch` 接收明确边界。`commit()` 只持久化原生 state 引用和待学习队列，setup/resume 可恢复；`close()` 不隐式学习尾批。自动模式不足一批时也可直接调用 `end_batch`。
- 原因：框架在每个 episode 反馈后调用 commit，不能把 commit 无条件解释为原生 batch 结束；需要在 adapter 中分别表示持久化与上游学习边界。
- 行为影响：普通逐 episode 的框架使用不会提前污染同批检索。最后不足一批需要调用方显式 `end_batch`；遗留的队列可以恢复，但不会因为 checkpoint 或关闭而额外执行学习。
- 测试证据：`test_profile_specific_native_update_order`、`test_bcb_inplace_preserves_native_order_observable_q_values` 对比真实 source 顺序；`test_explicit_batch_preserves_boundaries_and_separates_scopes`、`test_pending_batch_survives_snapshot_without_implicit_learning` 及自动尾批测试检查边界和恢复。

### 11. 输入配置与依赖声明

- 上游行为：`mos_config_path` 指向已有 MemOS JSON 输入；`MemoryOS==1.0.0` 要求 OpenAI SDK `>=1.77,<2`、tenacity `>=9.1.2,<10`，其 sentence chunker 和 Qdrant backend 需要可选包 chonkie 与 qdrant-client。
- MemAtlas 行为：从 `storage.path_parameters` 移除输入 `mos_config_path`，保留原路径；manifest 显式列出 `chonkie>=1.0.7,<2`、`qdrant-client>=1.14.2,<2` 及与 MemoryOS 一致的 SDK/重试库边界，README 提供独立环境安装命令。
- 原因：框架将 storage path 当作输出隔离到新 run 目录，不会复制输入文件；MemoryOS 基础包不保证安装全部可选 backend。
- 行为影响：自定义输入配置不会被重定位为不存在的文件；本次没有安装、卸载或修改共享环境依赖。现有共享 venv 的 litellm/OpenAI 冲突仍存在，不能据读写成功宣称该共享环境全量兼容。
- 测试证据：fresh workspace 配置路径回归检查、manifest 依赖版本检查；`test/memrl/README.md` 安装命令从该 manifest 生成 requirements，避免另维护一套未声明依赖。

### 12. 原生源码交付与预检

- 上游行为：算法源码属于独立 Git 仓库，memory service 依赖 providers、public config 和 utils；MIT license 允许随接入层再分发。
- MemAtlas 行为：manifest 声明 `source_subdir=methods/memrl/_vendor`；随插件保存完整 memory import closure 的 19 个 Python 文件、MIT LICENSE 和 `UPSTREAM.json`。所有原生 build/retrieve/update 策略均保留，benchmark runner、环境和数据不进入 memory 插件。helper 校验固定 revision、文件哈希、必须文件及必要 patch；显式 `runtime.source_roots.memrl` 仍可覆盖来源。
- 原因：`plan/` 是可删除的参考资料，插件必须在其缺席时正常运行；仅指向被忽略的外部 checkout 不能构成可复现交付。
- 行为影响：默认运行与测试不依赖 `plan/`、网络源码获取或 nested `.git`；除已声明维度 patch 及空 package marker 文件的结尾换行外，没有改写原生执行代码。缺失、哈希不符或已知错误 revision 不通过预检；无 Git/清单的外部 archive 可使用但标记 `source_revision_verified=false`。预检不导入原生 provider、不初始化数据库、不清理外部 dirty 源码。
- 测试证据：bundled inventory 校验返回 `source_provenance=vendored/source_file_hashes_verified=true/source_revision_verified=true`；source preflight 回归覆盖缺失目录、必须文件和维度 patch，离线 suite 使用 bundled source。helper 对临时副本的文件改动、缺失与错误 revision 均返回 unavailable。

### 13. 原生空库恢复的错误日志

- 上游行为：MemOS `GeneralTextMemory.load` 把空记录数组交给 Qdrant upsert，触发 `Empty update request`；异常被原生 loader 捕获并记录，没有向调用方抛出。
- MemAtlas 行为：保持原生 loader 的调用和错误处理；默认 HLE 的两条训练数据尚在 pending，空库的冻结恢复实际出现该 ERROR，但框架操作仍报告成功。
- 原因：这是保留的 MemOS/Qdrant 空库加载行为，没有因日志报错修改原生恢复路径或学习批次。
- 行为影响：CLI 退出0和操作失败数0不代表原生日志没有 ERROR；本次恢复前后库均为空，没有证据表明发生数据丢失。
- 测试证据：`.mematlas/runs/cli-e2e-20261024/memrl-default/cli.log` 两次 `Empty update request`；该运行最终原生记录0、pending batch2，汇总脚本同时统计原生日志信号与框架操作状态。

### 14. 原生快照中的 provider 配置

- 上游行为：MemOS GeneralMemCube 将运行时配置保存到 `cube/config.json`，其中 extractor/embedder 配置包含已经展开的 provider API key。
- MemAtlas 行为：输入 JSON 只指定凭据环境变量名，但原生保存行为仍将实际凭据写入本地快照。本次未改写快照内容；完整产物根目录权限已设为700，复现 launcher 采用 `umask 077`。
- 原因：环境变量输入不会自动使原生 serializer 脱敏；保持原生快照行为的同时，需如实说明产物内容与文件访问范围。
- 行为影响：日志脱敏不覆盖原生文件持久化；整个产物目录不能直接作为无凭据报告分享。目录访问限制不改变算法、存储格式或恢复语义。
- 测试证据：本次7份 `cube/config.json` 及2份备份的原生 cube 配置中，extractor/embedder API key 字段经只读检查确认含展开值，检查未输出值；`stat` 确认 `.mematlas/runs/cli-e2e-20261024/` 权限700，`test/cli_e2e/run.sh` 显式设置 `umask 077`。

## 框架 registry 说明

- 上游行为：不适用，这是 MemAtlas registry 问题。
- MemAtlas 行为：`PluginRegistry.load_builtins()` 会导入 `BUILTIN_PLUGIN_MODULES` 中的每个模块，不会把缺失模块当作不可用。目前 MemRL、MemQ、AEL、MemCon 都已实现并注册。
- 原因：registry 采用 fail-fast 行为，因此 registry 声明的每个 builtin 都必须可以导入。
- 行为影响：四个 adapter 都必须保持可导入，`default_registry()` 才能正常使用；核心 registry 协议没有改变。
- 测试证据：四个 method 自测都检查了 builtin registry discovery，并在当前 worktree 中通过。

## MemCon 接入差异

本节对应原生源码 `aa15afc9406aab8854b957ebfea08816997de882`，完整 MemCon/G-Memory 执行路径随插件交付在 `methods/memcon/_vendor`。原生自测入口为 `test/memcon/check_memcon.py`，依赖与运行命令见 `test/memcon/README.md`。

### 1. Task 生命周期映射

- 上游行为：runner 对每个新任务调用 `init_task_context(task_main, task_description)`，在同一任务的多步检索间保留 MDP、动作和当前轨迹。
- MemAtlas 行为：adapter 按显式 `episode_id` 区分新任务；没有 ID 时，以任务文本变化识别新任务。同一 episode 的多步检索继续使用原生上下文；新 episode 清理旧 pending action/seen event，并调用原生初始化。只读检索结束时连同初始化状态一起恢复。
- 原因：框架 method 生命周期没有直接传入原生 runner 的 task-start 回调，仅凭 scope 是否 active 无法区分连续评估中的不同任务。
- 行为影响：新任务不再继承前一任务的 context、step 或 pending action；被放弃的任务不生成虚构反馈。没有 ID 且任务文本相同的独立 episode 仍需调用方明确提供 episode identity。
- 测试证据：`test_episode_start_identity_resets_same_task_and_survives_snapshot` 检查不同 ID 重新初始化、同 ID 保留多步状态；`test_runtime_readonly_cross_task_queries_do_not_reuse_previous_goal_or_plan` 检查只读跨任务隔离。

### 2. Action 与 observation 事件映射

- 上游行为：HMF runner 在每次环境 action/observation 配对后调用一次 `move_memory_state(action, observation, reward, done)`；只思考不执行的 action 不会发送给 MemCon。
- MemAtlas 行为：adapter 暂存 `ACTION`，与后续 `OBSERVATION` 或 `TOOL_RESULT` 配对；已有 observation metadata 的 action 立即转发。忽略 think/thought action，在存在时保留 reward/done。连续未配对 action 及学习前最后一个未配对 action 以空 observation 转发。
- 原因：MemAtlas 把 action 和工具/环境结果作为独立事件发出，原生 API 接收的是成对调用。
- 行为影响：保留已有配对的原始内容与顺序；空 observation 仅表示框架没有提供对应结果，不补造环境信号。
- 测试证据：`test/memcon/check_memcon.py` 检查事件配对、reward/done metadata、重复事件过滤和未配对 action 收尾。

### 3. Episode 反馈与更新顺序

- 上游行为：任务结束时先 `save_task_context(label, feedback)`，再 `backward(label)`。wrapper 计算 success/failure 与效率 reward，按反向折扣更新 policy Q，再更新检索过的 insight。该 revision 的 wrapper 先调用 inner.save，再经 super.save/self.add 再次委托 inner.add，因此每个已学习 episode 原生写入两份 G-Memory 记录。
- MemAtlas 行为：`learn()` 要求外部布尔 `EpisodeOutcome.success` 和任务文本，先补齐 episode 事件，再依序调用这两个原生方法；不以框架数值 reward 替代原生 reward 公式。
- 原因：框架 learning 契约是 `MemoryEpisode/EpisodeOutcome`，原生 runner 使用独立 lifecycle 调用。
- 行为影响：反馈来源、Q 公式、两次 G-Memory 写入、success plan 和 insight 更新顺序均由上游执行，没有去重或修复原生双写；无 success 布尔值时明确拒绝学习。原生 save 非事务操作，异常可能留下部分 policy/plan 变化，不新增自动回滚策略。
- 测试证据：`test_native_add_retrieve_policy_update_insight_feedback_and_log_namespaces` 验证原生双写和 save/backward 顺序；`test_native_policy_reverse_discounted_credit_assignment` 对照折扣 Q 公式，`test_runtime_failed_native_save_can_be_explicitly_restored_and_retried` 验证显式恢复后重试。

### 4. 单次结果数量限制

- 上游行为：controller 自行选择 `top_k/insight_k/hop`，执行原生 RETRIEVE、PLAN_INJECT、RE_RETRIEVE、CONSOLIDATE、FORGET 或 NO_OP 分支。
- MemAtlas 行为：原样调用 wrapper；如果设置 `DecisionContext.top_k`，仅在合并 success/failure/insight 后截断 contribution。
- 原因：框架只有一个单次结果上限，MemCon 有不同检索通道及 policy 参数。
- 行为影响：较小的框架上限会省略靠后结果，不改变原生 action 选择、检索参数、排名、plan injection 或维护操作。
- 测试证据：`test/memcon/check_memcon.py` 检查返回截断不改写 policy 选择的原生参数，并覆盖原生维护 action 的只读恢复。

### 5. Scope 与原生持久化目录

- 上游行为：MAS backend 使用一个 `global_config.working_dir` 和 namespace，不表示框架组合 scope。
- MemAtlas 行为：每个完整 `MemoryScope` 创建独立 service，存储在 `<storage_path>/scopes/<scope-key>/`；未指定路径时使用 method artifact 目录。内层 Chroma/graph/insight 与外层 policy/plan 文件仍按原生 namespace 布局。
- 原因：一个框架实例可服务多个 user/agent/team/session/run scope，不能把无关任务写入同一个原生库。
- 行为影响：各 scope 数据隔离，物理路径包含 adapter 派生的 scope key；普通持久化复用需要稳定的 storage/artifact 目录。
- 测试证据：离线 suite 检查 scope 隔离和原生持久化；空快照恢复检查在快照之后创建的 scope 会恢复到 service 创建时 baseline 并移出 adapter。

### 6. Provider 与 package import 适配

- 上游行为：同步 `GPTChat` 在导入时读取相对当前目录的 `configs/configs.yaml` 并打印 key；GPT-5/o3/o4 使用 `max_completion_tokens`，其他模型使用 `max_tokens`。原始 package initializer 还会提前导入无关 HMF agent/framework 及 GMemory++/SkillRL。G-Memory 默认使用本地 CPU SentenceTransformer。
- MemAtlas 行为：以同步 wrapper 调用异步 `MethodServices.llm`，由 adapter 提供 source 形状的 `Message/LLMCallable`；不导入或交付未使用的 native provider/credential config。保留原始 initializer 文件，在 loader 建立对应 package namespace 来导入本插件所需 memory 模块。
- 原因：异步 provider 接口和从任意当前目录运行的框架不适配 source transport；引入无关 agent runners 不是 memory 算法接入的必要依赖。
- 行为影响：HTTP transport、凭据与计量由框架提供；原生 prompt、G-Memory/Chroma、策略、token-limit 模型分支和默认 `sentence-transformers/all-MiniLM-L6-v2` 不变。不会修改当前目录，也不会导入打印 key 的 provider。
- 测试证据：离线 suite 对照普通模型/GPT-5 token 字段、source 默认值及原始 prompt；本轮 native 文件逐字节核验固定 revision，只有已声明 package marker 结尾换行差异。

### 7. 完整逻辑快照与只读检索

- 上游行为：原生没有统一 snapshot API；retrieval 会修改 UCB/MDP/action/cache，并可能执行 CONSOLIDATE/FORGET。记忆同时存在于 Python 对象、Chroma、graph pickle、insight/policy/plan JSON 中。
- MemAtlas 行为：新增 adapter 层 `save_state/load_state`，将 policy/MDP/task context 全部 StateChain graph、Chroma records/向量、G-Memory graph/insight/cache、adapter event 状态，以及原生持久文件字节和存在性内联到 JSON。通过原生 Chroma collection API 差量恢复，不复制活跃 SQLite、不重算 embedding；未变化记录保持不写回。声明 `read_only_mode=snapshot`，直接传 `metadata.read_only=true` 的检索也自行保存/恢复。
- 原因：仅冻结 learn 不足以隔离 MemCon 检索期间的学习计数、上下文和维护操作；框架已有 save/load 协议，兼容实现只能放在接入层。
- 行为影响：普通训练检索保留原生副作用；只读调用在成功或异常后恢复逻辑算法状态，快照无外部目录引用。全局 Python/NumPy 随机源、UUID、物理 Chroma 历史和文件 mtime 不回滚，不承诺 save/learn/load 后逐字随机重演；新 scope 回滚后目录和诊断日志可留存，records/learned files 回到 baseline。原生无 per-instance close API，adapter close 只释放引用。
- 测试证据：`test_snapshot_preserves_active_agent_graphs_and_detaches_mutable_state` 及同脚本的连续 save/load、原生文件存在性测试检查完整逻辑 state；`test_runtime_readonly_forget_and_consolidate_restore_policy_mdp_insights_and_files` 验证维护动作恢复；new-scope 和 retrieval-failure 两项测试分别检查 baseline 与异常恢复。

### 8. 原生源码交付与来源预检

- 上游行为：固定 revision 的 memory 源码分布在 `mas/` 和 `hmf/`；上游根 README 声明 MIT，但根目录及这两个子目录没有 standalone license 文件。
- MemAtlas 行为：`methods/memcon/_vendor` 保存 16 个必要 native Python 文件、NOTICE 和 `UPSTREAM.json` revision/SHA-256 清单。保留全部 MemCon/G-Memory 算法与 prompt；排除其他方法、benchmark、agent 框架和 provider 凭据配置，不复制其子项目 license。helper 校验实际来源、必须文件、固定 revision 和清单哈希。
- 原因：可删除的 `plan/` 不能成为运行依赖；来源与缺失的 license 文件信息必须可审核，不能用其他项目 license 代替。
- 行为影响：默认运行和测试无需 `plan/`、网络源码获取或 nested Git；唯一 packaging 差异是 `mas/__init__.py` 的最终换行。显式 source override 仍支持；无 Git/清单的 archive 标记 revision-unverified，缺文件/哈希不符/已知错误 revision 不通过。
- 测试证据：16 个 Python 文件对照固定 revision 的 Git blob，通过 byte comparison（允许已声明最终换行）；helper 返回 `source_provenance=vendored/source_file_hashes_verified=true/source_revision_verified=true`。`test_vendored_source_inventory_detects_tampering_missing_files_and_wrong_revision` 检查预检拒绝错误源码，`test_complete_native_suite_runs_with_historical_source_inaccessible` 在禁止读取 plan 的子进程完整重跑原生 suite。

### 9. 依赖约束声明

- 上游行为：`requirements.txt` 给出 memory 包版本；langchain-chroma 0.2.3 metadata 声明 langchain-core/chromadb 边界。上游同时声明 finch-clust 0.2.0 与 finchpy 0.0.1，两包提供不同内容的同名 `finch` 模块，未定义覆盖优先级。
- MemAtlas 行为：保留两包和 source pin；将 `python_constraints` 包名与 `python` 列表统一，将“required by …”替换为包 metadata 的有效版本约束。README 从 manifest 生成独立安装命令，并提供可选 `pip install --no-deps --force-reinstall finch-clust==0.2.0`，明确选择本轮执行的 FINCH 实现，不由 adapter 自动执行。
- 原因：原约束键名不一致且部分值不是版本说明符；同名模块覆盖关系还会影响 fresh install 的实际实现，必须显式说明。
- 行为影响：本轮未安装或改写共享依赖，没有把 finch-clust 的优先级声称为上游保证。独立环境可显式选择本轮测试实现；当前回归不等于已验证一套完全满足全部官方 pin 的全新环境。
- 测试证据：所有 manifest 说明符经 `packaging` 解析；已安装 `finch` 两文件 SHA 均匹配 finch-clust RECORD、不匹配 finchpy RECORD，README 提供只读验证命令。当前回归使用 Chroma 0.6.3、NetworkX 3.6.1、NumPy 2.4.6、SentenceTransformer 3.4.1。

### 10. 原生 Chroma telemetry 兼容日志

- 上游行为：MemCon 使用原生 G-Memory/Chroma 存储；Chroma 的 telemetry 调用在本机现有依赖组合下记录 `capture() takes 1 positional argument but 3 were given`。
- MemAtlas 行为：保留相同数据库和依赖环境，没有为消除日志更换 backend、关闭原生功能或隐式安装其他版本。
- 原因：此次 CLI 检查区分非阻断的原生依赖错误日志与框架操作失败，不把静默屏蔽日志当作算法接入修复。
- 行为影响：本次实际写入和读取均成功，但不能宣称现有共享环境完全兼容；依赖约束和独立安装命令仍见 manifest/方法 README。
- 测试证据：`.mematlas/runs/cli-e2e-20261024/memcon-default/cli.log` 保存 telemetry ERROR；只读查询真实 Chroma SQLite 得到4条记录，两题各命中2条，框架操作失败数0。provider 重试另列于【框架待优化项】，不据 telemetry 日志推断其原因。

## MemCon 参数变体

- 上游行为：默认 policy 为 learning_rate 0.15、UCB c 1.4、discount 0.9、warm_start true；G-Memory 默认 hop 1、insight threshold/interval/point count 为 5/5/5。原始 `scripts/sweep_hyperparams.py` 声明参数搜索范围，没有交付已调优 `best_memcon_policy.json`。
- MemAtlas 行为：`profile=default` 保持这些官方默认，`config_schema.x-parameter-ranges` 保存 source sweep 值，保留可配置标量及原默认本地 embedding model。
- 原因：框架不能把 sweep 候选当作已验证的最佳组合，也不能自行优化策略。
- 行为影响：只有显式传参才改变 source 默认；不虚构额外 tuned profile，搜索范围不自动触发调参或额外学习。
- 测试证据：默认值与范围已对照固定 revision 的 policy/wrapper/G-Memory 和 sweep source；离线 suite 断言默认配置，完整算法仍调用 bundled native 实现。

## MemQ 接入差异

本节对应 `SII-MemQ/MemQ@9d66abf4f0252b084f088d5081c44b840842e47b`。完整 native memory import closure 已交付于 `methods/memq/_vendor`，运行和测试不依赖 `plan/`。原生自测入口为 `test/memq/check_memq.py`，安装与运行命令见同目录 README。

### 1. Episode 协议、反馈与检索 credit

- 上游行为：runner 提交一批 task/benchmark 轨迹、外部布尔 success、已使用记忆 ID、query/score 和配置的 success/failure reward；记录 child→parent 引用供直接 Q 更新和 Bellman pass。
- MemAtlas 行为：`learn()` 要求任务文本与布尔 `EpisodeOutcome.success`，序列化框架事件后按 scope 排队；只从显式 `MemoryEpisode.retrieved` 转发实际确认的记忆和 query/score，不回退到上次候选。image URI 按 source 的 `{'url': ...}` 形状传入。
- 原因：框架提供 `MemoryEpisode/EpisodeOutcome`，并区分候选检索与实际使用记录，原 runner 使用独立 task/trajectory 参数。
- 行为影响：保留外部反馈及 source reward，不使用 `EpisodeOutcome.reward` 替代。未确认候选不获得 credit；轨迹采用框架事件表示，不复制 benchmark 环境/agent 封装。
- 测试证据：`test_native_add_retrieve_two_stage_q_update_and_log_namespaces`、`test_explicit_native_batch_preserves_update_add_and_boundary_order` 检查真实 add/retrieve/update 及 success/failure；`test_empty_confirmed_retrieval_does_not_credit_recent_query_candidates` 检查候选不回退。image URI 转换经 adapter/source 接口核对，本轮离线测试未验证图片模型行为。

### 2. 完整 batch/epoch 更新链

- 上游行为：官方 LCB 在每个完整 batch 的 add 前调用 `update_values` 与 `_update_q_max`，同时在 epoch 末执行 Bellman TD、flush、reset 与 Q-max 更新。LLB 的 direct-value update 延迟到 epoch 末；其他 profile 按各自 runner 执行。
- MemAtlas 行为：`learn` 等待完整 `batch_size` 或 `end_batch`，按 profile 依序执行直接更新、整批 add 和相应 MemQ pass；epoch 更新由 `end_epoch` 触发。保留 `begin_epoch/set_epoch_info` 的零基 epoch/total_epochs，供原生 alpha schedule 使用。`commit` 只持久化已学数据和 pending 队列，close 不隐式学习。
- 原因：框架逐 episode commit 不等于原生 batch/epoch 结束，不能在其中无条件执行 Bellman；接入层需要表达直接更新与结构传播的各自边界和 runner 生命周期。
- 行为影响：恢复 source 完整更新链，没有统一或优化原公式。默认 LCB 同 scope 50 个 episode 才完成一批，尾批须显式结束，epoch pass 须显式 `end_epoch`。持久化和学习边界分离。
- 测试证据：`test_explicit_native_batch_preserves_update_add_and_boundary_order`、`test_cosine_epoch_schedule_and_native_single_step_formula` 对照真实 Q 更新及调用顺序；`test_default_commit_keeps_pending_learning_until_explicit_epoch_boundary` 检查 commit 不学习，`test_epoch_direct_value_stage_is_deferred_until_all_children_exist` 检查 deferred feedback 跨快照还原及两个更新阶段的实际 Q 值。

### 3. Profile runner 的实际例外

- 上游行为：LCB/LLB/ERQA/MMMU 的 batch pass 为 flush→Q-max→reset，epoch 为 flush→reset→Q-max。DSBench batch 为 flush→reset→Q-max；GPQA batch 为 flush→reset、Q-max 在 epoch 末刷新，且 runner 未调用 epoch setter，direct update guard 的 `"memrl" in self.algorithm` 不覆盖 memq。BFCL 不按 memq_update_mode 分支而总在 epoch 末传播；其 direct epoch guard 也不覆盖 algorithm=memq。
- MemAtlas 行为：按 profile 保留这些实际顺序与 guard；不把 BFCL 的 batch 配置强行改成其他 runner 的行为。提供显式 epoch setter 命令表达调用方指定的原生 epoch 状态。
- 原因：必须对齐可执行源码，不能为了配置名一致而修改遗漏分支或策略。
- 行为影响：官方所有 template 默认 epoch MemQ 模式不受 BFCL 参数忽略的额外影响；显式切换 batch 时仍保留 source 例外。GPQA 使用显式 epoch 命令时会设置 source runner 没有更新的 epoch 字段；默认 constant alpha 数值不变，非默认 cosine schedule 需理解该差异。
- 测试证据：固定 revision 的七个 runner update/reset/Q-max 分支逐一核对；`test_official_profile_runner_exceptions_and_adaptive_child_q_are_preserved` 运行 LLB/GPQA/BFCL/DSBench，断言顺序和实际 Q 值。

### 4. Provenance 容量与顺序

- 上游行为：LCB/LLB/BFCL/GPQA/ERQA/MMMU runner 以 OrderedDict 保存 child→parents，最多 10000 个 key，复用 key 时 extend 并 move_to_end。DSBench 用 setdefault/extend，无 FIFO 容量与移动。service 接收这张 runner DAG 并保留自己的 eligibility trace。
- MemAtlas 行为：每个 scope 维护对应 profile 的有序 DAG，在原生 add 返回 child ID 后加入，保留重复引用、插入顺序与原生容量规则；快照连同 DAG 与 trace 一起捕获。
- 原因：DAG 原来属于 benchmark runner，不是 MemoryService 的独立持久化表；接入层必须补齐 runner 数据。
- 行为影响：恢复后结构传播仍使用原顺序、重复引用与淘汰语义；不因扫描数据库而生成不同 DAG，不去重原生引用。
- 测试证据：`test_native_bellman_all_path_lambda_credit_and_contribution_normalization` 检查 parent/child/ancestor 的重复路径传播；`test_provenance_fifo_limit_and_dsbench_unbounded_variant` 检查 10000-key 边界和 DSBench 不淘汰；inline snapshot 测试检查 DAG 往返。

### 5. 检索上限、scope 和向量维度

- 上游行为：`k_retrieve` 控制候选 query key，source `topk` 控制 value-aware 选中记忆；每个 service 管理一个 user bank。launcher 实际探测 embedder 输出维度并传给 NumPy store，默认 text-embedding-3-large 输出 3072。
- MemAtlas 行为：分别保留两个 top-k；框架 `DecisionContext.top_k` 仅截断选中结果。完整 `MemoryScope` 映射独立 native service 与稳定 user_id；默认 vector_dim/embedding_dims 为 3072，可显式配置。
- 原因：框架单次数量上限、组合 scope 与异步 provider 设置不能直接等同上游候选数、单 user bank 或 launcher 的 I/O 探测。
- 行为影响：不重检索或重排；各 scope 不共享记忆。替换 embedding model 时需保持配置维度与实际向量一致，默认不增加一次远程 dimension probe。
- 测试证据：`test_official_defaults_and_native_numpy_store_initialization`、`test_q_max_normalization_preserves_native_weighted_ranking` 运行小维真实 NumPy store/search；readonly/snapshot 测试检查新 scope 不保留回滚后的记忆。两个 top-k 的映射另由 source/adapter 分支核对，本轮没有专门的 top-k 参数矩阵测试。

### 6. Native provider 的 transport 边界

- 上游行为：同步 OpenAILLM.generate 与 OpenAIEmbedder.embed 包含 source prompt、keyword/script、retry、circuit breaker、token truncation 及 chunk/merge；原 SDK response 还带 usage 和 finish_reason。LCB/DSBench/ERQA/MMMU launcher 使用 streaming 响应。
- MemAtlas 行为：直接继承这些 native 方法，用同步 client facade 转发至异步 MethodServices，再转换 text/vector 为 native 所需 response shape。选择 native non-streaming 分支，使用官方 breaker 参数，每个 scope service 建立原生 breaker；不构造虚假的 usage 或结束原因。
- 原因：框架 provider 只返回最终 text/vectors，没有 SDK stream 与完整 response metadata；只能在 transport 接口层转换。
- 行为影响：prompt、重试、熔断和 embedding 聚合路径保留；HTTP transport/凭据/真实 usage 由框架负责。native usage/finish_reason 未知、token counters 保持零，无法根据 SDK finish_reason 识别截断；不能拿它们代替框架 ledger。上游跨所有 agent/memory provider 的共享 breaker 在本 memory plugin 中以 scope service 为边界。
- 测试证据：`test_native_provider_methods_preserve_script_keyword_chunking_and_logs` 断言原生方法身份、script/keyword prompt、chunk/merge 与日志；provider-error readonly 测试运行五次原生 retry；`test_runtime_provider_defaults_metering_and_public_overrides` 检查框架 usage。breaker 代码/参数已核对，本轮未等待真实熔断暂停或验证在线 streaming。

### 7. 完整快照、恢复和只读检索

- 上游行为：native checkpoint 保存向量、payload 和 query cache，service loader 会筛选 cache 并可能重新生成 embedding；它没有完整保存 Q-max、eligibility/accumulation、epoch 和 runner pending 状态。检索会改 retrieval_count/last_used_at/cache。
- MemAtlas 行为：schema-2 state 内联 native MemoryStore.save 产生的 vectors.npy/payloads.json 字节，另保存 Q-max、trace、epoch、有序 query/cache、DAG、pending 和最近检索等 sidecar，以及 native epsilon-greedy 所用的进程全局 Python random state；load 使用 native MemoryStore.load 再恢复 sidecar，避免 service loader 额外转换。read_only_mode=snapshot，保留原生检索后恢复逻辑 state。commit 保存 `<storage_path>/adapter_state.json` 和 native content-hash checkpoint；未设置 storage_path 时用 `<artifact_dir>/memq`。
- 原因：框架冻结评估与可恢复训练要求保存全部算法状态；native checkpoint 外壳不足以恢复内存中的 runner/学习累积字段。
- 行为影响：新 state 不依赖外部 snapshot 路径；typed sidecar 保留 mapping 顺序、NumPy/date metadata，避免原生 JSON 的 default=str 改变值类型，不支持的任意对象明确抛 TypeError。不完整的旧 schema-1 adapter state 被明确拒绝，没有迁移支持。load/只读恢复会回拨同进程其他使用 random 的组件，scope 间不承诺独立随机上下文。NumPy RNG、provider/breaker 状态、usage/日志、UUID/时间与物理文件历史不回滚，不承诺学习后的逐字随机重演；不改变 source 搜索、排序或 Q 更新公式。
- 测试证据：`test_inline_snapshot_covers_qmax_retrieval_traces_pending_scopes_and_is_immutable` 检查 JSON 往返、向量/payload、trace/epoch、pending/DAG、跨 adapter 恢复、Python RNG 回拨、schema-1 拒绝与不可变性；`test_committed_pending_batch_and_edges_survive_restart_and_reset_does_not_resurrect` 检查磁盘重启与 reset 不复活；两个 runtime readonly 测试分别检查正常与五次 retry 失败后的逻辑恢复。

### 8. 原生源码与依赖交付

- 上游行为：memory 执行闭包为 package/version、service、providers、public config 和 utils；MIT LICENSE 允许原样再分发。核心依赖声明在 pyproject.toml，tqdm/tiktoken 另在 requirements.txt；token truncation 仅在配置启用时使用 tiktoken。
- MemAtlas 行为：vendor 21 个原生 Python 文件、MIT LICENSE 与 revision/22 文件 SHA-256 清单；未复制 benchmark/agent/数据或 credential config。所有 build/retrieve/update/provider 模块保留原文；service/keyer.py、service/strategies.py、service/__init__.py 与两个原为空的 configs/__init__.py、utils/__init__.py 仅补一个最终换行。manifest 保留 core package 依赖并显式声明 tqdm>=4.60.0/tiktoken>=0.5.0。
- 原因：`plan/` 可删除，不能成为运行依赖；启用上游 token truncation 时不应依赖环境偶然已有 tokenizer。
- 行为影响：默认运行和测试无需网络源码获取或 nested Git；没有安装或改动共享环境。source preflight 检查文件、hash 和 fixed revision；显式 override archive 无 Git/清单时标记 unverified。tiktoken 编码数据需本地 cache；不为可选 native LocalEmbedder 安装此接入路径不用的模型栈。
- 测试证据：21 个 Python 文件与 LICENSE 对照固定 Git blob 的 byte comparison 通过（仅已声明最终换行），helper 返回 source_provenance=vendored/hash/revision 全 true；manifest 约束可解析，no-plan suite 使用同一 bundled source。

### 9. 恢复 launcher 的 public RLConfig

- 上游行为：launcher 将 public Pydantic RLConfig 传给 service；内部 value_driven.RLConfig dataclass 缺少 q_min_threshold，但 service 用 getattr 读取。
- MemAtlas 行为：直接构造与 launcher 相同的 public Pydantic RLConfig；q_min_threshold 按官方声明为 number，q_floor/q_ceiling/max_tokens 保留合法 nullable 声明。
- 原因：public config 已提供所需字段，使用上游现有接口即可完整对齐。
- 行为影响：有效参数及过滤行为由官方 public config 校验，不再动态扩展 dataclass。合法 null 仍通过插件局部适配绕过框架 nullable 缺陷；官方不接受的 q_min_threshold=None 不再宣称支持。
- 测试证据：`test_official_defaults_and_native_numpy_store_initialization` 检查 public RLConfig 类；`test_runtime_provider_defaults_metering_and_public_overrides` 检查 q_floor/q_ceiling/llm_max_tokens 的合法 null；q_min_threshold 的非 nullable 声明与 public config 类型另经源码核对。

## MemQ 参数变体

- 上游行为：LCB/BFCL/GPQA/ERQA/MMMU 为五 epoch，batch 分别 50/100/100/50/50；LLB 为 64×20 且 direct epoch update，DSBench 为 8×10、memory LLM limit8192（agent16384）。各 launcher 用 batch_size 作为 native num_workers。LLB 使用 trajectory/vanilla、epsilon0、weights1/0；BFCL weights.5/.5；GPQA gamma.35；DSBench k5、normalize_dq/adaptive init。
- MemAtlas 行为：default/lcb 选择官方有效 LCB，variants 完整声明上述 memory 参数；LLM breaker 默认保留官方 enabled、threshold3、failure_window30、probe300/confirm30、successes2、max_pause3600、retry_delays1/2/5/10。
- 原因：默认需选择一套主实验配置；必须区分 agent 参数与真正传给 memory provider/service 的有效值，不能将 service 默认 workers32 误当 launcher 实验值。
- 行为影响：仅显式 profile/参数改变配置；不同 batch 大小改变批内可见性和更新边界，属于官方 profile 行为。ERQA/MMMU 的 max_memory_images10 是 benchmark prompt 注入上限，不是 MemoryService 参数；memory 插件按原生接口处理图片，caller 决定 agent prompt 装配。
- 测试证据：固定 revision 七个 template 与顶层 run/run_*.py launcher 已逐一核对，包括 DSBench memory_max_tokens 和 num_workers；default 初始化与 profile 例外测试检查有效配置及原生更新结果。

## AEL 接入差异

本节对应 `WujiangXu/AEL@c95f5f961667e7df4bdddf750b22332158951bf2`。运行与原生自测使用 `methods/ael/_vendor`，不依赖 `plan/`。主实验入口经 README、run_paper_5seed.sh 和 run_all.paper_all 交叉核对，默认采用 tool_memory_reflect 加 D-full 数据参数的 portfolio 路径。原生自测入口为 `test/ael/check_ael.py`，安装与命令见同目录 README。

### 1. Episode 记录与反馈来源

- 上游行为：portfolio runner 把外部连续 outcome_score、tool evaluation、prediction/actual 交给 `add_episode_experience`；原生将 outcome 映射为 `max(0.3,min(1,(score+1)/2))` 的写入 quality。
- MemAtlas 行为：要求有限数值 outcome_score 或 EpisodeOutcome.reward，不从 boolean success 推导分数。rich metadata 使用原生 add_episode_experience；通用框架事件缺少金融结构时通过原生 add_episodic 写轨迹文本，quality 保留同一公式。
- 原因：MemoryEpisode 表示通用事件，不必包含原生金融 task、tool/prediction/actual 对象，且协议没有统一连续奖励生成器。
- 行为影响：保留原生阈值、recency counter 和 metadata relevance；泛化轨迹的文本/字段来源不同于金融 runner。没有伪造 ground truth 或根据成功布尔值生成 credit。
- 测试证据：`test_native_add_retrieve_update_and_adapter_native_log_namespaces` 运行真实 local store；`test_numeric_outcome_and_trajectory_are_required_without_boolean_reward_inference` 检查非法反馈/输入被拒绝。

### 2. 原生检索策略与 portfolio 注入边界

- 上游行为：local MemoryStore 按 metadata、quality/usefulness、recency、tier/regime 评分；hybrid 只是配置标签，不调用 embedding。strategy 先检索再格式化，sliding_window/truncate 后被舍弃的 hit 已计 retrieval_count。portfolio 空策略命中会回退 semantic/procedural top5，再仅将 score>0.2 的前三条放进 prompt；episode 路径不做该回退/三条限制。
- MemAtlas 行为：按 runner_mode 保留 native strategy、portfolio fallback/filter/cap 和异常吞掉语义；框架 top_k 仅截断最终返回。默认 query 按对应 runner 构造，memory_query mapping 可显式透传。
- 原因：框架贡献结果应对应 source 实际注入的上下文，而非单纯暴露 MemoryStore 候选；同一个算法在两个上游 runner 中的后处理不同。
- 行为影响：不替换评分或重排；格式化前计数等原生副作用保留，top_k 不回退重新检索。portfolio 默认只用 task_type/market_regime，若要 ticker/sector 评分需显式 query 或 episode runner。
- 测试证据：`test_native_metadata_scoring_recency_tiers_quality_and_public_top_k` 与 `test_native_sliding_window_and_ranked_truncate_preserve_source_counting` 检查真实分值、格式化及检索计数；`test_runner_default_queries_portfolio_fallback_filter_and_native_exception_handling` 检查 fallback/filter、query 默认值与异常边界。

### 3. Controller credit、per-tool 更新与确认使用记录

- 上游行为：assign_credit 先对成功 tool call 做 per-tool 更新（correct/wrong 为0.8/0.2，无方向用 clip((score+1)/2,0.3,0.7)），再将 clip((score+1)/2,0.05,0.95) 和 module credits 传入 MetaController，最后遍历全部 trajectory.memory_retrievals 标记 usefulness。portfolio 记录的 retrievals 包括 prompt 筛选前的 hit。
- MemAtlas 行为：保留原生 per-tool→meta→usefulness 顺序和公式；默认 uniform credit 直接调用原生 uniform_credit。FCC/structural 等由外部提供 module_credits；只接受显式 controller context 或已确认 item 的 source context，显式 MemoryEpisode.retrieved 决定 credit，不回退最近候选。
- 原因：框架确认使用集合与候选检索是不同协议对象，且不执行原生 planner/tool/finance counterfactual，不能虚构模块归因或把候选默认当成已使用。
- 行为影响：布尔 success 不替代 credit，alpha_z/planner_alpha 的原生优先级保持。被 source prompt 筛掉、未返回或未确认的 hit 不在框架中自动获 usefulness credit；无 confirmed context 时不更新 controller。成功 tool call 必须有显式 success 来源。
- 测试证据：`test_native_linucb_thompson_credit_formulas_and_alpha_precedence` 检查真实 posterior/alpha；`test_per_tool_ground_truth_updates_precede_meta_and_memory_learning` 检查原生 dispatch 和顺序；`test_empty_confirmed_items_never_credit_recent_retrieval_candidates` 检查无候选 credit。

### 4. 两个 runner 的 history 与周期行为

- 上游行为：canonical run_episode 在 assign_credit 前 append trajectory，differential baseline 的最近20条含当前分数，至少5条时取均值；周期用 len(trajectories)>0 且十的倍数。论文 PortfolioRunner 没有训练 append，assign_credit 也只在 frozen 分支补 append，所以训练 episode_num 恒0，feedback 的十 episode 蒸馏不触发，differential baseline 为0。
- MemAtlas 行为：runner_mode=portfolio 保留该实际空 history；runner_mode=episode 在反馈前记录当前 outcome/confirmed memory history，并用原生条件执行 periodic distill→promote，再写当前 episodic。increment_episode 仍保留 native recency 用途，不拿它替代 trajectory 长度。
- 原因：store episode counter 不等于 source episode_num，baseline 是否包含当前分数也取决于 runner 的实际执行顺序；两个 runner 需要分别对齐。
- 行为影响：默认不修复 portfolio 缺失 append；其夜间 consolidation 仍可发生。canonical 模式保留 fifth/twentieth 边界及当前记录纳入 baseline 的行为，当前 episodic 写入在周期蒸馏之后。原生蒸馏后的 planner.update_prompt_from_memory 属于外层 planner，插件不调用它，因此少了该路径对 semantic/procedural top3 的额外检索计数，后续 usefulness-adjusted score 可能不同。
- 测试证据：`test_raw_sign_and_runner_specific_differential_fifth_and_twentieth_boundaries` 检查 baseline 实际值；`test_episode_distillation_and_promotion_order_and_portfolio_nontrigger_are_native` 运行周期边界并验证 call order、semantic/procedural 结果及 portfolio 不触发。

### 5. 夜间 memory 模块与外部日界

- 上游行为：do_nightly_evolution 先增加 day_index，bars_per_day>=4 每夜，否则每2夜执行 auto_distill(min8)→promote(min8)，随后遍历 planner.update_prompt_from_memory，并清 morning overrides。evolve_memory 且日>=10、每5日、所有 memory arms 均值<0.4 时，用 reflection 和最近 trajectories 调原生 evolve_memory_policy，合法新 policy 加 registry/bandit。
- MemAtlas 行为：end_day/nightly_memory command 表达上述 memory 调度并保存 day_index；直接执行 bundled native evolver 的完整 prompt/解析/验证，reflection 由 caller 提供。frozen/read_only/_warmup payload 跳过；commit/close 不隐式发日界。
- 原因：BaseMemoryMethod 没有交易日/夜间 callback，不能从通用 episode 推导市场 bars；native nightly 函数还包含框架未执行的 planner/code/skill 操作，接入仅搬运必要的 memory 调度。
- 行为影响：调用方需按 scope 明确发 end_day。D-full bars4 每夜蒸馏，standalone bars1 每2夜；portfolio 用原生空 history 进入 policy prompt。省略 planner prompt 更新还会省略其对 procedural/semantic top3 的原生 retrieve 副作用，retrieval_count 和后续 usefulness-adjusted score 可与完整 agent 不同。生成日反思、将昨日反思注入 prediction prompt、planner/code/skill 演化仍依赖外层 agent，不宣称复现完整 paper agent。
- 测试证据：`test_nightly_memory_uses_native_daily_hourly_cadence_and_skips_frozen_warmup` 检查日界与顺序；`test_tenth_night_low_reward_runs_native_policy_prompt_adds_arm_and_survives_snapshot` 实际调用原生 evolver，验证 prompt/options、新 arm 及快照恢复。未运行金融回测或在线反思模型质量实验。

### 6. Scope 与不支持的 XMem 事务语义

- 上游行为：local store/controller 是单 run 对象。XMemClient 的 export 失败可静默返回 counter-only state，import 非 replace，缺少 delete/reset；mark_useful/consolidation 也不完全受只读 flag 阻止。原生 factory 在健康检查失败时可回退 local。
- MemAtlas 行为：完整 MemoryScope 映射独立 local store/controller/有序 policies 与持久目录；backend=xmem 在预检/setup 明确拒绝，完整原生 XMemClient 仍打包保留，不提供假的可回滚 remote bank。
- 原因：框架 save/load、只读与 reset 要求恢复一个确定 bank，上游 REST API 无法提供该事务语义；自动换成 local 也会隐藏用户所选数据库的行为变化。
- 行为影响：官方 paper/local profile 不变；本次不交付框架托管的 XMem backend。没有修改 XMem REST 算法或实现新的远程 API。
- 测试证据：`test_scope_isolation_and_scoped_reset_keep_other_native_bank`、committed restart/reset 测试检查 local 隔离；XMem 的 export/import/flag 缺口经原生源码核对，未连远程 XMem server。

### 7. 完整逻辑 state、恢复与随机源

- 上游行为：native serializers 保存 entries 和 bandit 参数，但 local loader 合并旧记录，bandit loader 只更新已有 arm；LinUCB 漏 alpha_0/_total_pulls，controller 漏 registry/morning/fixed choices，且 native NumPy Thompson random、runner history/最近 retrieval 不在 checkpoint。
- MemAtlas 行为：schema-2 state 内联 native state 并保留上述 sidecar、typed order/NumPy/TaskFeatures/date 值、全部 scope、history/day index 和确认上下文；先重建精确 entries/arms，再用 native loaders 恢复。保存/恢复进程全局 NumPy RNG。setup 按原生 seed42 初始化后优先恢复 committed adapter_state.json；commit 另保留 native memory.json/controller sidecar。
- 原因：框架只读评估和训练恢复要求完整逻辑状态，仅原生 merge serializer 会丢动态 arm/衰减计数，或在回滚后残留新记录。额外 state 属于接入层，不改 Thompson/LinUCB 公式。
- 行为影响：state 无外部目录引用、返回映射无共享可变别名；不完整 schema1 明确拒绝，无迁移。load/只读恢复会回拨同进程其他 NumPy 用户，未提供 per-plugin RNG 隔离；Python random、UUID/时间、provider 日志和物理文件历史不回滚。不支持的任意 metadata 对象抛 TypeError。
- 测试证据：`test_snapshot_is_detached_json_and_restores_native_extras_rng_registry_and_last_retrieval` 检查动态 arms/registry、LinUCB 计数、RNG 与 JSON/不可变性；`test_committed_snapshot_restart_exact_state_and_scope_reset_does_not_resurrect` 检查重启/reset；两个 runtime readonly case 检查成功/失败恢复与文件 hash。

### 8. Native LLM 路由与依赖边界

- 上游行为：summary 与 memory-policy evolution 通过 ael.llm_utils→LiteLLM 调 provider；helper 设 drop_params=True、num_retries5、timeout120 并保留 model normalization。summary 异常回退字符截断，policy evolution 异常返回 None。
- MemAtlas 行为：保留 native helper、prompt、transport 与 fallback，不把它们替换为 MethodServices.llm；本地基础依赖 numpy/requests/PyYAML，optional summary/evolution 所需 pandas/litellm/tenacity/boto3 均在 manifest 明确记录，上游 finance-only 包另列。
- 原因：这条原生接口可以直接使用，无协议必要性要求改写 provider；按实际 import 路径区分本地依赖与启用特性依赖，避免隐式安装和错误缺包预检。
- 行为影响：可选 LLM 调用使用 native provider 凭据、LiteLLM 全局参数及其日志，不经过框架 usage ledger，internal_token_tracking=untracked；默认 metadata 检索不需要远程服务。触发可选模块前需按 README 安装依赖。
- 测试证据：`test_native_summary_prompt_options_and_provider_failure_truncation_are_preserved` 仅替换 litellm.completion I/O、验证原 prompt/options/model routing 与 fallback；preflight 测试检查缺少基础包时 unavailable、本地路径不强求远程依赖，无联网模型质量结论。

### 9. 原生源码、license 和日志交付

- 上游行为：所需 memory/controller/credit/reflection import 由原 package 提供，MIT LICENSE 写明 Copyright(c)2026 Anonymous Authors；memory store/bandits 大部分没有原生日志。
- MemAtlas 行为：vendor 14 个完整 native Python 模块、6个官方 YAML、README/requirements/LICENSE，共23个 SHA-256 文件；固定 revision，只有两个原空 package marker 补最终换行。source preflight 验证闭包/hash/revision，不初始化 provider。adapter logs 为 mematlas.methods.ael.adapter；ael.native.calls 明确是接入层插桩，真实 source log 保持 ael.evolution.* 等 namespace。
- 原因：plan/ 可以删除，必须闭合动态 llm_utils/reflection 导入并保留原 prompt 与许可；静默 native 算法需可区分的调用证据，但不能把 adapter 自写 trace 冒称上游原生日志。
- 行为影响：运行无需 historical checkout 或源码下载；外部 source_root override 显式保留，缺少 Git/清单的 archive 标记 revision unverified。没有改原算法代码，没有安装/更改共享依赖。
- 测试证据：14py 与其余文件对照固定 Git blob（仅已声明两个最终换行）；`test_vendored_inventory_detects_missing_tampered_and_wrong_revision` 检查预检；`test_complete_native_suite_runs_with_historical_source_inaccessible` 在禁止 plan 的子进程重跑；native add/retrieve/update 测试区分两个日志 namespace。

### 10. CLI 通用文本与原生 metadata 检索

- 上游行为：local store 的 relevance 从0开始，由 ticker、sector、task_type、planner、tool_name 等原生字段匹配加分；query 文本不参与向量检索。portfolio 默认 query 与 fallback 仍遵循本节第2项。
- MemAtlas 行为：保留 metadata 评分。通用 `experience.json` 没有匹配字段时，即使显式选择 `full_detailed` 也无命中；structured 样本通过 outcome metadata 写入合成 ticker/task_type，并用测试 metadata.memory_query 查询它们，配合 `full_detailed` 才得到命中。
- 原因：通用轨迹文本不自动等同原生金融 query/metadata，不能为方便 CLI 改成 embedding 检索或补造字段推断。
- 行为影响：AEL 需要调用方提供相容的结构字段。structured 对照仍通过通用 `add_episodic` 写入，没有覆盖含 tool_evaluations/prediction/actual 的 rich `add_episode_experience`。
- 测试证据：`.mematlas/runs/cli-e2e-20261024/` 下 `ael-default` 与 `ael-read-smoke` 两题均0命中；`ael-structured-read-smoke` 两题各1命中。输入见 `test/cli_e2e/{experience,ael-experience}.json`。

### 11. 默认冷启动策略与冻结随机状态

- 上游行为：memory bandit 使用 NumPy Thompson Sampling，候选包含 `none`；portfolio 无命中时仅回退 semantic/procedural。controller 的 frozen 分支不把全部 memory arms 改为 greedy 选择。
- MemAtlas 行为：保留 seed42 和默认策略集合。此次未获 controller 反馈的冷启动首次选择 `none`，只读快照又恢复 NumPy RNG，下一题再次选到 `none`；仅含 episodic 的库不能从上述 fallback 提供结果。
- 原因：标准 experience 导入没有训练期 controller selection/confirmed retrieval 上下文，测试又冻结学习；不能把单 policy smoke 覆盖写成默认策略来掩盖该行为。
- 行为影响：即使 structured 输入有可匹配 metadata，当前默认 CLI 流程仍两题0命中。这个结果由原生选择与框架冻结流程共同产生，不代表原生方法在完整训练后始终不检索，也不构成改写 Thompson Sampling 的依据。
- 测试证据：`.mematlas/runs/cli-e2e-20261024/ael-structured-default/` 的 prediction contribution 记录两题 `memory_policy=none`；最终 state 为2条 episodic、controller 更新0、day_index0。相同 structured 输入只显式覆盖 `memory_policy_names=["full_detailed"]` 后，两题各命中1条。

## AEL 参数变体

- 上游行为：README Full AEL 与 paper5seed 的 portfolio main path 使用 step3 tool_memory_reflect，固定 sequential/all、memory evolution、uniform credit，warmup15；D-full 覆盖 bars4，seed42。build_components 不传 tool_config_names，所以即使固定 all，工具 bandit 仍保留全部5个 preset arms。ael_full 配置是6个 planner/5个 preset 都演化、FCC、standalone bars1/warmup0；memory_only 固定 planner/tools、structural；step2b differential 保持 uniform 和固定 planner/tools。
- MemAtlas 行为：default/tool_memory_reflect 取 paper 有效 memory/controller 参数；ael_full、memory_only、tool_memory_diffmem 保留各自 native_config_path，episode 显式选择 ael_full 参数与 canonical runner。外层 EAEL/无 memory 消融在 x-unavailable-variants 声明，不当作已实现完整 agent。
- 原因：默认值需按论文运行入口和 dataset override 选择；memory 参数、runner quirk 与外层 agent 消融必须可核对。
- 行为影响：所有 profile 保留三层、k5、quality.3/eviction.2/cap500、raw_sign或原diffmem值；固定工具选择不裁剪 dormant arm 状态，只有显式 variant/override 改变。warmup15 是原始参数记录，调用方仍需提供 _warmup；不自行按次数修改 source 行为。
- 测试证据：bundled 原 YAML 与 native ExperimentConfig 字段已逐项核对；`test_official_main_profile_and_actual_native_components_initialize_without_providers` 检查默认值，runner baseline/periodic/nightly 测试分别检查有效行为；未执行 outer EAEL 消融回测。

## 本次 CLI 的显式测试参数

- 上游行为：官方 profile 及其模型、维度、batch、controller arms 由各方法 manifest/前述参数变体记录，不能用另选的测试组合冒称论文默认实验。
- MemAtlas 行为：测试 LLM 显式设为 `Qwen/Qwen2.5-72B-Instruct`；MemRL/MemQ embedding 为 `Qwen/Qwen3-Embedding-8B`、4096维，transport 使用 SiliconFlow。MemRL 另设 `llm_max_tokens=2048`。除默认算法对照，还运行已声明 `hle_episode`、MemQ `batch_size=1`、AEL 仅 `full_detailed` 的 smoke；MemCon 保留原生 CPU `all-MiniLM-L6-v2` 并从本地缓存加载。
- 原因：使用已有 provider/模型和小型合成输入检查真实 CLI 读写路径，并显式区分未满默认批次和已实际建库的情况。
- 行为影响：模型、向量维度、输出限额、批内可见性或策略选择与官方完整实验不同，不能推断论文效果。MemQ 仍为默认 epoch 更新模式，batch1 不代表执行过 Bellman epoch pass；AEL 单 policy 不代表训练过完整 controller。上述覆盖只存在测试配置，没有修改 method.yaml 的默认值或官方 variants。
- 测试证据：`test/cli_e2e/*.json`、[逐项结果与运行命令](test/cli_e2e/README.md)；MemRL/MemQ smoke 的原生 q_visits 均为0，MemQ pending edges2、epoch open，AEL controller 更新0。

## 【框架待优化项】

以下集中记录当前框架接口和流程限制；插件仅在 methods 接入层适配，core、runtime、providers 和基础调度保持不变。上游自身的 serializer、runner 或 REST 行为保留在对应方法的差异中，不作为框架缺陷修复。

### 1. Batch、epoch、交易日和 warmup 边界

- 上游行为：MemRL/MemQ runner 显式控制 batch/epoch；AEL runner 按交易 bar、warmup 和日末触发不同学习操作。
- MemAtlas 行为：通用协议提供 observe/retrieve/learn/commit/command，框架在每个 episode 后 commit，没有自动传递这些原生边界的 hook。接入层用显式命令表示 batch/epoch/end_day，AEL 从 metadata._warmup 接收原生阶段；当前公开 CLI/evaluation 路径没有发送这些 memory.command 的入口。
- 原因：完成一个框架 episode 不等于完成原生 mini-batch、epoch 或交易日，不能将持久化 commit 偷换为学习边界。
- 行为影响：MemRL/MemQ 调用方须显式结束尾批/epoch；AEL 调用方须提供 warmup 标记并发送 end_day。默认调度不会补齐这些命令，close 不代替学习收尾。小样本 CLI 可退出0但尚未执行原生建库或后续学习阶段。
- 测试证据：MemRL 的 `test_default_automatic_batch_and_explicit_tail_flush`、MemQ 的 `test_explicit_native_batch_preserves_update_add_and_boundary_order`、AEL 的 warmup/nightly 测试验证显式调用。2026-10-24 CLI 中 `memrl-default`/`memq-default` 最终各 pending batch2、原生记录0；`memq-batch-smoke` 记录2条但 pending edges2、epoch open；AEL day_index0，未收到 end_day，输入也未提供 _warmup，不能视为完成论文 warmup15。产物见 `.mematlas/runs/cli-e2e-20261024/`。

### 2. AEL 外层 agent 与 credit 上下文

- 上游行为：完整 AEL 执行 planner/tool、构造原生特征和连续 outcome、计算 FCC/structural credit，并把反思注入下一轮预测。周期及夜间 planner prompt 更新还会检索 semantic/procedural memory、增加 retrieval_count。
- MemAtlas 行为：BaseMemoryMethod 只有 memory 插槽，不能自动执行该 agent loop。adapter 执行原生 memory/controller、uniform credit 和显式日末的 memory-policy evolution；其他 credit、selection/features、成功工具调用和 reflection 由外层提供。不执行 planner prompt 更新或其附带检索。
- 原因：缺少 planner/tool 执行、反事实重跑、交易时钟、日反思和完整 trajectory 对象的统一接口；不能由 success 布尔值编造 FCC 或实际执行记录。
- 行为影响：这四个插件交付 memory 接入；AEL 完整论文 agent 尚需外层编排。非 uniform 模式需要真实外部 module_credits；零条确认记忆时 controller selection/features 需显式传回。省略 planner prompt 更新也会少产生其检索计数，可能影响后续 usefulness 排名；已在 AEL 差异中列明。
- 测试证据：AEL 的 credit 公式、空 confirmed、per-tool、原生 memory-policy prompt/新 arm 测试覆盖已实现部分；未声称执行完整金融回测、反事实 agent 或在线反思实验。

### 3. MemCon 的 task-start 身份传递

- 上游行为：runner 对每个新任务明确调用原生初始化，同名任务仍有独立边界。
- MemAtlas 行为：method 协议没有保证每次新任务都收到 task-start hook；adapter 从 decision/event metadata 的 episode ID 或任务文本变化识别边界。
- 原因：原生 task 生命周期与通用 observe/retrieve/learn 不完全对应，raw recall 也不保证提供 episode ID。
- 行为影响：显式 ID 可区分同名 episode；缺少 ID 且任务文本不变时，接口不足以判断是否开始新任务。同一 episode 的多步检索继续保留原生上下文。
- 测试证据：MemCon 的 `test_episode_start_identity_resets_same_task_and_survives_snapshot` 和跨任务 runtime readonly 用例验证身份及状态隔离。

### 4. JSON Schema nullable 类型校验

- 上游行为：原生合法参数中存在 null，例如 MemRL/MemQ 的部分 Q floor/ceiling 和 token 上限；JSON Schema 允许 type 联合列表。
- MemAtlas 行为：core.validation._matches_type 把 type 当作字典 key，收到列表会抛 TypeError。相关插件保留合法 nullable schema，并在插件局部处理联合类型。
- 原因：不能为绕过框架校验错误删去上游有效参数，本轮边界也不包含修改底层 validator。
- 行为影响：相关 memory 插件可校验合法 null；其他使用同类 schema 的方法仍受框架原有实现限制。
- 测试证据：MemRL/MemQ 的 profile/provider 预检和合法 nullable 参数检查通过；manifest 仍保留原生合法声明，未将 MemQ 原生仅接受数值的 q_min_threshold 扩为 null。

### 5. Source provenance 与依赖版本预检

- 上游行为：原实现依赖明确的源码 revision、文件和依赖版本；源码目录本身未提供 MemAtlas 的 source 获取与验证协议。
- MemAtlas 行为：基础 healthcheck 只检查显式 source_root 与包是否安装，不自动解析插件 bundled fallback，也不完整验证版本约束或来源。四个插件用 methods.source.inspect_source 补充 revision、闭包文件和 SHA-256 清单检查。
- 原因：plan/ 是可删除的参考资料，不能承担运行依赖；现有框架没有统一的 source acquisition、patch provenance 或依赖约束求解流程。
- 行为影响：四个默认源码随插件交付；缺失、哈希不符和已知错误 revision 预检失败。未附 Git/清单的显式外部 archive 标记 unverified；ready 不代表全环境满足所有版本约束，须按各 README 安装并执行 pip check。
- 测试证据：四插件 registry/preflight 均验证 bundled revision/hash 且不导入 native provider；各 suite 覆盖源码缺失/篡改，并在阻止 plan/ 文件访问和导入的子进程完成原生回归。未安装或调整共享环境依赖。

### 6. Provider response 与原生调用计量

- 上游行为：部分 native provider 使用 SDK response 的 usage/finish_reason、streaming 或独立 LiteLLM transport；返回值和重试边界不同。
- MemAtlas 行为：统一 LLM/embedding 协议仅提供最终文本/向量。MemRL/MemQ 适配原生调用边界并保留 runtime-owned services 以接受框架计量；MemQ 未提供的 SDK metadata 保持未知。AEL 可直接使用的原生 LiteLLM helper 保留原样，调用不经过框架服务。
- 原因：协议没有完整 SDK response/streaming 及原生外部 transport 的统一 instrumentation 接口；不能编造 usage、finish_reason 或改写原生 provider 策略。
- 行为影响：MemQ 使用 non-streaming facade，原生 SDK token counters 可能为零，框架 ledger 另计实际调用；AEL 可选原生 LLM 调用明确标记 internal_token_tracking=untracked，不保证纳入框架总计。
- 测试证据：MemRL 的 runtime provider 测试、MemQ 的 `test_runtime_provider_defaults_metering_and_public_overrides` 验证原生学习及框架 ledger；AEL summary/policy 测试验证原文 prompt、原生路由/参数和失败 fallback，替换的仅是外部 I/O。

### 7. 进程全局随机源隔离

- 上游行为：MemQ epsilon-greedy 使用 Python 全局 random；AEL Thompson Sampling 使用 NumPy 全局 RNG，原生没有每个 memory bank 的独立随机上下文。
- MemAtlas 行为：逻辑快照分别保存/恢复对应全局随机源，以恢复原生检索选择；框架没有 per-plugin RNG context。AEL 按原生实验 seed42 初始化，恢复 checkpoint 时以 checkpoint 的 NumPy state 为准，不回拨未使用的 Python random。
- 原因：冻结评估不能消耗训练随机流；改写上游为独立 RNG 会改变原生边界，不在本轮接入范围内。
- 行为影响：load_state/只读回滚可能影响同进程使用相同全局 RNG 的其他代码，不承诺并发 bank 的随机流隔离。UUID、时间和 provider 随机源不包含在此保证中。
- 测试证据：MemQ 的 `test_inline_snapshot_covers_qmax_retrieval_traces_pending_scopes_and_is_immutable` 验证 Python RNG；AEL 的完整快照及 runtime readonly 测试验证 NumPy RNG 恢复和 Python RNG 不被回拨。

### 8. CLI 的 outcome 输入与训练检索反馈流程

- 上游行为：MemRL/MemQ runner 在训练期检索并记录实际使用的记忆，结果确定后更新这些记忆的 Q/credit；MemCon/AEL 也依赖本任务的 action/selection/retrieval 上下文完成相应策略或 usefulness 更新。
- MemAtlas 行为：四插件均声明 `history_ingestion=outcome`。`StreamingMemoryProtocol.validate_inputs` 对此类方法只要 case 有 history 就拒绝，即使 case 已有合法 outcome。推荐的 experience 协议又在 `prime` 中仅调用 `ingest_history(..., outcome=...)`，不执行训练 retrieve；测试阶段使用 read_only，不反馈 learn。
- 原因：标准 CLI 的流式输入校验和 train/test 导入流程无法表达这些方法需要的训练检索→确认使用→反馈顺序；这属于框架协议编排缺口，不能在 adapter 中虚构已检索记忆或选择上下文。
- 行为影响：两种标准 CLI 路径都不能据本次数据验收完整反馈闭环。训练 learn 计数和成功写入不证明已使用记忆的 Q、controller 或 usefulness 更新完成；需保留各方法显式调用测试的证据边界。
- 测试证据：`evaluation/protocols/memory.py` 的 streaming 校验与 experience prime；`.mematlas/runs/cli-e2e-20261024/{memrl,memcon,memq,ael}-streaming/` 均在 start 阶段以相同 `CapabilityMismatchError` 失败，MemRL 使用合法 message 输入，其余使用合法 actions 输入。experience trace 训练 retrieve 数均0，MemRL/MemQ smoke 的 q_visits 均0，AEL controller 更新0。最初非法 fixture 和 MemRL 不支持 episode_start 的探针不计入该结论。

### 9. Provider 失败尝试的异常链可观测性

- 上游行为：原生同步 provider 调用可能包含重试；SDK 的 `APIConnectionError` 既可能包装网络异常，也可能包装本地客户端/事件循环异常，单看名称不足以确认根因。
- MemAtlas 行为：`providers.requests.metered_request` 的 ledger 只保存失败异常类型，没有保存 cause/context；外层 memory operation 在重试成功后仍标记成功。MemRL/MemCon/MemQ 的本次实际读写运行分别记录4/5/6次 `APIConnectionError`，随后均恢复。
- 原因：现有计量信息能够统计失败尝试，但不足以定位被 SDK 包装的原始错误；日志中也未找到可确认原因的异常链。
- 行为影响：不能将退出0或 operation failures0解释为 provider 从未报错，也不能把这些失败直接认定为外网故障。同步桥接反复创建事件循环、同时复用异步客户端是待查线索，尚未证实，未据此实施改写。
- 测试证据：`.mematlas/runs/cli-e2e-20261024/{memrl-episode,memcon-default,memq-batch-smoke}/artifacts/runs/*/usage.jsonl` 的失败/重试记录与对应 cli.log；`providers/requests.py` 的错误记录字段，以及 `methods/memrl/core.py`、`methods/memcon/core.py`、`methods/memq/core.py` 的同步桥接代码。检查只读取异常类型与调用序列，未记录凭据或请求内容。

# 真实 CLI 核查（2026-10-24）

结论：**四方法尚不能通过完整学习闭环的端到端验收。** 以下只记录本次真实 CLI 结果。

从 `fix/ael-integration@50200dc` 依序运行 MemRL → MemCon → MemQ → AEL，调用 `.venv/bin/python -m mematlas run`。使用真实 SiliconFlow provider 和原生本地数据库，没有替换算法或 provider。每个 experience 运行包含2条训练、2条测试，显式共享同一 scope。

| 运行 | 原生持久条目 | 两题各自命中 | 尚未执行的学习 |
| --- | ---: | --- | --- |
| MemRL 默认 HLE，batch32 | 0 | 0 / 0 | 2条训练仍在 pending |
| MemRL 显式 `hle_episode`，batch1 | 2 | 1 / 1 | 无训练检索后的 Q credit 证据 |
| MemCon 默认 | 4 | 2 / 2 | 导入不能证明检索动作的 policy 更新 |
| MemQ 默认 LCB，batch50 | 0 | 0 / 0 | pending2，epoch 未结束 |
| MemQ 显式 batch1 | 2 | 1 / 1 | pending edges2，epoch 未结束 |
| AEL 默认，匹配 metadata | 2 episodic | 0 / 0 | controller 更新0，day0 |
| AEL 显式 `full_detailed`，匹配 metadata | 2 episodic | 1 / 1 | controller 更新0，day0 |

以上运行均退出0、框架操作失败数0。有命中的四项，两题回答均为 `COBALT-42`，exact match=1。MemCon 的4条记录保留了原生双写行为。补充的 AEL 通用文本输入对照，无论默认策略还是显式 `full_detailed` 均0命中，符合原生 metadata 检索要求。

四方法的合法 `streaming` 输入均在 start 阶段被协议拒绝，退出1。`experience` 训练只导入历史，没有 retrieve，测试阶段只读、不 learn；因此 MemRL/MemQ 的 q_visits 均为0，无法验证已使用记忆的反馈更新。CLI 也没有发送原生边界命令，MemQ epoch、AEL nightly 未执行，AEL 输入未提供论文 warmup 阶段标记。

`streaming` 的具体拦截来自 MemAtlas `StreamingMemoryProtocol` 的[启动校验](evaluation/protocols/memory.py#L39)：当协议为 streaming、任一 case 含 history、插件声明 `history_ingestion="outcome"` 时，抛出 `CapabilityMismatchError: Streaming history requires labeled ingestion for this method; use the train/test protocol`。四个插件均声明 outcome，表示导入历史必须附带 success/reward 等任务结果；当前校验只检查 history 和策略，即使 case 已携带合法 outcome 也会拒绝。本次四份失败报告均为 `stage=start`。

调用链显示，streaming 先在[历史导入](evaluation/protocols/memory.py#L120)中调用 `ingest_history(history)`，没有传 outcome，然后才检索/回答，最后调用 `learn(outcome, retrieved=...)`。而 [runtime 的历史导入规则](runtime/memory.py#L268)要求 outcome 策略必须有标签，否则拒绝。因此仅删除启动校验仍会遇到 runtime 拒绝；直接提前传入当前 case 的 outcome 又会在预测前触发 learn，并与[预测后的 learn](evaluation/protocols/memory.py#L131)重复，可能让当前任务结果提前进入记忆。

上述定位属于框架历史导入与在线反馈的编排缺口，不能据此判定四种原生算法不支持在线学习。修正需保证“观察事件 → 检索/回答 → 确认使用并反馈学习”的顺序，并保留各方法的原生 batch/epoch/day 边界；本次只记录实测拒绝和源码调用链，没有删除校验或修改协议。

AEL 默认冷启动在 seed42 下选到 `none`；冻结快照恢复 NumPy RNG，第二题也选到 `none`。仅含 episodic 的库无法从 semantic/procedural fallback 返回记忆。显式单 policy 的成功结果不代替默认 controller 学习验收。

MemRL/MemCon/MemQ 读写运行分别有4/5/6次 `APIConnectionError`，重试后恢复；异常链未记录，根因未确认。MemRL 空库恢复记录 MemOS `Empty update request`，MemCon 记录 Chroma telemetry 签名错误。退出0不代表无原生错误日志或 provider 失败尝试。

配置、数据、可复现命令和日志分类见 [test/cli_e2e/README.md](test/cli_e2e/README.md)。完整本地产物为 `.mematlas/runs/cli-e2e-20261024/`，只读汇总工具为 `test/cli_e2e/summarize.py`；框架缺口统一记录在 [difference.md](difference.md) 的【框架待优化项】。

本次未独立测量冻结前后逻辑 state hash，未验证完整论文 agent、跨进程恢复或论文分数。诊断 guard 只阻断常规 Python open/import 对 `plan/` 词法路径的访问，不是完整文件系统沙箱；运行使用各插件内置源码，未发现依赖 `plan/` 的证据。未修改底层框架、原算法或依赖，未合并 main。

MemRL 原生快照会持久化展开后的 provider 凭据。本地产物根目录权限已设为700，复现脚本设置 `umask 077`；不要直接分享整个原始产物目录。

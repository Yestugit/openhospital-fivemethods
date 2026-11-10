已完成并提交到内层仓库分支 **`refactor/rsi-evaluation`**，基于 `refactor/evaluation-qa@882da1d`。

主要完成了 **5 项改进**：

1. **统一 Python 测评入口。**`ExperimentConfig` 直接配置 OpenHospital，使用 `runner.check()`、`runner.evaluate()`、`runner.results()`。场景选择独立放在 [scenes.py](<C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/evaluation/scenes.py>)，新增场景通过适配与登记接入。

2. **简化配置和服务启动。**保留三种病例流，删除重复规模参数和独立反馈生成选项。实现 `OpenHospitalServices(experiment)`，自动启动所需服务、等待就绪、退出时关闭。

3. **落实三层职责。**公共组件处理回合状态与冻结校验；医院保留诊疗、临床评分和经历映射；实验层负责调度、反馈交付和检查点评测。删除医院里的重复统计，将 MemRL 专用字段处理移到方法适配。

4. **完成适配与恢复。**移除失效的 `runtime.state_mode` 引用，接通当前 Memory、测评和方法适配链路。支持从原运行加载配置，恢复待反馈经历、训练进度和检查点。

5. **同步结果和 README。**报告补充临床评分、学习变化与进度；[改进版 README](<C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/docs/测评/OpenHospital_RSI快速上手_改进版.md>) 已更新为实际 Python 用法。

验证结果：**146 项回归通过**；最后的路径与条件检查修复后，**39 项相关专项测试通过**。真实 Qwen＋Mem0＋医院服务完成了 **1 例训练、训练前后各 1 次固定测试及恢复**，无失败，评分覆盖率为 100%。

| Commit | 内容 |
|---|---|
| `229e889` | 保存已确认的改进方案 |
| `4e212e4` | 实现统一入口、服务管理、公共回合与方法适配 |
| `6b018fe` | 补齐恢复路径和医生模型条件检查 |
| `4c69560` | 同步 Python README 与实跑验证说明 |

工作区中其他已有改动已保留。

---

11.1
可直接写入日志：

## OpenHospital RSI 测评改进

分支：`refactor/rsi-evaluation`  
基于：`refactor/evaluation-qa`（`882da1d`）

### 主要修改

1. 统一 OpenHospital 的测评入口，接入公共 `ExperimentRunner.check(...)` 与 `evaluate(...)`。
2. 增加通用场景注册机制，新场景通过适配器接入，Runner 无需重复修改。
3. 统一医院实验配置，集中管理病例流、训练规模、测试规模、检查点、反馈和输出目录。
4. 增加医院服务管理，支持按需启动、健康检查、端口冲突检查和自动关闭。
5. 保留三种病例流：`repeated`、`novel_same_department`、`novel_departments`。
6. 支持默认检查点和自定义检查点，支持实验中断后的配置恢复。
7. 将公共回合记录、冻结保护、状态校验和资源统计收拢到公共评测层。
8. 将 MemRL 的 `task_type`、`task_id` 和批次结束处理移回 MemRL 方法适配层。
9. 完善 RSI 结果报告，增加临床指标、评分覆盖率、学习进度和失败统计。
10. 更新 OpenHospital RSI 用户快速上手文档，补充配置、预检、服务启动、执行、结果读取和恢复示例。

### 新建文件

1. `evaluation/scenes.py`
2. `integrations/openhospital/config.py`
3. `integrations/openhospital/services.py`
4. `test/evaluation/test_rsi_public_flow.py`
5. `docs/测评/OpenHospital_RSI快速上手_改进版.md`

### 主要修改文件

1. `evaluation/experiment.py`
2. `evaluation/execution.py`
3. `evaluation/experience_memory.py`
4. `evaluation/qa.py`
5. `evaluation/reporting/writer.py`
6. `integrations/openhospital/application.py`
7. `integrations/openhospital/bridge.py`
8. `integrations/openhospital/execution.py`
9. `integrations/openhospital/planning.py`
10. `integrations/openhospital/cli.py`
11. `integrations/openhospital/__init__.py`
12. `methods/memrl/qa.py`
13. 相关 OpenHospital 测试文件

### Git 提交

1. `229e889`：新增 RSI 测评改进方案文档。
2. `4e212e4`：统一医院测评入口、服务管理与公共回合适配。
3. `6b018fe`：补齐恢复路径和医生模型条件检查。
4. `4c69560`：同步 Python 快速上手示例和真实链路验证结果。

### 验证结果

1. 常规回归测试：146 项通过。
2. RSI 专项测试：39 项通过。
3. 完成真实 Qwen、Mem0、OpenHospital 小规模链路验证。
4. 验证训练、固定测试、训练后测试、保存恢复、评分覆盖率和服务关闭流程。

当前工作区另有未提交文件：`.gitignore`、`docs/测评/OpenHospital_RSI快速上手.md`、`docs/测评/bug修复日志.md`、`docs/测评/测评用户快速上手新_改进版.md`。

举三个**现有代码中的具体例子**：

1. **冻结检查：从医院代码移到公共组件。**  
   现在 [医院 execution.py](<C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/integrations/openhospital/execution.py:113>) 的 `_frozen_operation()` 自己保存、恢复 Memory 状态，并检查诊疗或评分有没有改变冻结记忆。  
   **准备改为：**将这段通用保护逻辑收拢到公共 `evaluation/experience_memory.py`，医院执行时调用它。其他场景也能复用。

2. **资源统计：删除重复且未使用的计算。**  
   现在 [医院 bridge.py](<C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/integrations/openhospital/bridge.py:322>) 的 `resource_snapshot()` 自己计算检索次数、耗时、Token 和存储量；当前测评链路没有调用它，公共统计已有对应能力。  
   **准备改为：**删除这段重复代码，统一使用公共统计；医院侧继续采集医生和医院服务的用量。

3. **MemRL 专用处理：从医院桥接移到方法适配。**  
   现在医院 `bridge.py` 中直接写着：

   ```python
   if self.core_memory.method_id == "memrl":
       extensions["memrl"] = {
           "task_id": occurrence,
           "task_type": "clinical_reasoning",
           "end_of_training": occurrence == self._training_end,
       }
   ```

   **准备改为：**医院提供诊疗经历，实验层提供是否结束训练等信息，由 MemRL 自己的适配代码补充这些专有字段。已有 [MemRLQA.prepare_episode()](<C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/methods/memrl/qa.py:9>) 已处理训练结束标记，这次会接通并收拢相关处理。

这三处分别对应：**公共能力迁移、重复代码删除、方法职责归位。**


**公共测评部分主要改了以下 6 处。原有训练调度和评分逻辑没有重写，但增加了场景选择、配置保存和冻结校验。**

| 文件 | 修改内容 | 对原有逻辑的影响 |
|---|---|---|
| `evaluation/scenes.py`（新增） | 按 `benchmark.kind` 选择已登记的场景适配器 | OpenHospital 可自动接入；普通 Benchmark 继续走原有测评流程 |
| `evaluation/experiment.py` | Runner 自动选择场景；保存 `experiment.json`；支持从运行目录加载配置；统一解析相对路径 | 增加配置保存与恢复步骤，原有 `check / evaluate / results` 用法保持 |
| `evaluation/experience_memory.py` | 提供公共的回合完成记录、经历读取和冻结操作保护 | 医院复用这些能力，减少重复实现；经历内容和学习提交方式保持 |
| `evaluation/execution.py` | 普通测评执行每个测试案例时，增加冻结状态校验 | **行为更严格**：测试若改变受保护的记忆状态，会报错并阻止缓存结果；增加了状态采集与校验开销 |
| `evaluation/qa.py` | 增加 `retrieval_extensions(metadata)`，统一准备方法所需的检索扩展参数 | 默认仍读取原来的 `memory_extensions`，普通问答的默认输入保持 |
| `evaluation/reporting/writer.py` | 增加临床指标和学习进度展示 | 影响报告呈现，评分计算保持原样 |

**需要重点区分两点：**

1. **主要执行逻辑保持。**三种实验安排、训练批次调度、检查点调度、原生评分和学习曲线计算没有在本轮重写；`Memory` 核心接口与方法算法也没有修改。
2. **公共行为有两项新增。**每次运行会保存实验配置；普通冻结测试会逐例检查记忆状态。因此，这轮对原有逻辑有局部影响，最明显的是冻结检查更严格。`evolve` 没有新增自动调用，也没有与检查点绑定。
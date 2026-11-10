# Hospital Agent SDK

`hospital_agent_sdk` 是虚拟诊疗比赛使用的 Python SDK，用于帮助参赛队伍快速编写医生 Agent，并对接比赛提供的患者、检查、诊断、治疗和评测服务。

主要文件说明：

- `agent.py`：定义基础医生 Agent 类，参赛者可以继承它实现自己的训练和测试逻辑。
- `actions.py`：封装医生可执行的主要操作，如询问患者、申请检查、给出诊断和制定治疗方案。
- `client.py`：负责和比赛后端服务通信，获取患者信息、调用接口和提交结果。
- `llm.py`：封装大模型调用逻辑，供 Agent 在诊疗过程中使用。
- `runtime.py` / `builder.py`：负责加载配置、初始化运行环境，并启动训练或测试流程。
- `server.py`：提供测试阶段使用的 Flask 服务入口。
- `event_logger.py`：记录诊疗过程和运行结果，便于后续分析。

参赛项目通常只需要在自己的 `agent.py`、`train.py`、`memory.py` 中编写业务逻辑，并通过本 SDK 调用比赛相关能力。

## 用法

参赛者可以继承 SDK 提供的 `BasicAgent`，在自己的 Agent 中实现 `train` 和 `test` 方法，并通过 `self.actions` 调用询问患者、申请检查、诊断和治疗等能力。训练阶段可以在本地脚本中直接运行 Agent，测试阶段则通过 SDK 提供的服务入口对接比赛平台。

## Quick Start

安装 SDK：

```bash
pip install hospital-agent-sdk
```

在项目代码中使用：

```python
from hospital_agent_sdk import BasicAgent, AgentBuilder, load_config


class MyDoctorAgent(BasicAgent):
    async def train(self, patient_id: str):
        return await self.actions.prescribe_treatment(
            patient_id=patient_id,
            diagnosis=["待完善诊断"],
            treatment_plan="待完善治疗方案",
            reasoning="待完善推理过程",
        )

    async def test(self, patient_id: str):
        return await self.train(patient_id)


config = load_config("config.yaml")
agent = MyDoctorAgent(config=config)
AgentBuilder(agent).start()
```

本地运行前通常需要在 `config.yaml` 中配置输出目录和训练参数，并通过环境变量提供比赛服务地址、训练 token、模型 API Key 和队伍 ID。

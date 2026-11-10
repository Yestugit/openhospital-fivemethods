

| 文件 | 内容 | 实跑情况 |
|---|---|---|
| [01_react_agent_memory.py](/C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/demo/01_react_agent_memory.py) | ReAct 检索、每轮 reflection、回答并提交学习 | 通过，正确回答出行计划 |
| [02_memoryos_locomo.py](/C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/demo/02_memoryos_locomo.py) | 默认 MemoryOS 测评 LoCoMo 前 5 题 | 5 题完成，0 失败，原生 F1 为 0.3167 |
| [03_react_agent_locomo.py](/C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/demo/03_react_agent_locomo.py) | 自定义 ReAct + reflection Agent 测评前 2 题 | 2 题完成，0 失败；短期容量设为 256，减少首次构建耗时 |
| [04_memoryos_openhospital_rsi.py](/C:/Users/12084/Desktop/研究生/code/MemAtlas/src/mematlas/demo/04_memoryos_openhospital_rsi.py) | 2 次训练、1 个固定测试病例在 3 个检查点测试，共 5 次诊疗 | 文件已完成，实跑通过 |

每个文件开头都有安装与运行命令，配置直接写在代码里，没有增加其他配置文件或说明文档。


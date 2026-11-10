# MemRL 本地验证

更新：2026-10-31。——10.31

在 `src/mematlas` 目录使用独立 Python 环境。依赖与版本以 `methods/memrl/method.yaml` 为准：

```python
import json, subprocess, sys
from pathlib import Path
requirements = json.loads(Path("methods/memrl/method.yaml").read_text())["requirements"]
subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil",
    *(name + requirements["python_constraints"].get(name, "") for name in requirements["python"])])
```

```bash
python test/memrl/check_memrl.py
```

测试使用内置原生算法、受控模型与本地存储，不读取业务密钥。覆盖原生更新顺序、真实反馈、检索归因、自动演化、冻结、快照和恢复；部分用例使用真实 MemOS、Qdrant 与 SQLite。

用户配置、输入与调用要求通过 `help(mematlas.methods.memrl)` 查看，入口见 [方法包](../../methods/memrl/__init__.py)，完整调用见 [example.py](../../methods/memrl/example.py)。默认 HLE 批大小为 32；`learn` 积累经历，满批自动演化，不提供公共 `evolve`；`flush/save/close` 保存待处理状态，恢复后继续积累。

当前约束包括 MemoryOS 1.0.0、OpenAI SDK 1.x、Qdrant Client 1.14.2—1.15.x。内置源码版本与文件校验信息见 `_vendor/UPSTREAM.json`，测试会核对来源及 SHA-256。

# 测评入口：streaming 与 experience

本文介绍两种测评入口。它们都用来回答一个问题：

> Memory 方法读过历史经历后，能不能帮助 Agent 在后面的任务中答得更好？

Experience 会先把 train 案例交给 Memory，再用同一个 Memory 处理 test 案例；streaming 则按案例逐个处理。两条路线最后都会比较 Agent 的答案和参考答案。

这里的train是让 Memory 读取经历、按方法规则更新记忆。它不训练 LLM 的参数。

第一次使用时，先按下面的表选择路线，再按“快速开始”一步步操作：

| 你的数据 | 选择的 protocol | 适合回答的问题 |
| --- | --- | --- |
| LoCoMo、LongMemEval，或没有 train/test 划分的数据 | `streaming` | 每道题带着自己的历史，Memory 能否帮助回答当前问题 |
| 明确分成 train 和 test 的数据 | `experience` | Memory 先读过训练经历后，能否帮助回答后续测试问题 |


## 1. 测评数据

当前内置了 LoCoMo 和 LongMemEval 的数据读取方式，也支持自定义 JSON/JSONL。内置 Benchmark 文件需要使用者自行准备；自定义数据可按实际实验划分 train 和 test。

| 数据类型 | 可用流程 | 数据特点 |
| --- | --- | --- |
| LoCoMo | `streaming` | 每道问题带有历史对话；当前适配器提供 test 案例 |
| LongMemEval | `streaming` | 每道问题带有历史会话；当前适配器提供 test 案例 |
| 自定义 JSON/JSONL | `streaming` 或 `experience` | 可按实验需要设置 train/test |

一个案例的主要字段如下：

| 字段 | 含义 | 用途 |
| --- | --- | --- |
| `history` | 已经发生的对话或事件 | 可导入 Memory，也可作为本次检索的上下文 |
| `query` | 当前需要回答的问题 | 交给负责回答的 Agent/LLM |
| `reference` | 参考答案 | 评分时用来判断回答质量 |
| `outcome` | 任务结果反馈，如成功与否、奖励或文字反馈 | 训练导入时可交给需要反馈的 Memory 方法 |

`reference` 和 `outcome` 用途不同：`reference` 是答案评分依据，`outcome` 是任务经历的结果信息。默认 CLI 的回答提示词不包含 `reference`，LLM 返回自己的答案，再由评测器与 `reference` 比较。可编程的自定义 Agent/answerer 回调能访问完整案例，因此实现回调时仍需避免把 `reference` 泄漏到回答过程。

LoCoMo 和 LongMemEval 当前没有 train split，适合测“每道题使用自己的历史，再回答问题”。若要测“先积累训练经历，再回答后续问题”，需要有 train/test 划分的数据。

## 2. 整体流程

~~~text
选择 protocol
    ├─ streaming：当前案例 history → retrieve → LLM 回答 → 评分
    └─ experience：导入 train history（方法需要时提供 outcome）
                   → 只读测试时 retrieve（可带 test history 作上下文）
                   → LLM 回答 → 评分
~~~

本文介绍的 CLI 测评会让 LLM 对每个案例回答一次，并使用检索到的 Memory。它不自动提供通用的多轮工具 Agent。

## 3. 测评代码逻辑

一次测评可以理解为五步：

1. 选择 Memory 方法、Benchmark 数据和测评流程。
2. 框架读取案例中的 history、query 和 reference；训练案例需要反馈时还会读取 outcome。
3. 根据流程把历史交给 Memory，并用当前问题检索相关记忆。
4. 默认 CLI 将问题和检索结果交给 LLM，生成一次回答。
5. 框架把回答与 reference 比较，统计检索和资源用量，并生成报告。

`streaming` 会逐题导入当前案例的 history 后再检索。`experience` 会先把 train 案例导入 Memory，再在只读模式下测 test 案例。test 案例的 history 可以作为当次检索上下文，但不会通过 Memory 的写入或学习接口保存。

默认 CLI 每个案例只生成一次回答，不会自动进行多轮工具调用。它会替用户完成数据读取、Memory 调用、答案评分和报告整理；用户主要需要准备 Benchmark、选择 Memory 配置并提供模型服务。

## 4. 实验方案与指标

比较 Memory 方法时，应尽量固定 Benchmark、案例顺序、回答模型与提示词、工具条件、模型参数和评分方式，只替换 Memory 配置。一次完整比较通常包括：选数据和案例范围、选方法、运行各方法、检查逐题答案和失败、比较质量与资源指标。

当前 CLI 有两种运行方式：

| 流程 | 运行方式 | 适用情况 |
| --- | --- | --- |
| `streaming` | 每个案例到来时导入它的 history，检索并回答，再评分 | 不需要单独 train split；常用于 LoCoMo、LongMemEval |
| `experience` | train 案例导入 history；方法需要结果反馈时同时提供 outcome；test 案例只做只读检索和评分 | 研究训练经历能否改善后续测试 |

Experience 的 CLI 训练阶段不会重新执行训练任务。它把已有 `history` 和方法需要的 `outcome` 导入 Memory，由具体方法决定如何学习或提交。测试阶段进入只读模式，不调用写入或学习接口；测试案例若带有 `history`，仍会将其作为本次检索上下文。

质量分数与资源统计分开看：

| 类型 | 指标示例 | 说明 |
| --- | --- | --- |
| 通用答案指标 | `common.exact_match`、`common.token_f1` | 判断生成答案与参考答案的匹配程度；token F1 按空格切分文本 |
| Benchmark 指标 | LoCoMo 的 `native.qa_f1`、`native.retrieval_recall`；LongMemEval 的 `native.session_recall` | 由对应数据集定义；LongMemEval 的 `native.judge_accuracy` 需要启用 LLM judge，CLI 默认启用，也可用 `--no-official-judge` 关闭 |
| 资源统计 | `common.retrieval_count`、`common.observe_count`、`common.learn_count`、Token 和成本 | 记录 Memory 操作、模型用量与估算费用；部分值可能因能力或价格配置而为空 |

质量指标衡量回答或检索效果，资源统计衡量运行过程和成本。不要只凭一次 exact match，或只凭检索命中，判断一个方法的整体表现。

## 5. 快速开始

跑一次最小测评只需要三样东西：

1. 一个 Memory 配置文件，决定使用哪种方法。
2. 一个 JSON 或 JSONL Benchmark 文件。
3. 一个可以调用的 LLM，以及对应的 API 密钥环境变量。

下面的命令都在已安装项目依赖的 Python 环境中运行。直接使用源码时，请在包含 mematlas 目录的父目录执行命令。看到 mematlas 文件夹后，就可以执行下面的命令。data 和 configs 文件夹可以放在当前工作目录中，后面命令里的相对路径都从这里计算。若已经安装了 mematlas，则可在自己的实验目录运行，并在该目录下创建 data 和 configs。

先确认命令行入口可用，再查看当前可用方法：

~~~bash
python -m mematlas --help
python -m mematlas methods list
~~~

help 命令能显示用法和子命令；methods list 会逐行显示方法名称和简介。记下第一列的方法名称，下一步用它查询参数。

第一次使用时，按下面的顺序执行。每一步都不调用模型，直到第 7 步：

| 步骤 | 命令 | 你要确认什么 |
| --- | --- | --- |
| 1 | python -m mematlas --help | 能看到 methods、benchmark、check、run、results 等命令 |
| 2 | python -m mematlas methods list | 找到想比较的方法名称 |
| 3 | python -m mematlas benchmark ... | 数据文件能被读取，案例数量正确 |
| 4 | python -m mematlas methods describe 方法名 | 看到该方法的参数、依赖和支持能力 |
| 5 | 编写方法配置文件 | method.method_id 与第 4 步的方法名一致 |
| 6 | python -m mematlas check 配置文件 | 输出 ready 为 true |
| 7 | python -m mematlas run ... | 开始调用模型并生成报告 |
| 8 | python -m mematlas results 报告目录 | 查看答案、指标和失败信息 |

如果第 3 步失败，先检查数据路径和数据格式；如果第 6 步的 ready 为 false，检查 errors 和 health 中的缺少依赖或服务信息，按提示修正后重新执行第 6 步；第 7 步失败时，先查看报告目录中的 failures，再决定是否修改配置或数据。

### 第一步：准备 Benchmark 数据

框架已经内置 LoCoMo 和 LongMemEval 的读取适配，不必把它们改写成自定义格式。适配代码在项目中，但数据集文件需要你自己准备。

先确认数据文件能被正确读取。命令会显示 test 案例数和其中的历史条目数，不会调用 LLM。LoCoMo 文件格式应为数据集提供的 JSON：

~~~bash
python -m mematlas benchmark locomo data/locomo.json --limit 2
~~~

LongMemEval 使用自己的 JSON 文件格式，检查命令如下：

~~~bash
python -m mematlas benchmark longmemeval data/longmemeval.json --limit 2
~~~

检查输出中的 case_count 应大于 0。若命令报错或数量为 0，先确认数据文件路径正确，并且文件来自对应数据集、没有被改成自定义 JSON。查看其他选项可运行 python -m mematlas benchmark --help。

LoCoMo 和 LongMemEval 的案例目前都属于 test split。常规测评会先把每道题附带的历史对话写入 Memory，再让 LLM 回答。Experience 测评则需要先有 train 案例、再有 test 案例。当前这两个适配器没有提供 train split；因此如果要通过 CLI 跑 Experience，需要使用本身就有 train/test 划分的数据，或把自己的数据整理成下面的 JSON 格式。这里展示 custom 格式只是为了说明 Experience 的数据结构，不代表所有 Benchmark 都必须自定义。

自定义 Benchmark 是一个 JSON 文件，内容是案例数组，或者是包含 cases 数组的对象。每个案例最少包含 case_id、split、query 和 reference。下面的例子可以保存为 data/experience.json：

~~~json
{
  "cases": [
    {
      "case_id": "cedar-train-1",
      "split": "train",
      "scope": {
        "user_id": "demo-user",
        "session_id": "cedar"
      },
      "history": [
        {
          "event_id": "cedar-history-1",
          "role": "user",
          "text": "项目 Cedar 的发布代号是什么？"
        },
        {
          "event_id": "cedar-history-2",
          "role": "assistant",
          "text": "已核对发布说明，项目 Cedar 的发布代号是 COBALT-42。"
        }
      ],
      "query": "项目 Cedar 的发布代号是什么？",
      "reference": "COBALT-42",
      "outcome": {
        "success": true,
        "reward": 1.0
      }
    },
    {
      "case_id": "cedar-test-1",
      "split": "test",
      "scope": {
        "user_id": "demo-user",
        "session_id": "cedar"
      },
      "query": "请告诉我项目 Cedar 的发布代号。",
      "reference": "COBALT-42"
    }
  ]
}
~~~

字段的意思：

| 字段 | 用途 |
| --- | --- |
| split | train 表示先读入 Memory；test 表示只测效果 |
| history | 已经发生的对话或事件，训练阶段会写入 Memory |
| query | 当前要回答的问题 |
| reference | 正确答案，只交给评测器，不交给 Memory 或 LLM |
| scope | 记忆的归属。希望 train 和 test 共享记忆时，二者应使用相同 scope |
| outcome | 可选结果反馈。连续学习方法通常需要它 |

记住三点：训练案例的 history 会写入 Memory；测试案例的 query 会交给 LLM；在默认 CLI 流程中，reference 只用于评分，不要把它放进 history 或 query。train 和 test 使用相同的 scope，Memory 才能把它们当作同一用户的经历。

有些方法还需要训练案例的结果反馈。遇到缺少反馈字段的报错时，给 train 案例补充 outcome；可以用下面的命令查看所选方法需要哪些字段：

~~~bash
python -m mematlas methods describe 你的方法名
~~~

测试案例通常不需要 history 或 outcome。

写完自定义数据后，先检查它是否能被读取：

~~~bash
python -m mematlas benchmark custom data/experience.json
~~~

这个命令只检查数据格式并显示 test 案例数量，不会调用 LLM；当前检查命令不汇总 train 案例数量。

### 第二步：配置并检查 Memory

先查看所选方法支持的参数和依赖。例如，查看 BM25：

~~~bash
python -m mematlas methods describe baseline_bm25
~~~

把方法名替换成你在 methods list 中选出的名称。输出中的 config_schema 说明 params 可以填写什么；requirements 会列出额外依赖或模型服务。下面是一个不需要 Embedding 服务的 BM25 配置，可以保存为 configs/bm25.json：

~~~json
{
  "method": {
    "method_id": "baseline_bm25",
    "params": {
      "top_k": 5
    }
  }
}
~~~

把 baseline_bm25 换成其他方法的名称，并按该方法的 config_schema 修改 params。保存后，用 check 确认配置能否使用：

~~~bash
python -m mematlas check configs/bm25.json
~~~

check 会检查配置格式、方法依赖和运行条件，不会开始测评。若输出 ready 为 true，说明可以继续；若为 false，检查 errors 和 health 中缺少的依赖或服务信息，再修正配置或准备运行环境。

如果 methods describe 显示方法需要 llm 或 embedding 服务，在同一个 JSON 配置中增加 providers。例如：

~~~json
{
  "method": {"method_id": "你的方法名", "params": {}},
  "providers": {
    "llm": {
      "model": "你的模型名",
      "base_url": "https://你的服务地址/v1",
      "api_key_env": "MODEL_API_KEY"
    },
    "embedding": {
      "model": "你的向量模型名",
      "base_url": "https://你的服务地址/v1",
      "api_key_env": "MODEL_API_KEY",
      "dimensions": 1024
    }
  }
}
~~~

只保留该方法实际需要的服务；dimensions 按所选 Embedding 模型的输出维度填写。改完配置后，再运行一次 check。

### 第三步：配置模型并运行

CLI 默认使用 DashScope 的 qwen3.5-flash，并从环境变量 MODEL_API_KEY 读取密钥。先设置它：

~~~bash
export MODEL_API_KEY="你的模型服务密钥"
~~~

先根据数据类型选择测评路线：

| 你的数据 | 使用的流程 | 命令中的关键参数 |
| --- | --- | --- |
| LoCoMo 或 LongMemEval | 常规问答，每道题使用自己的历史对话 | protocol streaming |
| 自己整理的 train/test 数据 | Experience，先导入 train，再测 test | protocol experience、train-split train、split test |

按你选择的路线运行其中一种：

**路线 A：LoCoMo 或 LongMemEval 常规测评**

~~~bash
python -m mematlas run \
  --config configs/bm25.json \
  --benchmark locomo \
  --data data/locomo.json \
  --protocol streaming \
  --limit 2 \
  --report-dir reports/locomo
~~~

这里的 limit 2 只用于第一次小规模试跑，确认模型、Memory 和报告链路都正常。正式运行时删除这一行。使用 LongMemEval 时，把 locomo、data/locomo.json 和 reports/locomo 分别换成 longmemeval、对应数据文件和输出目录。

确认试跑成功后，删除 limit 2，再运行完整数据。

**路线 B：Experience 测评**

只有当数据含有 train/test 划分时使用：

~~~bash
python -m mematlas run \
  --config configs/bm25.json \
  --benchmark custom \
  --data data/experience.json \
  --protocol experience \
  --train-split train \
  --split test \
  --report-dir reports/experience
~~~

不确定某条命令的参数时，可以在运行前加 --help 查看说明，例如 python -m mematlas run --help。

命令行的 --model、--base-url 和 --api-key-env 配置的是“负责回答测试问题的 LLM”。部分 Memory 方法还需要自己的 LLM 或 Embedding 服务；这类配置写在 Memory 配置文件的 providers 部分。先按 methods describe 中的 requirements 配置，再运行 check。

如果回答测试问题的模型服务不是默认配置，在 run 命令中添加 --model、--base-url 和 --api-key-env，分别指定模型名、服务地址和密钥环境变量名。

Experience 的 custom 示例只有一个 train 和一个 test 案例，可以直接用来确认流程。正式实验时增加案例数量，并保持 train/test 的 scope 和 split 设置正确。

运行时发生的事情是：

1. 读取 train 案例的 history，写入所选 Memory。
2. 完成训练阶段后保留当前 Memory，测试期间只读取，不再写入或学习。
3. 对每个 test 案例检索 Memory，并让 LLM 根据问题和检索结果作答。
4. 将答案和 reference 比较，记录指标和资源用量。

训练阶段导入已有历史，不会重新让 Agent 执行训练案例。

### 第四步：查看结果

把 report-dir 的路径换成自己运行命令中指定的目录。目录通常包含：

~~~text
reports/<本次报告目录>/
├── report.md                 # 人类可读的汇总报告
├── results.json              # 完整预测、指标、资源和失败信息
└── <method_id>/<run_id>/
    ├── predictions.jsonl     # 每个案例的答案
    ├── traces.jsonl          # Memory 操作轨迹
    └── manifest.json         # 本次运行的配置和环境信息
~~~

先看 report.md：

- common.exact_match：标准化后的答案完全匹配比例。
- common.token_f1：答案的 token 级 F1。
- LoCoMo 的 native.qa_f1：LoCoMo 对答案的评分；native.retrieval_recall：证据被检索到的比例。
- LongMemEval 的 native.session_recall：相关历史会话被检索到的比例；默认还会调用 LLM judge 评分，结果在 native.judge_accuracy。judge 费用也会计入用量。
- common.retrieval_count：Memory 检索次数。
- common.observe_count：Memory 写入次数。
- common.learn_count：Memory 学习调用次数。
- common.total_evaluation_tokens：已统计到的模型 Token 数。
- common.total_evaluation_cost：已知价格下估算的费用；未配置价格时可能为空。
- 耗时字段：Memory 操作和整次运行的时间。

查看结果时，路径必须和 run 命令中的 report-dir 相同：

~~~bash
python -m mematlas results reports/locomo
python -m mematlas results reports/experience
~~~

这两个命令分别对应前面的 LoCoMo 和 Experience 示例。运行 LongMemEval 或其他输出目录时，也要把路径替换成对应的 report-dir。results 会输出完整结果；report.md 是便于浏览的汇总，predictions.jsonl 保存逐题答案。

run 命令结束时，终端会显示每个方法处理了多少题、失败多少题。如果失败数大于 0，打开 results.json 查看 failures 里的错误信息；如果预测数量少于预期，先确认 Benchmark 检查命令显示的案例数。

如果 predictions 数量少于预期，先看 failures；如果 failures 为空，再比较不同方法的指标。成本字段为空通常表示没有提供模型价格参数，不代表没有调用模型。

## 6. 最小运行示例：BM25

为了先确认基本测评链路可用，我用最简单的本地 Memory 方法 `baseline_bm25` 跑了两道合成测试题。BM25 不需要额外 Embedding 服务或 Memory 方法侧 LLM；评测时仍会调用回答问题的 LLM。

运行条件：2 条 train 案例、2 条 test 案例，`top_k=5`；回答模型为 `Qwen/Qwen2.5-72B-Instruct`。数据是仓库附带的 Cedar 示例，不是正式 Benchmark。

从包含 `mematlas` 文件夹的父目录执行：

本示例使用 SiliconFlow。先将自己的密钥设置到 `SILICONFLOW_API_KEY` 环境变量中，不要把密钥写进配置文件或命令：

~~~bash
export SILICONFLOW_API_KEY="你的 SiliconFlow API 密钥"
~~~

~~~bash
python -m mematlas run \
  --config configs/bm25.json \
  --benchmark custom \
  --data mematlas/test/cli_e2e/experience.json \
  --protocol experience \
  --train-split train \
  --split test \
  --model Qwen/Qwen2.5-72B-Instruct \
  --base-url https://api.siliconflow.cn/v1 \
  --api-key-env SILICONFLOW_API_KEY \
  --report-dir mematlas/.mematlas/runs/bm25-minimal-20261028
~~~

结果是 2/2 题回答正确，`exact_match=1.0`、`token_f1=1.0`，失败 0，Memory 检索 2 次，共记录 330 个模型 token。报告中的费用为 unknown，因为这次运行没有配置模型价格。

查看这次运行的完整结果：

~~~bash
python -m mematlas results mematlas/.mematlas/runs/bm25-minimal-20261028
~~~

这次结果只验证了从导入历史、检索、生成回答到评分和报告的基本流程。案例数量很少且为合成数据，不能据此判断 BM25 的正式效果或与其他方法的优劣。完整逐题结果保存在项目的 `.mematlas/runs/bm25-minimal-20261028/`。

## 7. 待确认问题

下面这些决定会影响正式实验方案。前面的 CLI 教程可用于接入核查；正式比较前，建议确认：

1. 主要结果使用 `streaming`、`experience`，还是同时报告两者。
2. Experience 的 train 数据是导入已有 history/outcome，还是由 Agent 实际执行任务后形成经历。当前 CLI 使用历史导入；Python 接口可配置训练 Agent。
3. `outcome` 要包含哪些字段，哪些结果反馈允许 Memory 使用；`reference` 默认只评分，是否保持这一边界。
4. test 案例是否允许携带 history。当前 CLI 会把它作为检索上下文，但测试阶段不调用 Memory 写入或学习接口。
5. 正式测评是否纳入多轮对话和工具调用；标准 CLI 当前为单次问答。
6. 最终主指标、每种方法的重复次数、随机种子，以及是否报告均值和方差。
7. 如何划分 train/test 和用户 scope，避免同一段经历同时进入训练和测试。
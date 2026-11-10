# OpenHospital Contest Service

Standalone HTTP API service for the OpenHospital contest carbon track. It owns patient data, examination lookup, evaluation, service-hosted doctor LLM calls, service-token authentication, concurrency limits, Prometheus metrics, and token usage records. The Flask/Vue web app in `../web_server` calls this service over HTTP instead of importing it as a Python package.

## Data Layout

```text
service/
├── config.yaml
├── asgi.py
├── server.py
├── data/
│   ├── normal_examinations/
│   ├── examination_aliases.json
│   ├── examination_indicators.json
│   └── patients_data/
│       ├── train/
│       ├── primary_test/
│       └── advanced_test/
├── patients/
├── exam/
└── evaluation/
```

Each `service/data/patients_data` subset contains `profiles.jsonl`, `examination_data.json`, and `ground_truth.json`.

`examination_indicators.json` is generated from all normal templates and existing patient results. It maps each unique canonical examination name to its indicator names, age groups, genders, and `abnormal_case_only` flags. Indicator IDs, indicator alias maps, and cross-examination overlays are not used at runtime. Historical examination request names remain supported through `examination_aliases.json`.

## Configuration

`service/config.yaml` controls the service:

- `server.host` and `server.port`: default local bind address. Docker overrides this through the `uvicorn` command in the root `Dockerfile`.
- `data_dir`: service data root.
- `pg_dsn`: PostgreSQL DSN for persisted token usage records, public model key rotation, and service access tokens. It must use the same database as `../web_server/backend/app/config.yaml` `pg_dsn` when the web service issues service tokens. The service stores aggregates in `contest_service_token_usage`, validates token hashes from `contest_service_access_tokens`, and requires request `team_id` to match the token-bound account.
- Missing examination results and imported static results are stored in `contest_patient_examination_results`, keyed by dataset, patient, and canonical examination name. Generated results are shared by all teams.
- `database.pool_max_size`: PostgreSQL connection pool maximum size used by auth and usage stores.
- `test_dataset`: `primary_test` or `advanced_test`.
- `model`: OpenAI-compatible endpoint used by patient dialogue, doctor-agent calls, and evaluation. These calls use the request API key or public key pool; `model.api_key` remains an optional compatibility fallback for standalone use.
- `exam_model`: dedicated channels for missing examination generation. Each `exam_model.channels` entry has its own `base_url`, `api_key`, `model`, timeout, and retry settings. Generation tries channels in order after a request failure, timeout, or invalid output. Generated rows keep `source=generated` and store the successful model in `model_name`; original rows keep `source=existing` without a model name. Multiple channels may use the same model. `EXAM_MODEL_API_KEY`, `EXAM_MODEL_API_KEY_2`, and so on can override channel keys. The previous flat `exam_model` format remains supported.
- `limits`: route-level concurrency limits for `/invoke`, `/llm/chat`, `/evaluate`, and `/evaluate/case`.
- `llm`: outbound LLM request concurrency and HTTP connection pool limits.

The following environment variables override matching values from `service/config.yaml`:

| Environment variable | Config field |
| --- | --- |
| `PG_DSN` | `pg_dsn` |
| `TEST_DATASET` | `test_dataset` |
| `API_KEY` | replaces `model.api_key` with one runtime key |
| `DB_POOL_MAX_SIZE` | `database.pool_max_size` |
| `INVOKE_MAX_CONCURRENCY` | `limits.invoke_max_concurrency` |
| `CHAT_MAX_CONCURRENCY` | `limits.chat_max_concurrency` |
| `EVALUATE_MAX_CONCURRENCY` | `limits.evaluate_max_concurrency` |
| `LLM_MAX_CONCURRENCY` | `llm.max_concurrency` |
| `LLM_HTTP_MAX_CONNECTIONS` | `llm.http_max_connections` |
| `LLM_HTTP_MAX_KEEPALIVE` | `llm.http_max_keepalive` |

Do not rely on committed credentials in `service/config.yaml` for deployment. Override `PG_DSN` and `API_KEY` in the runtime environment.

## Authentication

Train requests to `/invoke`, `/llm/chat`, `/evaluate`, and `/evaluate/case` must include an API key in the JSON body (`api_key`, `model_api_key`, or `model_key`; camelCase variants are also accepted). Test requests must not include any of those fields; the service rejects them and uses the built-in `model` configuration instead.

Except for `GET /health`, every service route requires both `X-Hospital-Service-Token: <service-token>` and a `team_id`/`teamId`/`account` value. GET routes pass it as a query parameter, and POST routes pass it in the JSON body. The service rejects requests whose team id does not exactly match the account bound to the service token. Token usage is aggregated by this `team_id`.

`POST /llm/chat` is the doctor-agent LLM proxy used by the silicon SDK during `/test`. It accepts either:

```json
{
  "team_id": "team_demo",
  "messages": [{"role": "user", "content": "Summarize this case."}],
  "temperature": 0.2
}
```

or a simple `prompt` plus optional `system_prompt`. Requests authenticate with the same `X-Hospital-Service-Token: <service-token>` header as `/patients`, `/invoke`, and `/exam/results`. In test mode, the temporary service token is enough; the service uses its configured model key and records usage for the token's account.

Keep `pg_dsn` in sync with `../web_server/backend/app/config.yaml` `pg_dsn` when the web service and contest service should share tokens and token usage.

Requests authenticate with `X-Hospital-Service-Token: <service-token>`. Train tokens are long-lived per team. Test tokens are temporary and can be revoked by the web service when an evaluation request ends.

## HTTP Routes

- `GET /health`: health check. Does not require authentication.
- `GET /metrics`: Prometheus metrics exposed by the ASGI app. Does not require authentication.
- `GET /patients`: list patient IDs and summaries for the authenticated dataset.
- `GET /usage`: return token usage for the authenticated team.
- `POST /invoke`: patient dialogue endpoint.
- `POST /llm/chat`: doctor-agent LLM proxy endpoint.
- `POST /exam/results`: examination result lookup.
- `POST /evaluate`: batch evaluation.
- `POST /evaluate/case`: single-case evaluation.

## Development

```bash
cd hospital_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn service.asgi:app --host 127.0.0.1 --port 8001 --reload
```

Service URL: `http://127.0.0.1:8001`

### Examination data

Validate that canonical indicator names and the generated catalog are current:

```bash
python -m hospital_service.service.scripts.build_examination_indicators --check
```

Rewrite confirmed indicator-name variants and rebuild the catalog:

```bash
python -m hospital_service.service.scripts.build_examination_indicators --write
```

Validate the existing-result database import without connecting to PostgreSQL:

```bash
python -m hospital_service.service.scripts.import_examination_results --dry-run
```

Import all existing results into PostgreSQL after configuring `PG_DSN`:

```bash
python -m hospital_service.service.scripts.import_examination_results
```

When `/exam/results` requests an applicable examination that the patient does not already have, the service sends the patient profile, the patient's original examination data, normal references, and the exact age- and gender-filtered indicator list to the model configured under `model`. The model determines each generated indicator's `normal` or `abnormal` status; the service validates the response structure and indicator names without clinically reassessing the generated values before persistence. Generated examinations do not use the public model Key pool and do not increment any team's Token usage.

## Docker

From the repository root:

```bash
docker build -t hospital-service .
docker run --rm -p 7860:7860 \
  -e PG_DSN='postgresql://user:password@host:5432/dbname' \
  -e API_KEY='your-model-api-key' \
  hospital-service
```

The root `Dockerfile` installs `hospital_service/requirements.txt`, copies `hospital_service/` into `/app`, and starts:

```bash
uvicorn service.asgi:app --host 0.0.0.0 --port 7860
```

## 复赛权限与切换

碳基训练/测试令牌从此版本部署起即被拒绝，与复赛是否启用无关；原训练密钥仍可登录网页。
医院服务与网页后端共享 PostgreSQL 账号及赛段状态。进入复赛后，每次使用训练/测试令牌
调用接口时都会核对固定晋级名单，仅硅基前 20 队可训练与测试。未晋级硅基和全部
碳基账号的旧训练密钥被服务端拒绝，但网页登录仍可使用原密钥。晋级队伍训练令牌
对应 `train`，复赛测试令牌必须对应 `advanced_test`；旧初赛测试令牌不能继续用于
复赛测试。管理员内部评估令牌仍可用于历史结果复核。

切换维护期间拒绝新的训练和测试请求。训练用量按进程缓冲，因此必须正常停止所有
医院服务进程和副本，等待在途请求结束；FastAPI lifespan 在退出时刷新该进程的
Token 用量缓冲。管理员只能在所有进程已停止且数据落盘后执行清零与启用复赛，
成功后再启动新版医院服务。不要仅重启网页或在医院服务仍运行时直接清零数据库。

完整操作顺序见 [网页服务复赛切换说明](../web_server/README.md#复赛切换2026-09)。
两套服务必须一起升级，并使用相同数据库。复赛测试令牌绑定复赛题库，旧
`TEST_DATASET=primary_test` 或灰度配置不能覆盖令牌所绑定的 `advanced_test`。

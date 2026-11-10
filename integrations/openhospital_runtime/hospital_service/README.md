# Hospital Service

这是一个面向虚拟诊疗比赛的服务端，提供患者数据、检查结果、诊疗交互和评测相关的 HTTP API。

服务代码位于 `hospital_service/` 目录，根目录 `Dockerfile` 可用于构建和部署该服务。当前容器入口使用 FastAPI ASGI 应用 `service.asgi:app`，默认监听容器内 `7860` 端口。

## Docker

```bash
docker build -t hospital-service .
docker run --rm -p 7860:7860 \
  -e PG_DSN='postgresql://user:password@host:5432/dbname' \
  -e API_KEY='your-model-api-key' \
  hospital-service
```

服务启动后：

- 健康检查：`http://127.0.0.1:7860/health`
- Prometheus 指标：`http://127.0.0.1:7860/metrics`

除 `/health` 和 `/metrics` 外，请求需要携带 `X-Hospital-Service-Token`，并在 query 或 JSON body 中提供与 token 绑定账号一致的 `team_id`。

更多配置、接口和本地开发说明见 [`hospital_service/README.md`](hospital_service/README.md)。

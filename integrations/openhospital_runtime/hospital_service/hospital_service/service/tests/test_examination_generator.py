from __future__ import annotations

import asyncio
import json
import unittest
from dataclasses import dataclass
from typing import Any

from hospital_service.service.exam.catalog import ExaminationIndicator
from hospital_service.service.exam.generator import (
    ExaminationModelChannel,
    ExaminationResultGenerator,
    SYSTEM_PROMPT,
    parse_generated_result,
)


@dataclass
class FakeResponse:
    content: str
    usage: dict[str, int]


class FakeClient:
    def __init__(
        self,
        responses: list[str | Exception],
        api_keys: list[str],
        model_names: list[str],
        calls: list[dict[str, Any]],
        **kwargs: Any,
    ):
        self.responses = responses
        self.calls = calls
        api_keys.append(str(kwargs["api_key"]))
        model_names.append(str(kwargs["model"]))

    async def chat(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return FakeResponse(
            content=response,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


class ExaminationGeneratorTest(unittest.TestCase):
    def test_system_prompt_is_chinese_and_enforces_indicator_scope(self) -> None:
        self.assertNotIn("You generate one clinically plausible examination result", SYSTEM_PROMPT)
        self.assertIn("指标自身检查所覆盖", SYSTEM_PROMPT)
        self.assertIn("不得包含证据原文、推理过程或病因分析", SYSTEM_PROMPT)
        self.assertIn("相关症状、病史和诊断作为推断背景", SYSTEM_PROMPT)
        self.assertIn("最终结论只能描述当前指标范围内", SYSTEM_PROMPT)
        self.assertIn("无法根据现有信息形成可靠判断", SYSTEM_PROMPT)
        self.assertNotIn("evidence", SYSTEM_PROMPT)

    def test_parser_requires_exact_indicator_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing indicators: 白细胞"):
            parse_generated_result(
                raw=json.dumps({
                    "examination": "血常规",
                    "indicators": [
                        {"name": "血红蛋白", "value": "88 g/L", "status": "abnormal"},
                    ],
                }, ensure_ascii=False),
                examination="血常规",
                expected_names=("血红蛋白", "白细胞"),
                model_name="qwen3.5-flash",
            )

    def test_parser_rejects_duplicate_and_illegal_status(self) -> None:
        duplicate = {
            "examination": "血常规",
            "indicators": [
                {"name": "血红蛋白", "value": "88", "status": "abnormal"},
                {"name": "血红蛋白", "value": "90", "status": "normal"},
            ],
        }
        with self.assertRaisesRegex(ValueError, "duplicate indicators"):
            parse_generated_result(
                raw=json.dumps(duplicate, ensure_ascii=False),
                examination="血常规",
                expected_names=("血红蛋白",),
                model_name="model",
            )
        duplicate["indicators"] = [
            {"name": "血红蛋白", "value": "88", "status": "uncertain"}
        ]
        with self.assertRaisesRegex(ValueError, "invalid status"):
            parse_generated_result(
                raw=json.dumps(duplicate, ensure_ascii=False),
                examination="血常规",
                expected_names=("血红蛋白",),
                model_name="model",
            )

    def test_parser_rejects_unexpected_fields(self) -> None:
        payload = {
            "examination": "血常规",
            "indicators": [
                {
                    "name": "血红蛋白",
                    "value": "88 g/L",
                    "status": "abnormal",
                    "evidence": "旧协议字段",
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            parse_generated_result(
                raw=json.dumps(payload, ensure_ascii=False),
                examination="血常规",
                expected_names=("血红蛋白",),
                model_name="model",
            )

    def test_retries_ambiguous_value_with_definite_output(self) -> None:
        api_keys: list[str] = []
        calls: list[dict[str, Any]] = []
        responses = [
            json.dumps({
                "examination": "阴囊体格检查",
                "indicators": [{
                    "name": "阴囊体格检查",
                    "value": "皮肤皱褶可能减少",
                    "status": "normal",
                }],
            }, ensure_ascii=False),
            json.dumps({
                "examination": "阴囊体格检查",
                "indicators": [{
                    "name": "阴囊体格检查",
                    "value": "皮肤皱褶减少",
                    "status": "normal",
                }],
            }, ensure_ascii=False),
        ]
        generator = self._generator(responses, api_keys, calls=calls)

        result = asyncio.run(generator.generate(
            examination="阴囊体格检查",
            indicators=(ExaminationIndicator("阴囊体格检查", (), (), False),),
            profile={"id": "Patient_001"},
            known_results={},
            normal_reference={"阴囊体格检查": "皮肤皱褶减少"},
        ))

        self.assertEqual(result.result, {"阴囊体格检查": "皮肤皱褶减少"})
        self.assertEqual(len(calls), 2)
        repair_request = json.loads(calls[1]["messages"][-1]["content"])
        self.assertEqual(
            [row["name"] for row in repair_request["required_indicators"]],
            ["阴囊体格检查"],
        )

    def test_repairs_only_blank_indicators_and_includes_normal_references(self) -> None:
        api_keys: list[str] = []
        calls: list[dict[str, Any]] = []
        responses = [
            json.dumps({
                "examination": "血常规",
                "indicators": [
                    {"name": "血红蛋白", "value": "140 g/L", "status": "normal"},
                    {"name": "白细胞", "value": "", "status": "normal"},
                ],
            }, ensure_ascii=False),
            json.dumps({
                "examination": "血常规",
                "indicators": [
                    {"name": "白细胞", "value": "7 x10^9/L", "status": "normal"},
                ],
            }, ensure_ascii=False),
        ]
        generator = self._generator(responses, api_keys, calls=calls)

        result = asyncio.run(generator.generate(
            examination="血常规",
            indicators=(
                ExaminationIndicator("血红蛋白", (), (), False),
                ExaminationIndicator("白细胞", (), (), False),
            ),
            profile={"id": "Patient_001"},
            known_results={},
            normal_reference={
                "血红蛋白": "130-175 g/L",
                "白细胞": "4-10 x10^9/L",
            },
        ))

        self.assertEqual(result.result, {
            "血红蛋白": "140 g/L",
            "白细胞": "7 x10^9/L",
        })
        initial_request = json.loads(calls[0]["messages"][1]["content"])
        self.assertEqual(initial_request["required_indicators"][1]["normal_reference"], "4-10 x10^9/L")
        repair_request = json.loads(calls[1]["messages"][-1]["content"])
        self.assertEqual(
            [row["name"] for row in repair_request["required_indicators"]],
            ["白细胞"],
        )

    def test_repair_explains_ambiguous_normal_reference_without_relaxing_validation(self) -> None:
        name = "终末导管小叶单位"
        reference = "终末导管小叶单位的数量和大小通常减少"
        definite = "终末导管小叶单位的数量和大小减少"
        def response(value):
            return json.dumps({"examination": "粗针穿刺活检（CNB）", "indicators": [
                {"name": name, "value": value, "status": "normal"}]}, ensure_ascii=False)
        calls = []
        result = asyncio.run(self._generator([response(reference), response(definite)], [], calls=calls).generate(
            examination="粗针穿刺活检（CNB）",
            indicators=(ExaminationIndicator(name, (), (), False),), profile={}, known_results={},
            normal_reference={name: reference}))
        self.assertEqual(result.result[name], definite)
        repair = json.loads(calls[-1]["messages"][-1]["content"])
        self.assertIn("ambiguous value", repair["validation_error"])
        self.assertIn("不得照抄其中的模糊措辞", repair["instruction"])
        self.assertEqual(repair["required_indicators"][0]["normal_reference"], reference)
        with self.assertRaisesRegex(RuntimeError, "ambiguous value"):
            asyncio.run(self._generator([response(reference), response(reference)], []).generate(
                examination="粗针穿刺活检（CNB）", indicators=(ExaminationIndicator(name, (), (), False),),
                profile={}, known_results={}, normal_reference={name: reference}))

    def test_model_status_is_preserved_without_rule_reassessment(self) -> None:
        api_keys: list[str] = []
        calls: list[dict[str, Any]] = []
        response = json.dumps({
            "examination": "直肠指检（DRE）",
            "indicators": [{
                "name": "前列腺",
                "value": (
                    "常因良性前列腺增生（BPH）而增大；其直径超过 4 cm，"
                    "但应仍保持光滑、对称，且质地呈橡皮样/有弹性"
                ),
                "status": "normal",
            }],
        }, ensure_ascii=False)
        generator = self._generator([response], api_keys, calls=calls)

        result = asyncio.run(generator.generate(
            examination="直肠指检（DRE）",
            indicators=(ExaminationIndicator("前列腺", (), ("male",), False),),
            profile={"demographics": {"age": 68, "gender": "男"}},
            known_results={},
            normal_reference={
                "前列腺": (
                    "常因良性前列腺增生（BPH）而增大；其直径可能超过 4 cm，"
                    "但应仍保持光滑、对称，且质地呈橡皮样/有弹性"
                ),
            },
        ))

        self.assertEqual(result.indicator_statuses, {"前列腺": "normal"})
        self.assertEqual(result.status, "normal")
        self.assertEqual(result.abnormal_indicators, ())

    def test_model_abnormal_value_is_preserved(self) -> None:
        api_keys: list[str] = []
        response = json.dumps({
            "examination": "前抽屉试验",
            "indicators": [{
                "name": "前抽屉试验结果",
                "value": "右膝前抽屉试验阳性",
                "status": "abnormal",
            }],
        }, ensure_ascii=False)
        generator = self._generator([response], api_keys)

        result = asyncio.run(generator.generate(
            examination="前抽屉试验",
            indicators=(ExaminationIndicator("前抽屉试验结果", (), (), False),),
            profile={"present_illness_history": {"symptoms": "既往记录：右膝前抽屉试验阳性"}},
            known_results={},
            normal_reference={"前抽屉试验结果": "阴性"},
        ))

        self.assertEqual(result.result, {"前抽屉试验结果": "右膝前抽屉试验阳性"})
        self.assertEqual(result.indicator_statuses, {"前抽屉试验结果": "abnormal"})

    def test_anthropometry_recomputes_bmi_from_height_and_weight(self) -> None:
        api_keys: list[str] = []
        response = json.dumps({
            "examination": "人体测量",
            "indicators": [
                {"name": "身高", "value": "170 cm", "status": "normal"},
                {"name": "体重", "value": "82.5 kg", "status": "normal"},
                {"name": "体重指数", "value": "26.8 kg/m²", "status": "normal"},
            ],
        }, ensure_ascii=False)
        generator = self._generator([response], api_keys)

        result = asyncio.run(generator.generate(
            examination="人体测量",
            indicators=(
                ExaminationIndicator("身高", (), (), False),
                ExaminationIndicator("体重", (), (), False),
                ExaminationIndicator("体重指数", (), (), False),
            ),
            profile={"id": "Patient_001"},
            known_results={},
            normal_reference={
                "身高": "160-180 cm",
                "体重": "60-90 kg",
                "体重指数": "22.0-29.9 kg/m²",
            },
        ))

        self.assertEqual(result.result["体重指数"], "28.5 kg/m²")
        self.assertEqual(result.indicator_statuses["体重指数"], "normal")

    def test_retries_invalid_names_with_configured_key(self) -> None:
        api_keys: list[str] = []
        responses = [
            json.dumps({"examination": "血常规", "indicators": []}, ensure_ascii=False),
            self._valid_response(),
        ]
        generator = self._generator(responses, api_keys)

        result = asyncio.run(generator.generate(
            examination="血常规",
            indicators=self._indicators(),
            profile={"id": "Patient_001"},
            known_results={},
        ))

        self.assertEqual(result.result, {"血红蛋白": "88 g/L"})
        self.assertEqual(api_keys, ["configured-key"])

    def test_falls_back_to_second_model_after_first_model_times_out(self) -> None:
        api_keys: list[str] = []
        model_names: list[str] = []
        generator = self._generator(
            [TimeoutError("primary model timed out"), self._valid_response()],
            api_keys,
            model_names=model_names,
            models=("primary-model", "fallback-model"),
        )

        result = asyncio.run(generator.generate(
            examination="血常规",
            indicators=self._indicators(),
            profile={"id": "Patient_001"},
            known_results={},
        ))

        self.assertEqual(result.model_name, "fallback-model")
        self.assertEqual(model_names, ["primary-model", "fallback-model"])
        self.assertEqual(api_keys, ["configured-key", "configured-key"])

    def test_falls_back_across_channels_with_separate_urls_and_keys(self) -> None:
        api_keys: list[str] = []
        model_names: list[str] = []
        client_options: list[dict[str, Any]] = []
        responses: list[str | Exception] = [
            TimeoutError("primary channel timed out"),
            self._valid_response(),
        ]

        def client_factory(**kwargs: Any) -> FakeClient:
            client_options.append(kwargs)
            return FakeClient(responses, api_keys, model_names, [], **kwargs)

        generator = ExaminationResultGenerator(
            model_channels=(
                ExaminationModelChannel(
                    model_api="https://primary.example/v1",
                    api_key="primary-key",
                    model_name="shared-model",
                    timeout_seconds=10,
                    request_retries=1,
                ),
                ExaminationModelChannel(
                    model_api="https://fallback.example/v1",
                    api_key="fallback-key",
                    model_name="shared-model",
                    timeout_seconds=20,
                    request_retries=1,
                ),
            ),
            client_factory=client_factory,
        )

        result = asyncio.run(generator.generate(
            examination="血常规",
            indicators=self._indicators(),
            profile={"id": "Patient_001"},
            known_results={},
        ))

        self.assertEqual(result.model_name, "shared-model")
        self.assertEqual(api_keys, ["primary-key", "fallback-key"])
        self.assertEqual(model_names, ["shared-model", "shared-model"])
        self.assertEqual(
            [options["model_api"] for options in client_options],
            ["https://primary.example/v1", "https://fallback.example/v1"],
        )

    def _generator(
        self,
        responses: list[str | Exception],
        api_keys: list[str],
        *,
        calls: list[dict[str, Any]] | None = None,
        model_names: list[str] | None = None,
        models: tuple[str, ...] = ("qwen3.5-flash",),
    ) -> ExaminationResultGenerator:
        captured_calls = calls if calls is not None else []
        captured_models = model_names if model_names is not None else []
        return ExaminationResultGenerator(
            model_api="https://example.com/v1",
            model_names=models,
            timeout_seconds=30,
            request_retries=2,
            api_key="configured-key",
            client_factory=lambda **kwargs: FakeClient(
                responses,
                api_keys,
                captured_models,
                captured_calls,
                **kwargs,
            ),
        )

    @staticmethod
    def _indicators() -> tuple[ExaminationIndicator, ...]:
        return (ExaminationIndicator("血红蛋白", (), (), False),)

    @staticmethod
    def _valid_response() -> str:
        return json.dumps({
            "examination": "血常规",
            "indicators": [
                {"name": "血红蛋白", "value": "88 g/L", "status": "abnormal"},
            ],
        }, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()

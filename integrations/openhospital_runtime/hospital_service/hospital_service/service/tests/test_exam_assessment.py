from __future__ import annotations

import unittest

from hospital_service.service.exam.assessment import (
    assess_examination_result,
    assess_indicator_value,
)


class IndicatorAssessmentTest(unittest.TestCase):
    def test_classifies_clear_normal_qualitative_values(self) -> None:
        for value in (
            "阴性",
            "未检出",
            "未见异常",
            "正常",
            "未分离出致病菌",
            "未检测到异常信号",
            "未见异常浸润或发育异常",
            "无局灶性异常；步态正常",
        ):
            with self.subTest(value=value):
                self.assertEqual(assess_indicator_value(value), "normal")

    def test_classifies_clear_abnormal_qualitative_values(self) -> None:
        for value in (
            "阳性",
            "检测结果异常",
            "明显增高",
            "低于参考范围",
            "检出致病菌",
        ):
            with self.subTest(value=value):
                self.assertEqual(assess_indicator_value(value), "abnormal")

    def test_normal_negation_takes_precedence_over_abnormal_tokens(self) -> None:
        self.assertEqual(assess_indicator_value("未见异常"), "normal")
        self.assertEqual(assess_indicator_value("未检出阳性信号"), "normal")

    def test_abnormal_measurement_takes_precedence_over_normal_reference(self) -> None:
        self.assertEqual(assess_indicator_value("乙肝表面抗原：阳性［参考值：阴性］"), "abnormal")
        self.assertEqual(assess_indicator_value("尿蛋白：2+［参考值：阴性］"), "abnormal")
        self.assertEqual(assess_indicator_value("尿蛋白：+［参考值：阴性］"), "abnormal")
        self.assertEqual(assess_indicator_value("Jones试验：阴性［参考值：阳性］"), "abnormal")

    def test_mixed_positive_and_negative_findings_are_abnormal(self) -> None:
        self.assertEqual(assess_indicator_value("抗体A阳性，抗体B阴性"), "abnormal")
        self.assertEqual(
            assess_indicator_value("痰液NAAT检出结核分枝杆菌；未检出利福平耐药突变"),
            "abnormal",
        )
        self.assertEqual(
            assess_indicator_value("食管内有食物潴留；存在食管炎；未见梗阻性肿块"),
            "abnormal",
        )

    def test_gram_negative_is_conservatively_abnormal(self) -> None:
        self.assertEqual(assess_indicator_value("革兰染色显示革兰阴性杆菌"), "abnormal")

    def test_exact_template_equivalence_is_normal(self) -> None:
        self.assertEqual(
            assess_indicator_value(
                "  未见异常。 ",
                normal_value="未见异常",
                has_reference=True,
            ),
            "normal",
        )

    def test_similar_definite_text_to_normal_template_is_normal(self) -> None:
        self.assertEqual(
            assess_indicator_value(
                "提睾肌张力减退，阴囊位置较低，皮肤皱褶减少",
                normal_value=(
                    "由于提睾筋膜平滑肌张力减退而变得更下垂（位置更低）；"
                    "皮肤皱褶可能减少。"
                ),
                has_reference=True,
            ),
            "normal",
        )

    def test_similar_text_with_opposite_direction_is_abnormal(self) -> None:
        cases = (
            ("皮肤皱褶明显增多", "皮肤皱褶减少"),
            ("检查发现明显肿块", "检查未发现明显肿块"),
        )
        for value, normal_value in cases:
            with self.subTest(value=value):
                self.assertEqual(
                    assess_indicator_value(
                        value,
                        normal_value=normal_value,
                        has_reference=True,
                    ),
                    "abnormal",
                )

    def test_unrecognized_text_is_conservatively_abnormal(self) -> None:
        self.assertEqual(assess_indicator_value("建议结合临床判断"), "abnormal")
        self.assertEqual(assess_indicator_value(""), "abnormal")

    def test_classifies_scalar_against_closed_reference_interval(self) -> None:
        self.assertEqual(
            assess_indicator_value(
                "血红蛋白 145 g/L",
                normal_value="130-175 g/L",
                has_reference=True,
            ),
            "normal",
        )
        self.assertEqual(
            assess_indicator_value(
                "血红蛋白 88 g/L",
                normal_value="130-175 g/L",
                has_reference=True,
            ),
            "abnormal",
        )

    def test_classifies_embedded_reference_interval(self) -> None:
        self.assertEqual(
            assess_indicator_value("AFP：1250 ng/mL［参考范围：<10 ng/mL］"),
            "abnormal",
        )
        self.assertEqual(
            assess_indicator_value("钾：4.2 mmol/L（参考范围 3.5-5.5 mmol/L）"),
            "normal",
        )
        self.assertEqual(
            assess_indicator_value("胆囊壁厚5.2 mm［参考范围：<3 mm］，伴胆结石"),
            "abnormal",
        )
        self.assertEqual(
            assess_indicator_value("DeMeester评分：42［参考：>14.7为异常］"),
            "abnormal",
        )
        self.assertEqual(
            assess_indicator_value("2.0 mm ST段压低［参考：阴性；无ST段压低>=1 mm］"),
            "abnormal",
        )

    def test_supports_one_sided_reference_bounds(self) -> None:
        self.assertEqual(assess_indicator_value("8 ng/mL", normal_value="<10 ng/mL", has_reference=True), "normal")
        self.assertEqual(assess_indicator_value("12 ng/mL", normal_value="<10 ng/mL", has_reference=True), "abnormal")
        self.assertEqual(assess_indicator_value("6 U/L", normal_value=">=5 U/L", has_reference=True), "normal")
        self.assertEqual(assess_indicator_value("<5 mIU/mL［参考范围：<5］，阴性"), "normal")

    def test_does_not_guess_when_units_or_numbers_are_ambiguous(self) -> None:
        self.assertEqual(
            assess_indicator_value("145 mg/L", normal_value="130-175 g/L", has_reference=True),
            "abnormal",
        )
        self.assertEqual(
            assess_indicator_value("嗜酸性粒细胞18%", normal_value="每低倍视野<25", has_reference=True),
            "abnormal",
        )
        self.assertEqual(
            assess_indicator_value("第一次 88 g/L，第二次 145 g/L", normal_value="130-175 g/L", has_reference=True),
            "abnormal",
        )


class ExaminationAssessmentTest(unittest.TestCase):
    def test_all_normal_fields_produce_normal_status(self) -> None:
        assessment = assess_examination_result(
            {"培养结果": "未分离出致病菌", "补充描述": "未见异常"},
            {"培养结果": "未分离出致病菌", "补充描述": "未见异常"},
        )

        self.assertEqual(assessment.status, "normal")
        self.assertEqual(assessment.abnormal_indicators, ())

    def test_all_binary_abnormal_fields_are_listed(self) -> None:
        assessment = assess_examination_result(
            {"甲胎蛋白": "1250 ng/mL", "备注": "建议结合临床判断"},
            {"甲胎蛋白": "<10 ng/mL", "备注": "无"},
        )

        self.assertEqual(assessment.status, "abnormal")
        self.assertEqual(assessment.abnormal_indicators, ("甲胎蛋白", "备注"))
        self.assertEqual(
            assessment.indicator_statuses,
            {"甲胎蛋白": "abnormal", "备注": "abnormal"},
        )

    def test_ambiguous_direct_result_is_binary_abnormal(self) -> None:
        assessment = assess_examination_result(
            {"结论": "建议结合临床判断"},
            {"结论": "未见异常"},
        )

        self.assertEqual(assessment.status, "abnormal")
        self.assertEqual(assessment.abnormal_indicators, ("结论",))
        self.assertEqual(assessment.indicator_statuses, {"结论": "abnormal"})


if __name__ == "__main__":
    unittest.main()

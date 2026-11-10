# MemAtlas Evaluation Report

## Comparison

| Method | Implementation | Mode | Benchmark | Predictions | overall_clinical_score | diagnosis_accuracy | examination_precision | treatment_overall_score | evaluation_coverage | Memory construction time (s) | Memory learning time (s) | Mean retrieval latency (ms/request) | Mean question response time (s/question) | Memory + answer model tokens (tokens) | Sampled peak process-tree RSS (MiB) | Live memory data storage (MiB) | Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| memos | upstream_native | standard | openhospital | 100 | 0.426391 | 0.49 | 0.370572 | 0.4186 | 1 | 34.7 | 4095.61 | 2255.36 | 135.22 | unknown | 3927.54 | 0.00022316 | 0 |

## Run 1: memos

Benchmark: `openhospital`  
Run ID: `a43ec5fba8324936bfdad1e2f5529e2c`  
Predictions: 100  
Failures: 0

### Quality Metrics

| Metric | Value |
| --- | --- |
| complete_clinical_coverage | True |
| diagnosis_accuracy | 0.49 |
| evaluated_cases | 100 |
| evaluation_coverage | 1 |
| examination_precision | 0.370572 |
| examination_recall | 0.170984 |
| expected_cases | 100 |
| failed_evaluations | 0 |
| metric_counts | {'diagnosis_accuracy': 100, 'examination_precision': 100, 'treatment_overall_score': 100, 'treatment_safety': 100, 'treatment_effectiveness_alignment': 100, 'treatment_personalization': 100} |
| metric_missing_counts | {'diagnosis_accuracy': 0, 'examination_precision': 0, 'treatment_overall_score': 0, 'treatment_safety': 0, 'treatment_effectiveness_alignment': 0, 'treatment_personalization': 0} |
| no_examination_rate | 0.2 |
| official_examination_precision | 0.463215 |
| overall_clinical_score | 0.426391 |
| pending_evaluations | 0 |
| per_department | {'血液科': {'diagnosis_accuracy': 0.49, 'examination_precision': 0.370572, 'treatment_overall_score': 0.4186, 'treatment_safety': 0.782, 'treatment_effectiveness_alignment': 0.434, 'treatment_personalization': 0.47000000000000003, 'evaluated_cases': 100, 'failed_evaluations': 0, 'pending_evaluations': 0, 'expected_cases': 100, 'evaluation_coverage': 1.0, 'metric_counts': {'diagnosis_accuracy': 100, 'examination_precision': 100, 'treatment_overall_score': 100, 'treatment_safety': 100, 'treatment_effectiveness_alignment': 100, 'treatment_personalization': 100}, 'no_examination_rate': 0.2, 'examination_recall': 0.170984126984127, 'official_examination_precision': 0.463215, 'overall_clinical_score': 0.4263906666666667, 'metric_missing_counts': {'diagnosis_accuracy': 0, 'examination_precision': 0, 'treatment_overall_score': 0, 'treatment_safety': 0, 'treatment_effectiveness_alignment': 0, 'treatment_personalization': 0}, 'complete_clinical_coverage': True}} |
| treatment_effectiveness_alignment | 0.434 |
| treatment_overall_score | 0.4186 |
| treatment_personalization | 0.47 |
| treatment_safety | 0.782 |

### Efficiency and Resource Measurements

| Dimension | Metric | Unit | Value |
| --- | --- | --- | --- |
| Model usage | Agent output tokens | tokens | unknown |
| Model usage | Agent embedding requests | requests | 0 |
| Model usage | Agent LLM requests | requests | 1004 |
| Model usage | Agent input tokens | tokens | unknown |
| Model usage | Memory + answer embedding requests | requests | 2793 |
| Model usage | Judge output tokens | tokens | 5965 |
| Model usage | Judge embedding requests | requests | 0 |
| Model usage | Judge LLM requests | requests | 100 |
| Model usage | Judge input tokens | tokens | 298530 |
| Time efficiency | Total scoring time | s | 0 |
| Model usage | Memory + answer LLM requests | requests | 2099 |
| Model usage | Memory output tokens | tokens | 471221 |
| Time efficiency | Memory construction time | s | 34.7 |
| Model usage | Memory embedding requests | requests | 2793 |
| Time efficiency | Memory learning time | s | 4095.61 |
| Model usage | Memory LLM requests | requests | 1095 |
| Model usage | Memory input tokens | tokens | 3096141 |
| Space overhead | Live memory data storage | MiB | 0.00022316 |
| Space overhead | Sampled peak process-tree RSS | MiB | 3927.54 |
| Time efficiency | Mean question response time | s/question | 135.22 |
| Time efficiency | Mean retrieval latency | ms/request | 2255.36 |
| Model usage | All recorded model tokens including judges | tokens | unknown |
| Model usage | Memory + answer model tokens | tokens | unknown |

### Run Diagnostics

| Field | Value |
| --- | --- |
| common.add_count | 100 |
| common.add_failures | 0 |
| common.agent_cache_reported_calls | 0 |
| common.agent_cache_reported_prompt_tokens | 0 |
| common.agent_cache_token_coverage | unknown |
| common.agent_cached_prompt_tokens | unknown |
| common.agent_calls | 1004 |
| common.agent_costed_calls | 0 |
| common.agent_embedding_completion_tokens | 0 |
| common.agent_embedding_prompt_tokens | 0 |
| common.agent_embedding_token_coverage | complete |
| common.agent_embedding_total_tokens | 0 |
| common.agent_estimated_calls | 0 |
| common.agent_failed_calls | 9 |
| common.agent_known_cache_creation_prompt_tokens | 0 |
| common.agent_known_cached_prompt_tokens | 0 |
| common.agent_known_completion_tokens | 169973 |
| common.agent_known_prompt_tokens | 1970148 |
| common.agent_known_total_tokens | 2140121 |
| common.agent_llm_completion_tokens | unknown |
| common.agent_llm_prompt_tokens | unknown |
| common.agent_llm_token_coverage | partial |
| common.agent_llm_total_tokens | unknown |
| common.agent_prompt_cache_hit_rate | unknown |
| common.agent_token_coverage | partial |
| common.agent_total_tokens | unknown |
| common.agent_tracked_calls | 1004 |
| common.agent_uncosted_calls | 1004 |
| common.agent_untracked_calls | 9 |
| common.cache_reported_calls | 0 |
| common.cache_reported_prompt_tokens | 0 |
| common.cache_token_coverage | unknown |
| common.cached_prompt_tokens | unknown |
| common.completion_tokens | unknown |
| common.costed_calls | 2793 |
| common.delete_count | 0 |
| common.delete_failures | 0 |
| common.episode_end_count | 100 |
| common.episode_end_failures | 0 |
| common.estimated_calls | 0 |
| common.failed_calls | 9 |
| common.flush_count | 7 |
| common.flush_failures | 0 |
| common.judge_cache_reported_calls | 0 |
| common.judge_cache_reported_prompt_tokens | 0 |
| common.judge_cache_token_coverage | unknown |
| common.judge_cached_prompt_tokens | unknown |
| common.judge_calls | 100 |
| common.judge_costed_calls | 0 |
| common.judge_embedding_completion_tokens | 0 |
| common.judge_embedding_prompt_tokens | 0 |
| common.judge_embedding_token_coverage | complete |
| common.judge_embedding_total_tokens | 0 |
| common.judge_estimated_calls | 0 |
| common.judge_failed_calls | 0 |
| common.judge_known_cache_creation_prompt_tokens | 0 |
| common.judge_known_cached_prompt_tokens | 0 |
| common.judge_known_completion_tokens | 5965 |
| common.judge_known_prompt_tokens | 298530 |
| common.judge_known_total_tokens | 304495 |
| common.judge_llm_completion_tokens | 5965 |
| common.judge_llm_prompt_tokens | 298530 |
| common.judge_llm_token_coverage | complete |
| common.judge_llm_total_tokens | 304495 |
| common.judge_prompt_cache_hit_rate | unknown |
| common.judge_token_coverage | complete |
| common.judge_total_tokens | 304495 |
| common.judge_tracked_calls | 100 |
| common.judge_uncosted_calls | 100 |
| common.judge_untracked_calls | 0 |
| common.known_cache_creation_prompt_tokens | 0 |
| common.known_cached_prompt_tokens | 0 |
| common.known_completion_tokens | 641194 |
| common.known_prompt_tokens | 5066289 |
| common.known_total_tokens | 5707483 |
| common.memory_cache_reported_calls | 0 |
| common.memory_cache_reported_prompt_tokens | 0 |
| common.memory_cache_token_coverage | unknown |
| common.memory_cached_prompt_tokens | unknown |
| common.memory_calls | 3888 |
| common.memory_costed_calls | 2793 |
| common.memory_embedding_completion_tokens | 0 |
| common.memory_embedding_prompt_tokens | 621274 |
| common.memory_embedding_token_coverage | complete |
| common.memory_embedding_total_tokens | 621274 |
| common.memory_estimated_calls | 0 |
| common.memory_failed_calls | 0 |
| common.memory_known_cache_creation_prompt_tokens | 0 |
| common.memory_known_cached_prompt_tokens | 0 |
| common.memory_known_completion_tokens | 471221 |
| common.memory_known_prompt_tokens | 3096141 |
| common.memory_known_total_tokens | 3567362 |
| common.memory_llm_completion_tokens | 471221 |
| common.memory_llm_prompt_tokens | 2474867 |
| common.memory_llm_token_coverage | complete |
| common.memory_llm_total_tokens | 2946088 |
| common.memory_prompt_cache_hit_rate | unknown |
| common.memory_token_coverage | complete |
| common.memory_total_tokens | 3567362 |
| common.memory_tracked_calls | 3888 |
| common.memory_uncosted_calls | 1095 |
| common.memory_untracked_calls | 0 |
| common.process_baseline_rss_bytes | 75128832 |
| common.process_memory_monitor_error | unknown |
| common.process_memory_sample_count | 43123 |
| common.process_memory_scope | current_process_and_descendants_excludes_external_services |
| common.process_peak_rss_bytes | 4118319104 |
| common.process_peak_rss_delta_bytes | 4043190272 |
| common.prompt_cache_hit_rate | unknown |
| common.prompt_tokens | unknown |
| common.reset_count | 0 |
| common.reset_failures | 0 |
| common.response_count | 100 |
| common.response_time_s | 13522 |
| common.response_timing_coverage | 1 |
| common.retrieval_count | 677 |
| common.retrieval_failures | 0 |
| common.retrieval_max_ms | 13643.2 |
| common.retrieval_median_ms | 1369.04 |
| common.retrieval_time_ms | 1.52688e+06 |
| common.run_wall_time_ms | 5.9129e+06 |
| common.storage_bytes | 234 |
| common.storage_scope | live_method_data_excludes_logs_and_recovery_snapshots |
| common.token_coverage | partial |
| common.tracked_calls | 4892 |
| common.uncosted_calls | 2099 |
| common.untracked_calls | 9 |
| common.update_count | 0 |
| common.update_failures | 0 |
| current_process_usage | {'tracked_calls': 388, 'cached_prompt_tokens': None, 'known_cached_prompt_tokens': 0, 'known_cache_creation_prompt_tokens': 0, 'cache_reported_calls': 0, 'cache_reported_prompt_tokens': 0, 'cache_token_coverage': 'unknown', 'prompt_cache_hit_rate': None, 'llm_calls': 282, 'embedding_calls': 106, 'prompt_tokens': 541042, 'completion_tokens': 38166, 'total_tokens': None, 'known_total_tokens': 579208, 'costed_calls': 106, 'uncosted_calls': 282, 'untracked_calls': 1, 'token_coverage': 'partial', 'estimated_calls': 0, 'failed_calls': 1} |

# MemAtlas Evaluation Report

## Comparison

| Method | Implementation | Mode | Benchmark | Predictions | overall_clinical_score | diagnosis_accuracy | examination_precision | treatment_overall_score | evaluation_coverage | Memory construction time (s) | Memory learning time (s) | Mean retrieval latency (ms/request) | Mean question response time (s/question) | Memory + answer model tokens (tokens) | Sampled peak process-tree RSS (MiB) | Live memory data storage (MiB) | Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| memmachine | upstream_native | standard | openhospital | 100 | 0.522599 | 0.64 | 0.444396 | 0.4834 | 1 | 26.1613 | 5020.14 | 3532.85 | 150.572 | 53683841 | 3178.31 | 31.962 | 0 |

## Run 1: memmachine

Benchmark: `openhospital`  
Run ID: `6ccece932fd345e3b22111ddf7523d2b`  
Predictions: 100  
Failures: 0

### Quality Metrics

| Metric | Value |
| --- | --- |
| complete_clinical_coverage | True |
| diagnosis_accuracy | 0.64 |
| evaluated_cases | 100 |
| evaluation_coverage | 1 |
| examination_precision | 0.444396 |
| examination_recall | 0.246016 |
| expected_cases | 100 |
| failed_evaluations | 0 |
| metric_counts | {'diagnosis_accuracy': 100, 'examination_precision': 100, 'treatment_overall_score': 100, 'treatment_safety': 100, 'treatment_effectiveness_alignment': 100, 'treatment_personalization': 100} |
| metric_missing_counts | {'diagnosis_accuracy': 0, 'examination_precision': 0, 'treatment_overall_score': 0, 'treatment_safety': 0, 'treatment_effectiveness_alignment': 0, 'treatment_personalization': 0} |
| no_examination_rate | 0.09 |
| official_examination_precision | 0.488347 |
| overall_clinical_score | 0.522599 |
| pending_evaluations | 0 |
| per_department | {'血液科': {'diagnosis_accuracy': 0.64, 'examination_precision': 0.444396, 'treatment_overall_score': 0.4834, 'treatment_safety': 0.826, 'treatment_effectiveness_alignment': 0.524, 'treatment_personalization': 0.528, 'evaluated_cases': 100, 'failed_evaluations': 0, 'pending_evaluations': 0, 'expected_cases': 100, 'evaluation_coverage': 1.0, 'metric_counts': {'diagnosis_accuracy': 100, 'examination_precision': 100, 'treatment_overall_score': 100, 'treatment_safety': 100, 'treatment_effectiveness_alignment': 100, 'treatment_personalization': 100}, 'no_examination_rate': 0.09, 'examination_recall': 0.24601587301587302, 'official_examination_precision': 0.4883472527472527, 'overall_clinical_score': 0.5225986666666667, 'metric_missing_counts': {'diagnosis_accuracy': 0, 'examination_precision': 0, 'treatment_overall_score': 0, 'treatment_safety': 0, 'treatment_effectiveness_alignment': 0, 'treatment_personalization': 0}, 'complete_clinical_coverage': True}} |
| treatment_effectiveness_alignment | 0.524 |
| treatment_overall_score | 0.4834 |
| treatment_personalization | 0.528 |
| treatment_safety | 0.826 |

### Efficiency and Resource Measurements

| Dimension | Metric | Unit | Value |
| --- | --- | --- | --- |
| Model usage | Agent output tokens | tokens | 249612 |
| Model usage | Agent embedding requests | requests | 0 |
| Model usage | Agent LLM requests | requests | 1162 |
| Model usage | Agent input tokens | tokens | 24813034 |
| Model usage | Memory + answer embedding requests | requests | 6055 |
| Model usage | Judge output tokens | tokens | 6103 |
| Model usage | Judge embedding requests | requests | 0 |
| Model usage | Judge LLM requests | requests | 100 |
| Model usage | Judge input tokens | tokens | 319654 |
| Time efficiency | Total scoring time | s | 0 |
| Model usage | Memory + answer LLM requests | requests | 7352 |
| Model usage | Memory output tokens | tokens | 698416 |
| Time efficiency | Memory construction time | s | 26.1613 |
| Model usage | Memory embedding requests | requests | 6055 |
| Time efficiency | Memory learning time | s | 5020.14 |
| Model usage | Memory LLM requests | requests | 6190 |
| Model usage | Memory input tokens | tokens | 27922779 |
| Space overhead | Live memory data storage | MiB | 31.962 |
| Space overhead | Sampled peak process-tree RSS | MiB | 3178.31 |
| Time efficiency | Mean question response time | s/question | 150.572 |
| Time efficiency | Mean retrieval latency | ms/request | 3532.85 |
| Model usage | All recorded model tokens including judges | tokens | 55357065 |
| Model usage | Memory + answer model tokens | tokens | 53683841 |

### Run Diagnostics

| Field | Value |
| --- | --- |
| common.add_count | 100 |
| common.add_failures | 0 |
| common.agent_cache_reported_calls | 1047 |
| common.agent_cache_reported_prompt_tokens | 24591913 |
| common.agent_cache_token_coverage | partial |
| common.agent_cached_prompt_tokens | unknown |
| common.agent_calls | 1162 |
| common.agent_costed_calls | 0 |
| common.agent_embedding_completion_tokens | 0 |
| common.agent_embedding_prompt_tokens | 0 |
| common.agent_embedding_token_coverage | complete |
| common.agent_embedding_total_tokens | 0 |
| common.agent_estimated_calls | 0 |
| common.agent_failed_calls | 0 |
| common.agent_known_cache_creation_prompt_tokens | 23482552 |
| common.agent_known_cached_prompt_tokens | 12286 |
| common.agent_known_completion_tokens | 249612 |
| common.agent_known_prompt_tokens | 24813034 |
| common.agent_known_total_tokens | 25062646 |
| common.agent_llm_completion_tokens | 249612 |
| common.agent_llm_prompt_tokens | 24813034 |
| common.agent_llm_token_coverage | complete |
| common.agent_llm_total_tokens | 25062646 |
| common.agent_prompt_cache_hit_rate | 0.000499595 |
| common.agent_token_coverage | complete |
| common.agent_total_tokens | 25062646 |
| common.agent_tracked_calls | 1162 |
| common.agent_uncosted_calls | 1162 |
| common.agent_untracked_calls | 0 |
| common.cache_reported_calls | 7209 |
| common.cache_reported_prompt_tokens | 49347381 |
| common.cache_token_coverage | partial |
| common.cached_prompt_tokens | unknown |
| common.completion_tokens | 948028 |
| common.costed_calls | 6055 |
| common.delete_count | 0 |
| common.delete_failures | 0 |
| common.episode_end_count | 100 |
| common.episode_end_failures | 0 |
| common.estimated_calls | 0 |
| common.failed_calls | 0 |
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
| common.judge_known_completion_tokens | 6103 |
| common.judge_known_prompt_tokens | 319654 |
| common.judge_known_total_tokens | 325757 |
| common.judge_llm_completion_tokens | 6103 |
| common.judge_llm_prompt_tokens | 319654 |
| common.judge_llm_token_coverage | complete |
| common.judge_llm_total_tokens | 325757 |
| common.judge_prompt_cache_hit_rate | unknown |
| common.judge_token_coverage | complete |
| common.judge_total_tokens | 325757 |
| common.judge_tracked_calls | 100 |
| common.judge_uncosted_calls | 100 |
| common.judge_untracked_calls | 0 |
| common.known_cache_creation_prompt_tokens | 23522968 |
| common.known_cached_prompt_tokens | 15874770 |
| common.known_completion_tokens | 948028 |
| common.known_prompt_tokens | 52735813 |
| common.known_total_tokens | 53683841 |
| common.memory_cache_reported_calls | 6162 |
| common.memory_cache_reported_prompt_tokens | 24755468 |
| common.memory_cache_token_coverage | partial |
| common.memory_cached_prompt_tokens | unknown |
| common.memory_calls | 12245 |
| common.memory_costed_calls | 6055 |
| common.memory_embedding_completion_tokens | 0 |
| common.memory_embedding_prompt_tokens | 2107174 |
| common.memory_embedding_token_coverage | complete |
| common.memory_embedding_total_tokens | 2107174 |
| common.memory_estimated_calls | 0 |
| common.memory_failed_calls | 0 |
| common.memory_known_cache_creation_prompt_tokens | 40416 |
| common.memory_known_cached_prompt_tokens | 15862484 |
| common.memory_known_completion_tokens | 698416 |
| common.memory_known_prompt_tokens | 27922779 |
| common.memory_known_total_tokens | 28621195 |
| common.memory_llm_completion_tokens | 698416 |
| common.memory_llm_prompt_tokens | 25815605 |
| common.memory_llm_token_coverage | complete |
| common.memory_llm_total_tokens | 26514021 |
| common.memory_prompt_cache_hit_rate | 0.640767 |
| common.memory_token_coverage | complete |
| common.memory_total_tokens | 28621195 |
| common.memory_tracked_calls | 12245 |
| common.memory_uncosted_calls | 6190 |
| common.memory_untracked_calls | 0 |
| common.process_baseline_rss_bytes | 97931264 |
| common.process_memory_monitor_error | unknown |
| common.process_memory_sample_count | 26294 |
| common.process_memory_scope | current_process_and_descendants_excludes_external_services |
| common.process_peak_rss_bytes | 3332702208 |
| common.process_peak_rss_delta_bytes | 3234770944 |
| common.prompt_cache_hit_rate | 0.321694 |
| common.prompt_tokens | 52735813 |
| common.reset_count | 0 |
| common.reset_failures | 0 |
| common.response_count | 100 |
| common.response_time_s | 15057.2 |
| common.response_timing_coverage | 1 |
| common.retrieval_count | 838 |
| common.retrieval_failures | 0 |
| common.retrieval_max_ms | 14211.6 |
| common.retrieval_median_ms | 2719.62 |
| common.retrieval_time_ms | 2.96053e+06 |
| common.run_wall_time_ms | 3.02402e+06 |
| common.storage_bytes | 33514549 |
| common.storage_scope | live_method_data_excludes_logs_and_recovery_snapshots |
| common.token_coverage | complete |
| common.tracked_calls | 13407 |
| common.uncosted_calls | 7352 |
| common.untracked_calls | 0 |
| common.update_count | 0 |
| common.update_failures | 0 |
| current_process_usage | {'tracked_calls': 1638, 'cached_prompt_tokens': None, 'known_cached_prompt_tokens': 1735960, 'known_cache_creation_prompt_tokens': 2914086, 'cache_reported_calls': 798, 'cache_reported_prompt_tokens': 6015188, 'cache_token_coverage': 'partial', 'prompt_cache_hit_rate': 0.288596133653678, 'llm_calls': 937, 'embedding_calls': 701, 'prompt_tokens': 6561684, 'completion_tokens': 107718, 'total_tokens': 6669402, 'known_total_tokens': 6669402, 'costed_calls': 701, 'uncosted_calls': 937, 'untracked_calls': 0, 'token_coverage': 'complete', 'estimated_calls': 0, 'failed_calls': 0} |

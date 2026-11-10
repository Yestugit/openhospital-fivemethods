# MemAtlas Evaluation Report

## Comparison

| Method | Implementation | Mode | Benchmark | Predictions | overall_clinical_score | diagnosis_accuracy | examination_precision | treatment_overall_score | evaluation_coverage | Memory construction time (s) | Memory learning time (s) | Mean retrieval latency (ms/request) | Mean question response time (s/question) | Memory + answer model tokens (tokens) | Sampled peak process-tree RSS (MiB) | Live memory data storage (MiB) | Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mem0 | native_wrapper | standard | openhospital | 100 | 0.562381 | 0.7 | 0.470543 | 0.5166 | 1 | 12.2621 | 5889 | 3565.36 | 185.099 | unknown | 1340.04 | 29.9583 | 0 |

## Run 1: mem0

Benchmark: `openhospital`  
Run ID: `2a69a056bbea4d4db2c5e4a7c1434205`  
Predictions: 100  
Failures: 0

### Quality Metrics

| Metric | Value |
| --- | --- |
| complete_clinical_coverage | True |
| diagnosis_accuracy | 0.7 |
| evaluated_cases | 100 |
| evaluation_coverage | 1 |
| examination_precision | 0.470543 |
| examination_recall | 0.29346 |
| expected_cases | 100 |
| failed_evaluations | 0 |
| metric_counts | {'diagnosis_accuracy': 100, 'examination_precision': 100, 'treatment_overall_score': 100, 'treatment_safety': 100, 'treatment_effectiveness_alignment': 100, 'treatment_personalization': 100} |
| metric_missing_counts | {'diagnosis_accuracy': 0, 'examination_precision': 0, 'treatment_overall_score': 0, 'treatment_safety': 0, 'treatment_effectiveness_alignment': 0, 'treatment_personalization': 0} |
| no_examination_rate | 0.07 |
| official_examination_precision | 0.50596 |
| overall_clinical_score | 0.562381 |
| pending_evaluations | 0 |
| per_department | {'血液科': {'diagnosis_accuracy': 0.7, 'examination_precision': 0.470543, 'treatment_overall_score': 0.5166000000000001, 'treatment_safety': 0.84, 'treatment_effectiveness_alignment': 0.55, 'treatment_personalization': 0.56, 'evaluated_cases': 100, 'failed_evaluations': 0, 'pending_evaluations': 0, 'expected_cases': 100, 'evaluation_coverage': 1.0, 'metric_counts': {'diagnosis_accuracy': 100, 'examination_precision': 100, 'treatment_overall_score': 100, 'treatment_safety': 100, 'treatment_effectiveness_alignment': 100, 'treatment_personalization': 100}, 'no_examination_rate': 0.07, 'examination_recall': 0.2934603174603175, 'official_examination_precision': 0.5059602150537634, 'overall_clinical_score': 0.562381, 'metric_missing_counts': {'diagnosis_accuracy': 0, 'examination_precision': 0, 'treatment_overall_score': 0, 'treatment_safety': 0, 'treatment_effectiveness_alignment': 0, 'treatment_personalization': 0}, 'complete_clinical_coverage': True}} |
| treatment_effectiveness_alignment | 0.55 |
| treatment_overall_score | 0.5166 |
| treatment_personalization | 0.56 |
| treatment_safety | 0.84 |

### Efficiency and Resource Measurements

| Dimension | Metric | Unit | Value |
| --- | --- | --- | --- |
| Model usage | Agent output tokens | tokens | unknown |
| Model usage | Agent embedding requests | requests | 0 |
| Model usage | Agent LLM requests | requests | 1133 |
| Model usage | Agent input tokens | tokens | unknown |
| Model usage | Memory + answer embedding requests | requests | 1898 |
| Model usage | Judge output tokens | tokens | 6089 |
| Model usage | Judge embedding requests | requests | 0 |
| Model usage | Judge LLM requests | requests | 100 |
| Model usage | Judge input tokens | tokens | 299558 |
| Time efficiency | Total scoring time | s | 0 |
| Model usage | Memory + answer LLM requests | requests | 1233 |
| Model usage | Memory output tokens | tokens | 66405 |
| Time efficiency | Memory construction time | s | 12.2621 |
| Model usage | Memory embedding requests | requests | 1898 |
| Time efficiency | Memory learning time | s | 5889 |
| Model usage | Memory LLM requests | requests | 100 |
| Model usage | Memory input tokens | tokens | 2805514 |
| Space overhead | Live memory data storage | MiB | 29.9583 |
| Space overhead | Sampled peak process-tree RSS | MiB | 1340.04 |
| Time efficiency | Mean question response time | s/question | 185.099 |
| Time efficiency | Mean retrieval latency | ms/request | 3565.36 |
| Model usage | All recorded model tokens including judges | tokens | unknown |
| Model usage | Memory + answer model tokens | tokens | unknown |

### Run Diagnostics

| Field | Value |
| --- | --- |
| common.add_count | 100 |
| common.add_failures | 0 |
| common.agent_cache_reported_calls | 513 |
| common.agent_cache_reported_prompt_tokens | 1168189 |
| common.agent_cache_token_coverage | partial |
| common.agent_cached_prompt_tokens | unknown |
| common.agent_calls | 1133 |
| common.agent_costed_calls | 0 |
| common.agent_embedding_completion_tokens | 0 |
| common.agent_embedding_prompt_tokens | 0 |
| common.agent_embedding_token_coverage | complete |
| common.agent_embedding_total_tokens | 0 |
| common.agent_estimated_calls | 0 |
| common.agent_failed_calls | 1 |
| common.agent_known_cache_creation_prompt_tokens | 651730 |
| common.agent_known_cached_prompt_tokens | 33573 |
| common.agent_known_completion_tokens | 180053 |
| common.agent_known_prompt_tokens | 2617827 |
| common.agent_known_total_tokens | 2797880 |
| common.agent_llm_completion_tokens | unknown |
| common.agent_llm_prompt_tokens | unknown |
| common.agent_llm_token_coverage | partial |
| common.agent_llm_total_tokens | unknown |
| common.agent_prompt_cache_hit_rate | 0.0287394 |
| common.agent_token_coverage | partial |
| common.agent_total_tokens | unknown |
| common.agent_tracked_calls | 1133 |
| common.agent_uncosted_calls | 1133 |
| common.agent_untracked_calls | 1 |
| common.cache_reported_calls | 613 |
| common.cache_reported_prompt_tokens | 2717932 |
| common.cache_token_coverage | partial |
| common.cached_prompt_tokens | unknown |
| common.completion_tokens | unknown |
| common.costed_calls | 1898 |
| common.delete_count | 0 |
| common.delete_failures | 0 |
| common.episode_end_count | 100 |
| common.episode_end_failures | 0 |
| common.estimated_calls | 0 |
| common.failed_calls | 1 |
| common.flush_count | 6 |
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
| common.judge_known_completion_tokens | 6089 |
| common.judge_known_prompt_tokens | 299558 |
| common.judge_known_total_tokens | 305647 |
| common.judge_llm_completion_tokens | 6089 |
| common.judge_llm_prompt_tokens | 299558 |
| common.judge_llm_token_coverage | complete |
| common.judge_llm_total_tokens | 305647 |
| common.judge_prompt_cache_hit_rate | unknown |
| common.judge_token_coverage | complete |
| common.judge_total_tokens | 305647 |
| common.judge_tracked_calls | 100 |
| common.judge_uncosted_calls | 100 |
| common.judge_untracked_calls | 0 |
| common.known_cache_creation_prompt_tokens | 814490 |
| common.known_cached_prompt_tokens | 684613 |
| common.known_completion_tokens | 246458 |
| common.known_prompt_tokens | 5423341 |
| common.known_total_tokens | 5669799 |
| common.memory_cache_reported_calls | 100 |
| common.memory_cache_reported_prompt_tokens | 1549743 |
| common.memory_cache_token_coverage | complete |
| common.memory_cached_prompt_tokens | 651040 |
| common.memory_calls | 1998 |
| common.memory_costed_calls | 1898 |
| common.memory_embedding_completion_tokens | 0 |
| common.memory_embedding_prompt_tokens | 1255771 |
| common.memory_embedding_token_coverage | complete |
| common.memory_embedding_total_tokens | 1255771 |
| common.memory_estimated_calls | 0 |
| common.memory_failed_calls | 0 |
| common.memory_known_cache_creation_prompt_tokens | 162760 |
| common.memory_known_cached_prompt_tokens | 651040 |
| common.memory_known_completion_tokens | 66405 |
| common.memory_known_prompt_tokens | 2805514 |
| common.memory_known_total_tokens | 2871919 |
| common.memory_llm_completion_tokens | 66405 |
| common.memory_llm_prompt_tokens | 1549743 |
| common.memory_llm_token_coverage | complete |
| common.memory_llm_total_tokens | 1616148 |
| common.memory_prompt_cache_hit_rate | 0.420095 |
| common.memory_token_coverage | complete |
| common.memory_total_tokens | 2871919 |
| common.memory_tracked_calls | 1998 |
| common.memory_uncosted_calls | 100 |
| common.memory_untracked_calls | 0 |
| common.process_baseline_rss_bytes | 51638272 |
| common.process_memory_monitor_error | unknown |
| common.process_memory_sample_count | 306805 |
| common.process_memory_scope | current_process_and_descendants_excludes_external_services |
| common.process_peak_rss_bytes | 1405132800 |
| common.process_peak_rss_delta_bytes | 1353494528 |
| common.prompt_cache_hit_rate | 0.251887 |
| common.prompt_tokens | unknown |
| common.reset_count | 0 |
| common.reset_failures | 0 |
| common.response_count | 100 |
| common.response_time_s | 18509.9 |
| common.response_timing_coverage | 1 |
| common.retrieval_count | 758 |
| common.retrieval_failures | 0 |
| common.retrieval_max_ms | 17729.2 |
| common.retrieval_median_ms | 2541.82 |
| common.retrieval_time_ms | 2.70254e+06 |
| common.run_wall_time_ms | 2.3098e+07 |
| common.storage_bytes | 31413511 |
| common.storage_scope | live_method_data_excludes_logs_and_recovery_snapshots |
| common.token_coverage | partial |
| common.tracked_calls | 3131 |
| common.uncosted_calls | 1233 |
| common.untracked_calls | 1 |
| common.update_count | 0 |
| common.update_failures | 0 |
| current_process_usage | {'tracked_calls': 3922, 'cached_prompt_tokens': None, 'known_cached_prompt_tokens': 684613, 'known_cache_creation_prompt_tokens': 814490, 'cache_reported_calls': 613, 'cache_reported_prompt_tokens': 2717932, 'cache_token_coverage': 'partial', 'prompt_cache_hit_rate': 0.2518874644398756, 'llm_calls': 2024, 'embedding_calls': 1898, 'prompt_tokens': 6632891, 'completion_tokens': 366923, 'total_tokens': None, 'known_total_tokens': 6999814, 'costed_calls': 1898, 'uncosted_calls': 2024, 'untracked_calls': 1, 'token_coverage': 'partial', 'estimated_calls': 0, 'failed_calls': 1} |

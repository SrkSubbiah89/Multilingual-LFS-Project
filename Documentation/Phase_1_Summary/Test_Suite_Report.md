# Test Suite Report — Multilingual LFS Conversational AI

**M.Tech Thesis | IIIT Kottayam 2026 | ILO ICLS-19 Standard**

---

## Summary

| Metric | Value |
|---|---|
| Total tests | **1178** |
| Passed | **1178** |
| Failed | **0** |
| Test files | **25** |
| Deselected (slow-marked) | **1** — `backend/tests/load_test.py` (`@pytest.mark.slow`, excluded by `pytest.ini` default; run explicitly with `pytest -m slow`) |
| Run date | 2026-08-02 |
| Python | 3.11 |
| Framework | pytest |
| Infrastructure | SQLite in-memory (DB), FakeRedis, MagicMock (Qdrant / LLM / CrewAI) |

---

## Tests by Module

| # | Test File | Component | Tests |
|---|---|---|---|
| 1 | `test_audit_logger.py` | AuditLogger | 77 |
| 2 | `test_auth_and_api_extended.py` | Auth & API (extended edge cases) | 27 |
| 3 | `test_auth_routes.py` | Authentication Routes | 22 |
| 4 | `test_context_memory.py` | ContextMemory | 83 |
| 5 | `test_conversation_manager.py` | ConversationManager | 63 |
| 6 | `test_conversation_manager_extended.py` | ConversationManager (extended — full questionnaire, skip logic) | 42 |
| 7 | `test_emotional_intelligence.py` | EmotionalIntelligence | 74 |
| 8 | `test_evaluation.py` | Evaluation Framework (BM25 / Flat / Hierarchical) | 16 |
| 9 | `test_hitl_and_e2e_extended.py` | HITL & End-to-End (extended) | 37 |
| 10 | `test_hitl_quality_manager.py` | HITLQualityManager | 81 |
| 11 | `test_isced_classifier.py` | ISCEDClassifier (ISCED 2011 + ISCED-F 2013) | 17 |
| 12 | `test_isco_classifier.py` | ISCOClassifier | 31 |
| 13 | `test_isco_classifier_extended.py` | ISCOClassifier (extended — hierarchical RAG, keyword pre-filter) | 80 |
| 14 | `test_isic_classifier.py` | ISICClassifier (ISIC Rev.4) | 19 |
| 15 | `test_language_processor.py` | LanguageProcessor | 52 |
| 16 | `test_language_processor_extended.py` | LanguageProcessor (extended — Gulf Arabic, Devanagari, code-switch) | 47 |
| 17 | `test_nationality_classifier.py` | NationalityClassifier | 45 |
| 18 | `test_person_register.py` | PersonRegister | 10 |
| 19 | `test_rag_expert.py` | RAGExpert | 46 |
| 20 | `test_report_generator.py` | ReportGenerator | 45 |
| 21 | `test_survey_orchestrator.py` | SurveyOrchestrator | 65 |
| 22 | `test_survey_routes.py` | Survey Routes (API) | 29 |
| 23 | `test_validation_agent.py` | ValidationAgent | 82 |
| 24 | `test_validation_agent_extended.py` | ValidationAgent (extended) | 59 |
| 25 | `test_vector_store.py` | VectorStore | 29 |
| | **TOTAL** | | **1178** |

---

## AuditLogger
**File:** `backend/tests/test_audit_logger.py` &nbsp;|&nbsp; **Tests: 77**

### TestAccessTypeConstants (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_anonymize` | PASS |
| 2 | `test_delete` | PASS |
| 3 | `test_export` | PASS |
| 4 | `test_read` | PASS |
| 5 | `test_write` | PASS |

### TestAgentDecisionTypeConstants (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_isco_classification` | PASS |
| 2 | `test_ner_extraction` | PASS |
| 3 | `test_validation` | PASS |

### TestBuildFallbackReport (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_gdpr_article_mentioned_in_ar` | PASS |
| 2 | `test_gdpr_article_mentioned_in_en` | PASS |
| 3 | `test_no_activity_ar` | PASS |
| 4 | `test_no_activity_en` | PASS |
| 5 | `test_returns_tuple_of_two_strings` | PASS |
| 6 | `test_with_activity_ar_includes_counts` | PASS |
| 7 | `test_with_activity_en_includes_counts` | PASS |

### TestEventTypeConstants (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_auth_otp_requested` | PASS |
| 2 | `test_message_sent` | PASS |
| 3 | `test_purge_executed` | PASS |
| 4 | `test_session_completed` | PASS |
| 5 | `test_session_started` | PASS |

### TestGenerateReport (10 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_counts_agent_decisions` | PASS |
| 2 | `test_counts_data_accesses` | PASS |
| 3 | `test_counts_interactions` | PASS |
| 4 | `test_custom_period_respected` | PASS |
| 5 | `test_default_period_is_30_days` | PASS |
| 6 | `test_event_counts_dict` | PASS |
| 7 | `test_generated_at_is_iso` | PASS |
| 8 | `test_llm_report_text_propagated` | PASS |
| 9 | `test_returns_audit_report` | PASS |
| 10 | `test_session_filter` | PASS |

### TestGetAuditLogger (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_returns_audit_logger` | PASS |
| 2 | `test_returns_same_instance_on_second_call` | PASS |
| 3 | `test_reuses_existing_instance` | PASS |

### TestGetUserDataAccesses (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_does_not_return_other_users_entries` | PASS |
| 2 | `test_empty_for_unknown_user` | PASS |
| 3 | `test_ordered_newest_first` | PASS |
| 4 | `test_returns_entries_for_user` | PASS |
| 5 | `test_returns_list` | PASS |

### TestLogAgentDecision (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_id_assigned` | PASS |
| 2 | `test_persists_to_db` | PASS |
| 3 | `test_returns_agent_decision_entry` | PASS |
| 4 | `test_stores_optional_fields` | PASS |
| 5 | `test_stores_required_fields` | PASS |
| 6 | `test_timestamp_is_iso_string` | PASS |

### TestLogDataAccess (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_persists_to_db` | PASS |
| 2 | `test_retained_until_is_future` | PASS |
| 3 | `test_retained_until_respects_retention_days` | PASS |
| 4 | `test_returns_data_access_entry` | PASS |
| 5 | `test_stores_accessor_id` | PASS |
| 6 | `test_stores_optional_resource_id` | PASS |
| 7 | `test_stores_required_fields` | PASS |

### TestLogInteraction (9 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_id_is_positive_integer` | PASS |
| 2 | `test_no_metadata_stores_none` | PASS |
| 3 | `test_persists_to_db` | PASS |
| 4 | `test_returns_audit_entry` | PASS |
| 5 | `test_serialises_metadata_as_json` | PASS |
| 6 | `test_stores_description` | PASS |
| 7 | `test_stores_event_type` | PASS |
| 8 | `test_stores_optional_fields` | PASS |
| 9 | `test_timestamp_is_iso_string` | PASS |

### TestParseReportResponse (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_empty_string_triggers_fallback` | PASS |
| 2 | `test_invalid_json_triggers_fallback` | PASS |
| 3 | `test_json_with_markdown_fences` | PASS |
| 4 | `test_missing_en_triggers_fallback` | PASS |
| 5 | `test_regex_fallback_on_surrounding_text` | PASS |
| 6 | `test_valid_json` | PASS |

### TestPurgeExpired (11 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_cutoff_timestamp_is_iso` | PASS |
| 2 | `test_deletes_expired_access_logs` | PASS |
| 3 | `test_deletes_old_audit_logs` | PASS |
| 4 | `test_deletes_old_decision_logs` | PASS |
| 5 | `test_dry_run_does_not_delete` | PASS |
| 6 | `test_dry_run_flag_in_result` | PASS |
| 7 | `test_not_dry_run_flag_in_result` | PASS |
| 8 | `test_purge_timestamp_is_iso` | PASS |
| 9 | `test_recent_records_not_deleted` | PASS |
| 10 | `test_retention_days_in_result` | PASS |
| 11 | `test_returns_purge_result` | PASS |

---

## Auth & API (extended edge cases)
**File:** `backend/tests/test_auth_and_api_extended.py` &nbsp;|&nbsp; **Tests: 27**

### TestAllEndpointsCovered (8 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_create_session_wrong_language_code` | PASS |
| 2 | `test_delete_another_users_session` | PASS |
| 3 | `test_docs_accessible` | PASS |
| 4 | `test_get_nonexistent_session_404` | PASS |
| 5 | `test_health_check_no_auth` | PASS |
| 6 | `test_hitl_queue_requires_auth` | PASS |
| 7 | `test_report_endpoint_requires_auth` | PASS |
| 8 | `test_send_message_to_completed_session_rejected` | PASS |

### TestInputSanitisation (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_rtl_message_accepted` | PASS |
| 2 | `test_empty_message_handled` | PASS |
| 3 | `test_sql_injection_in_email` | PASS |
| 4 | `test_very_long_message_handled` | PASS |
| 5 | `test_xss_in_message` | PASS |

### TestJWTEdgeCases (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_bearer_prefix_required` | PASS |
| 2 | `test_expired_jwt_rejected` | PASS |
| 3 | `test_jwt_with_wrong_secret_rejected` | PASS |
| 4 | `test_malformed_jwt_rejected` | PASS |
| 5 | `test_no_auth_header_returns_403` | PASS |

### TestOTPEdgeCases (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_new_otp_request_invalidates_old` | PASS |
| 2 | `test_otp_exactly_at_expiry_rejected` | PASS |
| 3 | `test_otp_non_numeric_rejected` | PASS |
| 4 | `test_otp_reuse_rejected` | PASS |
| 5 | `test_otp_with_extra_whitespace_rejected` | PASS |
| 6 | `test_otp_wrong_by_one_digit_rejected` | PASS |

### TestResponseFormats (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_error_response_has_detail` | PASS |
| 2 | `test_message_response_has_all_fields` | PASS |
| 3 | `test_session_response_has_id` | PASS |

---

## Authentication Routes
**File:** `backend/tests/test_auth_routes.py` &nbsp;|&nbsp; **Tests: 22**

### TestLogout (2 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_returns_200` | PASS |
| 2 | `test_returns_correct_message` | PASS |

### TestRequestOtp (9 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_creates_new_user_on_first_request` | PASS |
| 2 | `test_deactivated_user_returns_403` | PASS |
| 3 | `test_does_not_duplicate_existing_user` | PASS |
| 4 | `test_email_send_failure_returns_502` | PASS |
| 5 | `test_invalid_email_returns_422` | PASS |
| 6 | `test_invalidates_previous_otp_on_new_request` | PASS |
| 7 | `test_missing_email_returns_422` | PASS |
| 8 | `test_returns_success_message` | PASS |
| 9 | `test_stores_otp_in_db` | PASS |

### TestVerifyOtp (11 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_already_used_code_returns_401` | PASS |
| 2 | `test_deactivated_user_returns_403` | PASS |
| 3 | `test_expired_code_returns_401` | PASS |
| 4 | `test_invalid_email_format_returns_422` | PASS |
| 5 | `test_jwt_subject_matches_user_id` | PASS |
| 6 | `test_missing_code_returns_422` | PASS |
| 7 | `test_missing_email_returns_422` | PASS |
| 8 | `test_unknown_email_returns_404` | PASS |
| 9 | `test_valid_otp_is_marked_used` | PASS |
| 10 | `test_valid_otp_returns_200_with_jwt` | PASS |
| 11 | `test_wrong_code_returns_401` | PASS |

---

## ContextMemory
**File:** `backend/tests/test_context_memory.py` &nbsp;|&nbsp; **Tests: 83**

### TestAppendTurn (8 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_assistant_turn_does_not_increment` | PASS |
| 2 | `test_content_stored` | PASS |
| 3 | `test_creates_stub_when_session_absent` | PASS |
| 4 | `test_detected_language_stored` | PASS |
| 5 | `test_history_grows_on_each_append` | PASS |
| 6 | `test_returns_turn_record` | PASS |
| 7 | `test_role_stored` | PASS |
| 8 | `test_user_turn_increments_count` | PASS |

### TestBuildFallbackSummary (10 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_fields_collected_ar` | PASS |
| 2 | `test_all_fields_collected_en` | PASS |
| 3 | `test_known_state_label_used_en` | PASS |
| 4 | `test_missing_fields_listed_en` | PASS |
| 5 | `test_no_fields_ar_says_no_data` | PASS |
| 6 | `test_no_fields_en_says_no_data` | PASS |
| 7 | `test_returns_tuple_of_two_strings` | PASS |
| 8 | `test_some_fields_listed_en` | PASS |
| 9 | `test_turn_count_appears_in_summary` | PASS |
| 10 | `test_unknown_state_uses_raw_value` | PASS |

### TestContextMemoryInit (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_custom_ttl` | PASS |
| 2 | `test_default_ttl` | PASS |
| 3 | `test_env_redis_url_used` | PASS |
| 4 | `test_host_port_mode` | PASS |
| 5 | `test_url_mode_calls_from_url` | PASS |

### TestDeleteSession (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_key_removed_after_delete` | PASS |
| 2 | `test_returns_false_when_absent` | PASS |
| 3 | `test_returns_true_when_key_exists` | PASS |

### TestGetCollectedFields (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_returns_copy_not_reference` | PASS |
| 2 | `test_returns_empty_when_absent` | PASS |
| 3 | `test_returns_fields_when_present` | PASS |

### TestGetContextMemory (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_returns_context_memory_instance` | PASS |
| 2 | `test_returns_same_instance_on_second_call` | PASS |
| 3 | `test_reuses_existing_instance` | PASS |

### TestGetHistory (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_empty_when_no_session` | PASS |
| 2 | `test_last_n_larger_than_history_returns_all` | PASS |
| 3 | `test_last_n_returns_tail` | PASS |
| 4 | `test_returns_full_history` | PASS |
| 5 | `test_returns_list_of_turn_records` | PASS |

### TestGetMissingFields (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_missing_when_no_session` | PASS |
| 2 | `test_none_missing_when_all_collected` | PASS |
| 3 | `test_returns_list_type` | PASS |
| 4 | `test_returns_sorted_list` | PASS |
| 5 | `test_subset_missing` | PASS |

### TestGetSummary (9 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_collected_fields_in_result` | PASS |
| 2 | `test_fallback_on_bad_llm_response` | PASS |
| 3 | `test_language_from_session` | PASS |
| 4 | `test_llm_summaries_propagated` | PASS |
| 5 | `test_missing_fields_correct` | PASS |
| 6 | `test_non_existent_session_uses_stub` | PASS |
| 7 | `test_returns_context_summary` | PASS |
| 8 | `test_session_id_in_result` | PASS |
| 9 | `test_turn_count_in_result` | PASS |

### TestKey (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_different_ids_produce_different_keys` | PASS |
| 2 | `test_key_format` | PASS |
| 3 | `test_key_starts_with_prefix` | PASS |

### TestLoadSession (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_history_round_trips` | PASS |
| 2 | `test_returns_none_on_bad_json` | PASS |
| 3 | `test_returns_none_on_empty_value` | PASS |
| 4 | `test_returns_none_when_absent` | PASS |
| 5 | `test_returns_session_memory_when_present` | PASS |
| 6 | `test_state_round_trips` | PASS |

### TestNow (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_contains_utc_indicator` | PASS |
| 2 | `test_monotonic` | PASS |
| 3 | `test_returns_string` | PASS |

### TestParseSummaryResponse (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_empty_response_triggers_fallback` | PASS |
| 2 | `test_invalid_json_triggers_fallback` | PASS |
| 3 | `test_json_with_markdown_fences` | PASS |
| 4 | `test_missing_summary_ar_triggers_fallback` | PASS |
| 5 | `test_missing_summary_en_triggers_fallback` | PASS |
| 6 | `test_regex_fallback_on_surrounding_text` | PASS |
| 7 | `test_valid_json` | PASS |

### TestSaveSession (8 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_casts_field_values_to_str` | PASS |
| 2 | `test_coerces_dict_turns_to_turn_record` | PASS |
| 3 | `test_counts_only_user_turns` | PASS |
| 4 | `test_persists_to_redis` | PASS |
| 5 | `test_preserves_created_at_on_update` | PASS |
| 6 | `test_returns_session_memory` | PASS |
| 7 | `test_stores_language` | PASS |
| 8 | `test_turn_count_zero_for_empty_history` | PASS |

### TestUpdateFields (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_creates_stub_when_absent` | PASS |
| 2 | `test_ignores_empty_string_values` | PASS |
| 3 | `test_merges_new_fields` | PASS |
| 4 | `test_overwrites_existing_field` | PASS |
| 5 | `test_returns_full_collected_fields` | PASS |

---

## ConversationManager
**File:** `backend/tests/test_conversation_manager.py` &nbsp;|&nbsp; **Tests: 63**

### TestExtractFieldsEmploymentStatus (8 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_employed_keyword` | PASS |
| 2 | `test_arabic_unemployed_keyword` | PASS |
| 3 | `test_does_not_overwrite_existing_status` | PASS |
| 4 | `test_employed_keyword` | PASS |
| 5 | `test_looking_for_work_keyword` | PASS |
| 6 | `test_retired_maps_to_not_in_labour_force` | PASS |
| 7 | `test_unemployed_keyword` | PASS |
| 8 | `test_working_keyword` | PASS |

### TestExtractFieldsEmploymentType (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_freelance_maps_to_self_employed` | PASS |
| 2 | `test_full_time_extracted` | PASS |
| 3 | `test_fulltime_hyphenated` | PASS |
| 4 | `test_part_time_extracted` | PASS |
| 5 | `test_self_employed_extracted` | PASS |

### TestExtractFieldsHours (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_extracts_arabic_hours_keyword` | PASS |
| 2 | `test_extracts_digit_hours_pattern` | PASS |
| 3 | `test_extracts_hrs_abbreviation` | PASS |
| 4 | `test_no_hours_pattern_leaves_field_absent` | PASS |

### TestExtractFieldsIndustry (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_technology_sector` | PASS |
| 2 | `test_education_sector` | PASS |
| 3 | `test_finance_sector` | PASS |
| 4 | `test_government_sector` | PASS |
| 5 | `test_healthcare_sector` | PASS |
| 6 | `test_technology_sector` | PASS |

### TestExtractFieldsJobTitle (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_doctor_keyword_captured` | PASS |
| 2 | `test_doctor_keyword_captured` | PASS |
| 3 | `test_engineer_keyword_captures_raw_text` | PASS |

### TestIsAmbiguous (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_vague_word_is_ambiguous` | PASS |
| 2 | `test_empty_string_is_ambiguous` | PASS |
| 3 | `test_ok_is_ambiguous` | PASS |
| 4 | `test_single_vague_word_yes_is_ambiguous` | PASS |
| 5 | `test_substantive_arabic_not_ambiguous` | PASS |
| 6 | `test_substantive_english_not_ambiguous` | PASS |
| 7 | `test_very_short_string_is_ambiguous` | PASS |

### TestIsConfirmed (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_\u0645\u0648\u0627\u0641\u0642_confirms` | PASS |
| 2 | `test_arabic_confirmation` | PASS |
| 3 | `test_correct_confirms_english` | PASS |
| 4 | `test_empty_string_not_confirmed` | PASS |
| 5 | `test_looks_good_confirms_english` | PASS |
| 6 | `test_unrelated_text_not_confirmed` | PASS |
| 7 | `test_yes_confirms_english` | PASS |

### TestNewContext (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_collected_data_starts_empty` | PASS |
| 2 | `test_history_starts_empty` | PASS |
| 3 | `test_initial_state_is_greeting` | PASS |
| 4 | `test_invalid_language_defaults_to_en` | PASS |
| 5 | `test_language_ar_stored` | PASS |
| 6 | `test_language_en_stored` | PASS |
| 7 | `test_session_id_stored` | PASS |

### TestProcessMessage (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_assistant_reply_appended_to_history` | PASS |
| 2 | `test_history_grows_with_each_turn` | PASS |
| 3 | `test_returns_string_reply` | PASS |
| 4 | `test_state_transitions_after_turn` | PASS |
| 5 | `test_user_message_appended_to_history` | PASS |

### TestTransition (11 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_clarifying_returns_to_collecting` | PASS |
| 2 | `test_collecting_advances_to_validating_when_all_fields_present` | PASS |
| 3 | `test_collecting_moves_to_clarifying_on_ambiguous_input` | PASS |
| 4 | `test_collecting_stays_when_fields_incomplete` | PASS |
| 5 | `test_completing_is_terminal` | PASS |
| 6 | `test_greeting_always_advances_to_collecting` | PASS |
| 7 | `test_validating_correction_arabic` | PASS |
| 8 | `test_validating_correction_unknown_field_still_transitions` | PASS |
| 9 | `test_validating_correction_updates_field` | PASS |
| 10 | `test_validating_moves_to_completing_on_confirmation` | PASS |
| 11 | `test_validating_returns_to_collecting_on_non_confirmation` | PASS |

---

## ConversationManager (extended — full questionnaire, skip logic)
**File:** `backend/tests/test_conversation_manager_extended.py` &nbsp;|&nbsp; **Tests: 42**

### TestArabicLanguageFlows (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_confirmation` | PASS |
| 2 | `test_arabic_denial_stays_collecting` | PASS |
| 3 | `test_arabic_employed_response` | PASS |
| 4 | `test_arabic_hours_response` | PASS |
| 5 | `test_arabic_unemployed_response` | PASS |

### TestClarificationFlow (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_ambiguous_triggers_clarifying` | PASS |
| 2 | `test_clarifying_returns_to_collecting` | PASS |
| 3 | `test_clarifying_state_has_follow_up_question` | PASS |
| 4 | `test_max_clarification_attempts` | PASS |
| 5 | `test_repeated_ambiguous_stays_clarifying` | PASS |

### TestConcurrentSessions (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_session_data_not_shared` | PASS |
| 2 | `test_ten_concurrent_sessions` | PASS |
| 3 | `test_two_sessions_independent` | PASS |

### TestEmployedPathEndToEnd (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_employed_fields_collected` | PASS |
| 2 | `test_employed_path_collects_hours` | PASS |
| 3 | `test_employed_path_collects_job_title` | PASS |
| 4 | `test_employed_path_reaches_completing` | PASS |

### TestErrorRecovery (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_empty_string_handled` | PASS |
| 2 | `test_llm_failure_returns_fallback_reply` | PASS |
| 3 | `test_none_input_handled` | PASS |
| 4 | `test_very_long_input_handled` | PASS |

### TestFieldExtractionEdgeCases (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_hours_as_arabic_numeral` | PASS |
| 2 | `test_hours_range_expression` | PASS |
| 3 | `test_job_title_with_seniority` | PASS |
| 4 | `test_sector_public_sector` | PASS |
| 5 | `test_wage_range_extracted` | PASS |

### TestNILFPath (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_homemaker_status_accepted` | PASS |
| 2 | `test_nilf_short_path` | PASS |
| 3 | `test_nilf_status_accepted` | PASS |
| 4 | `test_student_status_accepted` | PASS |

### TestSkipLogic (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_employed_requires_job_duties` | PASS |
| 2 | `test_nilf_minimal_fields` | PASS |
| 3 | `test_unemployed_skips_employment_type` | PASS |
| 4 | `test_unknown_status_uses_max_fields` | PASS |

### TestStateMachineInvariants (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_completing_is_terminal` | PASS |
| 2 | `test_history_grows_monotonically` | PASS |
| 3 | `test_language_unchanged_during_session` | PASS |
| 4 | `test_session_id_unchanged` | PASS |
| 5 | `test_state_always_valid` | PASS |

### TestUnemployedPathEndToEnd (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_unemployed_path_completes` | PASS |
| 2 | `test_unemployed_path_no_hours_field` | PASS |
| 3 | `test_unemployed_skips_industry` | PASS |

---

## EmotionalIntelligence
**File:** `backend/tests/test_emotional_intelligence.py` &nbsp;|&nbsp; **Tests: 74**

### TestAnalyze (10 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_analyze_returns_emotional_analysis` | PASS |
| 2 | `test_detected_language_set_in_result` | PASS |
| 3 | `test_empty_text_language_hint_ar_preserved` | PASS |
| 4 | `test_empty_text_language_hint_en_preserved` | PASS |
| 5 | `test_empty_text_returns_neutral` | PASS |
| 6 | `test_llm_exception_triggers_fallback` | PASS |
| 7 | `test_llm_result_propagated` | PASS |
| 8 | `test_raw_text_preserved_in_result` | PASS |
| 9 | `test_signals_in_result` | PASS |
| 10 | `test_whitespace_only_returns_neutral` | PASS |

### TestBuildFallbackAnalysis (8 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_action_from_high_frustration` | PASS |
| 2 | `test_action_from_low_stress` | PASS |
| 3 | `test_ar_adapted_prompt_from_template` | PASS |
| 4 | `test_en_adapted_prompt_from_template` | PASS |
| 5 | `test_propagates_signals` | PASS |
| 6 | `test_raw_text_preserved` | PASS |
| 7 | `test_returns_emotional_analysis_instance` | PASS |
| 8 | `test_support_messages_from_template` | PASS |

### TestDetectEmotionRules (15 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_confused_pattern` | PASS |
| 2 | `test_arabic_stressed_pattern` | PASS |
| 3 | `test_caps_words_boost_frustrated` | PASS |
| 4 | `test_confidence_capped_at_0_95` | PASS |
| 5 | `test_confused_keyword_detected` | PASS |
| 6 | `test_engaged_keyword_detected` | PASS |
| 7 | `test_frustrated_keyword_detected` | PASS |
| 8 | `test_intensity_capped_at_1_0` | PASS |
| 9 | `test_keyword_signal_weight_is_correct` | PASS |
| 10 | `test_neutral_for_plain_text` | PASS |
| 11 | `test_neutral_signals_empty` | PASS |
| 12 | `test_stressed_keyword_detected` | PASS |
| 13 | `test_triple_exclamation_adds_punctuation_signal` | PASS |
| 14 | `test_triple_exclamation_boosts_frustrated` | PASS |
| 15 | `test_triple_question_marks_boosts_confused` | PASS |

### TestDetectScript (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_empty_string_returns_en` | PASS |
| 2 | `test_low_arabic_returns_en` | PASS |
| 3 | `test_mixed_returns_mixed` | PASS |
| 4 | `test_mostly_arabic_returns_ar` | PASS |
| 5 | `test_numbers_only_returns_en` | PASS |
| 6 | `test_pure_arabic_returns_ar` | PASS |
| 7 | `test_pure_english_returns_en` | PASS |

### TestDetermineAction (9 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_actions_have_reason_string` | PASS |
| 2 | `test_confused_high_intensity_returns_pause` | PASS |
| 3 | `test_confused_low_intensity_returns_slow_down` | PASS |
| 4 | `test_engaged_returns_continue` | PASS |
| 5 | `test_frustrated_high_intensity_returns_end` | PASS |
| 6 | `test_frustrated_low_intensity_returns_slow_down` | PASS |
| 7 | `test_neutral_returns_continue` | PASS |
| 8 | `test_stressed_high_intensity_returns_pause` | PASS |
| 9 | `test_stressed_low_intensity_returns_slow_down` | PASS |

### TestEmotionalStateEnum (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_confused_value` | PASS |
| 2 | `test_engaged_value` | PASS |
| 3 | `test_frustrated_value` | PASS |
| 4 | `test_neutral_value` | PASS |
| 5 | `test_stressed_value` | PASS |

### TestGetEmotionalIntelligence (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_returns_emotional_intelligence_instance` | PASS |
| 2 | `test_returns_same_instance_on_second_call` | PASS |
| 3 | `test_reuses_existing_instance` | PASS |

### TestParseLLMResponse (13 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_action_reason_from_data` | PASS |
| 2 | `test_confidence_clamped_above_one` | PASS |
| 3 | `test_empty_string_triggers_fallback` | PASS |
| 4 | `test_invalid_json_triggers_fallback` | PASS |
| 5 | `test_invalid_state_falls_back_to_rb_state` | PASS |
| 6 | `test_json_embedded_in_surrounding_text_extracted` | PASS |
| 7 | `test_markdown_fences_stripped` | PASS |
| 8 | `test_missing_action_reason_uses_determine_action` | PASS |
| 9 | `test_missing_text_fields_use_templates` | PASS |
| 10 | `test_valid_json_action_parsed` | PASS |
| 11 | `test_valid_json_confidence_parsed` | PASS |
| 12 | `test_valid_json_state_parsed` | PASS |
| 13 | `test_valid_json_text_fields_parsed` | PASS |

### TestSurveyActionEnum (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_continue_value` | PASS |
| 2 | `test_end_value` | PASS |
| 3 | `test_pause_value` | PASS |
| 4 | `test_slow_down_value` | PASS |

---

## Evaluation Framework (BM25 / Flat / Hierarchical)
**File:** `backend/tests/test_evaluation.py` &nbsp;|&nbsp; **Tests: 16**

### Module-level tests (16 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_10_major_groups_covered` | PASS |
| 2 | `test_bm25_confidence_in_range` | PASS |
| 3 | `test_bm25_predict_returns_prediction_result` | PASS |
| 4 | `test_bm25_predicted_in_top3` | PASS |
| 5 | `test_bm25_top3_length_respects_top_k` | PASS |
| 6 | `test_comparison_table_printed` | PASS |
| 7 | `test_corpus_has_100_cases` | PASS |
| 8 | `test_each_case_is_2_tuple` | PASS |
| 9 | `test_evaluate_system_returns_system_metrics` | PASS |
| 10 | `test_flat_predict_returns_prediction_result` | PASS |
| 11 | `test_flat_predicted_in_top3` | PASS |
| 12 | `test_hierarchical_confidence_in_range` | PASS |
| 13 | `test_hierarchical_predict_returns_prediction_result` | PASS |
| 14 | `test_hitl_rate_is_one_when_all_low_confidence` | PASS |
| 15 | `test_hitl_rate_is_zero_when_all_high_confidence` | PASS |
| 16 | `test_isco_codes_are_4_digits` | PASS |

---

## HITL & End-to-End (extended)
**File:** `backend/tests/test_hitl_and_e2e_extended.py` &nbsp;|&nbsp; **Tests: 37**

### TestEndToEndLanguageScenarios (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_hindi_turn_processed` | PASS |
| 2 | `test_language_switches_mid_session` | PASS |
| 3 | `test_tagalog_turn_processed` | PASS |
| 4 | `test_urdu_turn_processed` | PASS |

### TestEndToEndScenarios (12 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_audit_logged_every_turn` | PASS |
| 2 | `test_code_switched_input` | PASS |
| 3 | `test_frustrated_respondent_handled` | PASS |
| 4 | `test_happy_path_arabic_employed` | PASS |
| 5 | `test_happy_path_english_employed` | PASS |
| 6 | `test_hitl_escalated_session_flagged` | PASS |
| 7 | `test_isco_classification_included_for_job_entity` | PASS |
| 8 | `test_memory_saved_every_turn` | PASS |
| 9 | `test_multiple_turns_accumulate_data` | PASS |
| 10 | `test_result_has_expected_fields` | PASS |
| 11 | `test_session_completed_flag_on_completion` | PASS |
| 12 | `test_validation_failure_notified_in_response` | PASS |

### TestHITLEdgeCases (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_100_responses_session` | PASS |
| 2 | `test_empty_session_is_escalated` | PASS |
| 3 | `test_missing_isco_on_non_job_field_not_flagged` | PASS |
| 4 | `test_random_quality_check_rate` | PASS |
| 5 | `test_single_response_session` | PASS |

### TestHITLPriorityQueue (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_high_confidence_creates_pass_review` | PASS |
| 2 | `test_high_priority_before_low_priority` | PASS |
| 3 | `test_low_confidence_creates_escalated_review` | PASS |
| 4 | `test_pending_queue_sorted_oldest_first` | PASS |
| 5 | `test_resolved_items_not_in_queue` | PASS |

### TestHITLReviewActions (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_approve_action` | PASS |
| 2 | `test_correct_action` | PASS |
| 3 | `test_double_resolve_allowed` | PASS |
| 4 | `test_reject_action` | PASS |
| 5 | `test_unknown_review_id_raises` | PASS |

### TestPersonRegisterPreFill (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_new_respondent_no_prefill` | PASS |
| 2 | `test_no_active_record_returns_none` | PASS |
| 3 | `test_prefill_reduces_required_questions` | PASS |
| 4 | `test_returning_respondent_prefilled` | PASS |

### TestPersonRegisterUpdate (2 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_register_overwrites_old_data` | PASS |
| 2 | `test_register_updated_after_survey_completion` | PASS |

---

## HITLQualityManager
**File:** `backend/tests/test_hitl_quality_manager.py` &nbsp;|&nbsp; **Tests: 81**

### TestBuildFallbackReport (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_ar_report_non_empty` | PASS |
| 2 | `test_escalated_en_contains_escalated` | PASS |
| 3 | `test_fail_en_contains_fail` | PASS |
| 4 | `test_pass_en_contains_pass` | PASS |
| 5 | `test_returns_tuple_of_two_strings` | PASS |
| 6 | `test_score_appears_in_report` | PASS |

### TestComputeMetrics (9 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_avg_confidence_computed` | PASS |
| 2 | `test_avg_confidence_zero_when_no_scores` | PASS |
| 3 | `test_flagged_count_from_flagged_list` | PASS |
| 4 | `test_isco_coverage_fraction` | PASS |
| 5 | `test_low_confidence_count` | PASS |
| 6 | `test_missing_isco_count` | PASS |
| 7 | `test_responses_with_isco_count` | PASS |
| 8 | `test_total_responses_count` | PASS |
| 9 | `test_zero_responses` | PASS |

### TestComputeQualityScore (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_low_confidence_penalises_score` | PASS |
| 2 | `test_missing_isco_penalises_coverage` | PASS |
| 3 | `test_perfect_session` | PASS |
| 4 | `test_score_bounded_between_zero_and_one` | PASS |
| 5 | `test_score_is_rounded_to_four_decimals` | PASS |
| 6 | `test_zero_responses_returns_zero` | PASS |

### TestDetermineStatus (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_high_score_no_flags_returns_pass` | PASS |
| 2 | `test_score_at_escalation_threshold_not_escalated` | PASS |
| 3 | `test_score_at_pass_threshold_returns_pass` | PASS |
| 4 | `test_score_below_escalation_returns_escalated` | PASS |
| 5 | `test_score_just_below_pass_returns_fail` | PASS |
| 6 | `test_too_many_flags_returns_escalated` | PASS |
| 7 | `test_two_flags_not_escalated` | PASS |

### TestEscalationReason (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_reason_for_low_score_only` | PASS |
| 2 | `test_reason_for_too_many_flags_only` | PASS |
| 3 | `test_reason_is_non_empty_string` | PASS |
| 4 | `test_reason_mentions_both_triggers` | PASS |

### TestFlagItems (9 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_empty_isco_string_flagged` | PASS |
| 2 | `test_empty_responses_returns_empty` | PASS |
| 3 | `test_exact_threshold_not_flagged` | PASS |
| 4 | `test_flagged_item_confidence_preserved` | PASS |
| 5 | `test_good_response_not_flagged` | PASS |
| 6 | `test_isco_with_null_confidence_not_flagged` | PASS |
| 7 | `test_low_confidence_flagged` | PASS |
| 8 | `test_missing_isco_flagged` | PASS |
| 9 | `test_response_id_preserved_in_flag` | PASS |

### TestFlagReasonEnum (2 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_low_confidence_isco_value` | PASS |
| 2 | `test_missing_isco_value` | PASS |

### TestGetHITLQualityManager (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_returns_instance` | PASS |
| 2 | `test_returns_same_instance_on_second_call` | PASS |
| 3 | `test_reuses_existing_instance` | PASS |

### TestGetPendingReviews (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_escalated_unresolved_included` | PASS |
| 2 | `test_pending_review_fields` | PASS |
| 3 | `test_pending_reviews_oldest_first` | PASS |
| 4 | `test_resolved_review_not_in_pending` | PASS |
| 5 | `test_returns_list` | PASS |

### TestParseReportResponse (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_empty_string_triggers_fallback` | PASS |
| 2 | `test_invalid_json_triggers_fallback` | PASS |
| 3 | `test_json_in_surrounding_text_extracted` | PASS |
| 4 | `test_markdown_fences_stripped` | PASS |
| 5 | `test_missing_en_triggers_fallback` | PASS |
| 6 | `test_valid_json_ar_parsed` | PASS |
| 7 | `test_valid_json_en_parsed` | PASS |

### TestQualityThresholds (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_escalation_threshold` | PASS |
| 2 | `test_low_confidence_threshold` | PASS |
| 3 | `test_max_flags_before_escalation` | PASS |
| 4 | `test_min_pass_quality_score` | PASS |

### TestResolveReview (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_resolve_returns_pending_review` | PASS |
| 2 | `test_resolve_sets_reviewed_at` | PASS |
| 3 | `test_resolve_sets_reviewed_by` | PASS |
| 4 | `test_resolve_sets_reviewer_notes` | PASS |
| 5 | `test_resolve_unknown_id_raises` | PASS |

### TestReviewSession (11 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_escalated_has_escalation_reason` | PASS |
| 2 | `test_escalated_status_for_empty_session` | PASS |
| 3 | `test_fail_status_for_moderate_session` | PASS |
| 4 | `test_flagged_items_populated` | PASS |
| 5 | `test_llm_error_triggers_fallback_report` | PASS |
| 6 | `test_llm_report_text_propagated` | PASS |
| 7 | `test_metrics_session_id_matches` | PASS |
| 8 | `test_pass_status_for_good_session` | PASS |
| 9 | `test_quality_score_in_unit_interval` | PASS |
| 10 | `test_returns_quality_report` | PASS |
| 11 | `test_review_id_is_positive` | PASS |

### TestReviewStatusEnum (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_escalated_value` | PASS |
| 2 | `test_fail_value` | PASS |
| 3 | `test_pass_value` | PASS |

---

## ISCEDClassifier (ISCED 2011 + ISCED-F 2013)
**File:** `backend/tests/test_isced_classifier.py` &nbsp;|&nbsp; **Tests: 17**

### Module-level tests (17 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_phd_returns_level_8` | PASS |
| 2 | `test_arabic_secondary_returns_level_3` | PASS |
| 3 | `test_arabic_university_returns_level_6` | PASS |
| 4 | `test_bachelor_returns_level_6` | PASS |
| 5 | `test_confidence_in_range` | PASS |
| 6 | `test_each_level_has_required_keys` | PASS |
| 7 | `test_empty_string_returns_default` | PASS |
| 8 | `test_isced_levels_0_to_8` | PASS |
| 9 | `test_level_title_is_string` | PASS |
| 10 | `test_masters_returns_level_7` | PASS |
| 11 | `test_method_is_keyword` | PASS |
| 12 | `test_phd_returns_level_8` | PASS |
| 13 | `test_primary_returns_level_1` | PASS |
| 14 | `test_raw_text_preserved` | PASS |
| 15 | `test_returns_isced_classification` | PASS |
| 16 | `test_secondary_returns_level_3` | PASS |
| 17 | `test_whitespace_returns_default` | PASS |

---

## ISCOClassifier
**File:** `backend/tests/test_isco_classifier.py` &nbsp;|&nbsp; **Tests: 31**

### TestClassifyEdgeCases (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_empty_input_returns_sentinel` | PASS |
| 2 | `test_no_candidates_returns_sentinel` | PASS |
| 3 | `test_whitespace_only_returns_sentinel` | PASS |

### TestClassifyFastPath (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_high_confidence_does_not_call_llm` | PASS |
| 2 | `test_high_confidence_primary_is_top_candidate` | PASS |
| 3 | `test_high_confidence_uses_semantic_method` | PASS |

### TestClassifyLanguage (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_input_detected_as_ar` | PASS |
| 2 | `test_english_input_detected_as_en` | PASS |
| 3 | `test_mixed_input_detected_as_mixed` | PASS |

### TestClassifyLlmPath (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_candidates_included_in_result` | PASS |
| 2 | `test_llm_failure_falls_back_to_top_semantic` | PASS |
| 3 | `test_llm_selected_code_is_primary` | PASS |
| 4 | `test_low_confidence_uses_llm_ranked_method` | PASS |

### TestClassifyResultShape (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_context_passed_to_store_search` | PASS |
| 2 | `test_min_usable_confidence_constant_exposed` | PASS |
| 3 | `test_reasoning_is_non_empty_string` | PASS |
| 4 | `test_result_is_isco_classification` | PASS |

### TestDetectScript (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_text_returns_ar` | PASS |
| 2 | `test_digits_and_punctuation_only_returns_other` | PASS |
| 3 | `test_empty_string_returns_other` | PASS |
| 4 | `test_english_text_returns_en` | PASS |
| 5 | `test_mixed_returns_mixed` | PASS |
| 6 | `test_mostly_arabic_with_small_latin_returns_ar` | PASS |
| 7 | `test_pure_latin_no_arabic_is_en` | PASS |

### TestParseLlmResponse (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_empty_string_falls_back_to_top_candidate` | PASS |
| 2 | `test_json_embedded_in_prose_is_extracted` | PASS |
| 3 | `test_malformed_json_falls_back_to_top_candidate` | PASS |
| 4 | `test_reasoning_preserved_in_return` | PASS |
| 5 | `test_strips_markdown_code_fence` | PASS |
| 6 | `test_unrecognised_code_falls_back_to_top_candidate` | PASS |
| 7 | `test_valid_json_selects_correct_candidate` | PASS |

---

## ISCOClassifier (extended — hierarchical RAG, keyword pre-filter)
**File:** `backend/tests/test_isco_classifier_extended.py` &nbsp;|&nbsp; **Tests: 80**

### TestAllISCOMajorGroups (11 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_ten_major_groups_have_distinct_codes` | PASS |
| 2 | `test_major_group_classified[Administrative Secretary-4]` | PASS |
| 3 | `test_major_group_classified[Armed Forces Officer-0]` | PASS |
| 4 | `test_major_group_classified[Chief Executive Officer-1]` | PASS |
| 5 | `test_major_group_classified[Cleaner-9]` | PASS |
| 6 | `test_major_group_classified[Construction Electrician-7]` | PASS |
| 7 | `test_major_group_classified[Crop Farmer-6]` | PASS |
| 8 | `test_major_group_classified[Laboratory Technician-3]` | PASS |
| 9 | `test_major_group_classified[Machine Operator-8]` | PASS |
| 10 | `test_major_group_classified[Retail Sales Assistant-5]` | PASS |
| 11 | `test_major_group_classified[Software Engineer-2]` | PASS |

### TestArabicJobTitles (12 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_result_has_arabic_label` | PASS |
| 2 | `test_arabic_result_has_english_label` | PASS |
| 3 | `test_arabic_title_classified[\u0633\u0627\u0626\u0642]` | PASS |
| 4 | `test_arabic_title_classified[\u0637\u0627\u0647\u064d]` | PASS |
| 5 | `test_arabic_title_classified[\u0637\u0628\u064a\u0628]` | PASS |
| 6 | `test_arabic_title_classified[\u0639\u0627\u0645\u0644 \u0628\u0646\u0627\u0621]` | PASS |
| 7 | `test_arabic_title_classified[\u0645\u062d\u0627\u0633\u0628]` | PASS |
| 8 | `test_arabic_title_classified[\u0645\u062f\u064a\u0631]` | PASS |
| 9 | `test_arabic_title_classified[\u0645\u0639\u0644\u0645]` | PASS |
| 10 | `test_arabic_title_classified[\u0645\u0645\u0631\u0636\u0629]` | PASS |
| 11 | `test_arabic_title_classified[\u0645\u0647\u0646\u062f\u0633 \u0628\u0631\u0645\u062c\u064a\u0627\u062a]` | PASS |
| 12 | `test_arabic_title_classified[\u0645\u0648\u0638\u0641 \u0625\u062f\u0627\u0631\u064a]` | PASS |

### TestConfidenceThresholds (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_confidence_bounded_0_to_1` | PASS |
| 2 | `test_confidence_exactly_at_threshold` | PASS |
| 3 | `test_hitl_not_required_above_threshold` | PASS |
| 4 | `test_hitl_required_below_threshold` | PASS |
| 5 | `test_zero_confidence_triggers_hitl` | PASS |

### TestErrorResilience (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_classify_html_injection` | PASS |
| 2 | `test_classify_only_numbers` | PASS |
| 3 | `test_classify_special_characters` | PASS |
| 4 | `test_classify_very_long_input` | PASS |
| 5 | `test_llm_exception_propagates_or_sentinel` | PASS |
| 6 | `test_qdrant_timeout_propagates_or_sentinel` | PASS |

### TestHierarchicalStages (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_fast_path_skips_llm` | PASS |
| 2 | `test_llm_used_for_low_similarity` | PASS |
| 3 | `test_result_has_confidence_score` | PASS |
| 4 | `test_stage_confidences_when_present` | PASS |
| 5 | `test_weighted_confidence_formula` | PASS |

### TestISCOCodeFormat (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_major_group_code_length_1` | PASS |
| 2 | `test_primary_code_is_4_digits` | PASS |
| 3 | `test_primary_code_is_numeric_string` | PASS |
| 4 | `test_sentinel_code_on_empty_input` | PASS |
| 5 | `test_sentinel_code_on_whitespace` | PASS |
| 6 | `test_unit_group_code_starts_with_valid_major` | PASS |

### TestInformalDescriptions (14 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_informal_description_classified[I build houses]` | PASS |
| 2 | `test_informal_description_classified[I clean offices]` | PASS |
| 3 | `test_informal_description_classified[I cook food]` | PASS |
| 4 | `test_informal_description_classified[I drive people around]` | PASS |
| 5 | `test_informal_description_classified[I fix machines]` | PASS |
| 6 | `test_informal_description_classified[I manage a team]` | PASS |
| 7 | `test_informal_description_classified[I sell things in a shop]` | PASS |
| 8 | `test_informal_description_classified[I take care of patients]` | PASS |
| 9 | `test_informal_description_classified[I teach children]` | PASS |
| 10 | `test_informal_description_classified[I work with computers]` | PASS |
| 11 | `test_informal_description_classified[\u0623\u0628\u064a\u0639 \u0628\u0636\u0627\u0626\u0639]` | PASS |
| 12 | `test_informal_description_classified[\u0623\u0634\u062a\u063a\u0644 \u0641\u064a \u0627\u0644\u0643\u0645\u0628\u064a\u0648\u062a\u0631]` | PASS |
| 13 | `test_informal_description_classified[\u0623\u0639\u062a\u0646\u064a \u0628\u0627\u0644\u0645\u0631\u0636\u0649]` | PASS |
| 14 | `test_very_vague_description_triggers_hitl` | PASS |

### TestUAESpecificOccupations (21 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_domestic_helper_not_manager` | PASS |
| 2 | `test_uae_occupation_classified[Accountant]` | PASS |
| 3 | `test_uae_occupation_classified[Architect]` | PASS |
| 4 | `test_uae_occupation_classified[Cashier]` | PASS |
| 5 | `test_uae_occupation_classified[Civil Engineer]` | PASS |
| 6 | `test_uae_occupation_classified[Construction Laborer]` | PASS |
| 7 | `test_uae_occupation_classified[Delivery Driver]` | PASS |
| 8 | `test_uae_occupation_classified[Doctor]` | PASS |
| 9 | `test_uae_occupation_classified[Domestic Helper]` | PASS |
| 10 | `test_uae_occupation_classified[HR Manager]` | PASS |
| 11 | `test_uae_occupation_classified[Hotel Housekeeper]` | PASS |
| 12 | `test_uae_occupation_classified[IT Support Technician]` | PASS |
| 13 | `test_uae_occupation_classified[Legal Advisor]` | PASS |
| 14 | `test_uae_occupation_classified[Nurse]` | PASS |
| 15 | `test_uae_occupation_classified[Restaurant Waiter]` | PASS |
| 16 | `test_uae_occupation_classified[Sales Executive]` | PASS |
| 17 | `test_uae_occupation_classified[Security Guard]` | PASS |
| 18 | `test_uae_occupation_classified[Shop Assistant]` | PASS |
| 19 | `test_uae_occupation_classified[Taxi Driver]` | PASS |
| 20 | `test_uae_occupation_classified[Teacher]` | PASS |
| 21 | `test_uae_occupation_classified[Warehouse Worker]` | PASS |

---

## ISICClassifier (ISIC Rev.4)
**File:** `backend/tests/test_isic_classifier.py` &nbsp;|&nbsp; **Tests: 19**

### Module-level tests (19 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_entries_have_required_keys` | PASS |
| 2 | `test_alternatives_list_populated` | PASS |
| 3 | `test_arabic_hospital_maps_to_section_Q` | PASS |
| 4 | `test_arabic_software_maps_to_section_J` | PASS |
| 5 | `test_bank_maps_to_section_K` | PASS |
| 6 | `test_confidence_in_range` | PASS |
| 7 | `test_construction_maps_to_section_F` | PASS |
| 8 | `test_division_code_for_it_services` | PASS |
| 9 | `test_division_codes_are_two_digits` | PASS |
| 10 | `test_empty_text_returns_fallback` | PASS |
| 11 | `test_high_confidence_skips_llm` | PASS |
| 12 | `test_hospital_maps_to_section_Q` | PASS |
| 13 | `test_isic_data_not_empty` | PASS |
| 14 | `test_llm_rerank_called_for_low_confidence` | PASS |
| 15 | `test_llm_rerank_exception_falls_back_to_keyword` | PASS |
| 16 | `test_returns_isic_classification` | PASS |
| 17 | `test_school_maps_to_section_P` | PASS |
| 18 | `test_sections_A_to_U_present` | PASS |
| 19 | `test_software_company_maps_to_section_J` | PASS |

---

## LanguageProcessor
**File:** `backend/tests/test_language_processor.py` &nbsp;|&nbsp; **Tests: 52**

### TestBuildSegments (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_segment_has_arabic_script` | PASS |
| 2 | `test_code_switched_text_yields_multiple_scripts` | PASS |
| 3 | `test_empty_string_returns_empty_list` | PASS |
| 4 | `test_pure_latin_has_no_arabic_segment` | PASS |
| 5 | `test_short_segment_has_no_detected_language` | PASS |
| 6 | `test_whitespace_does_not_create_empty_segments` | PASS |

### TestDetectLanguage (8 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_clear_arabic_returns_ar` | PASS |
| 2 | `test_clear_english_returns_en` | PASS |
| 3 | `test_hindi_detection` | PASS |
| 4 | `test_hindi_english_code_switch` | PASS |
| 5 | `test_langdetect_exception_falls_back_gracefully` | PASS |
| 6 | `test_tagalog_detection` | PASS |
| 7 | `test_urdu_detection` | PASS |
| 8 | `test_urdu_english_code_switch` | PASS |

### TestGulfArabicNormalisation (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_gulf_arabic_classified_as_ar_gulf` | PASS |
| 2 | `test_normalise_gulf_arabic_direct` | PASS |
| 3 | `test_normalised_text_none_for_english` | PASS |
| 4 | `test_normalised_text_present_for_arabic` | PASS |
| 5 | `test_normalised_text_replaces_gulf_tokens` | PASS |
| 6 | `test_raw_text_preserved_unchanged_for_gulf` | PASS |

### TestParseEntities (11 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_adds_character_offsets` | PASS |
| 2 | `test_all_lfs_labels_are_accepted` | PASS |
| 3 | `test_empty_array_returns_empty_list` | PASS |
| 4 | `test_entity_not_in_text_has_no_offsets` | PASS |
| 5 | `test_falls_back_to_regex_on_bad_json` | PASS |
| 6 | `test_multiple_entities_returned` | PASS |
| 7 | `test_strips_markdown_code_fence` | PASS |
| 8 | `test_totally_malformed_returns_empty_list` | PASS |
| 9 | `test_unknown_label_is_silently_dropped` | PASS |
| 10 | `test_unknown_language_normalised_to_en` | PASS |
| 11 | `test_valid_json_returns_entities` | PASS |

### TestProcess (10 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_and_latin_ratios_sum_to_one` | PASS |
| 2 | `test_arabic_text_detected_as_ar` | PASS |
| 3 | `test_code_switched_text_flagged` | PASS |
| 4 | `test_confidence_in_range` | PASS |
| 5 | `test_empty_input_returns_default_result` | PASS |
| 6 | `test_english_text_detected_as_en` | PASS |
| 7 | `test_entities_extracted_from_ner_response` | PASS |
| 8 | `test_raw_text_preserved_in_result` | PASS |
| 9 | `test_returns_language_processor_result` | PASS |
| 10 | `test_whitespace_only_treated_as_empty` | PASS |

### TestScriptFallback (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_half_arabic_returns_ar` | PASS |
| 2 | `test_majority_arabic_returns_ar` | PASS |
| 3 | `test_majority_latin_returns_en` | PASS |
| 4 | `test_no_alphabetic_chars_returns_other` | PASS |

### TestSegmentScripts (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_and_latin_ratios_sum_to_one` | PASS |
| 2 | `test_arabic_ratio_correct_for_pure_arabic` | PASS |
| 3 | `test_each_segment_has_valid_script` | PASS |
| 4 | `test_mixed_text_exceeding_threshold_is_code_switched` | PASS |
| 5 | `test_pure_arabic_not_code_switched` | PASS |
| 6 | `test_pure_english_not_code_switched` | PASS |
| 7 | `test_segments_list_not_empty` | PASS |

---

## LanguageProcessor (extended — Gulf Arabic, Devanagari, code-switch)
**File:** `backend/tests/test_language_processor_extended.py` &nbsp;|&nbsp; **Tests: 47**

### TestCodeSwitchingEdgeCases (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_alternating_language_tokens` | PASS |
| 2 | `test_email_address_not_switch` | PASS |
| 3 | `test_numbers_not_counted_as_switch` | PASS |
| 4 | `test_single_foreign_word_not_switched` | PASS |
| 5 | `test_three_language_mix` | PASS |
| 6 | `test_url_in_response` | PASS |
| 7 | `test_very_long_code_switched_text` | PASS |

### TestConfidenceScores (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_confidence_in_valid_range` | PASS |
| 2 | `test_confidence_not_none` | PASS |
| 3 | `test_high_confidence_for_clear_arabic` | PASS |
| 4 | `test_high_confidence_for_clear_english` | PASS |
| 5 | `test_low_confidence_for_ambiguous_input` | PASS |

### TestGulfArabicNormalisation (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_empty_arabic_text` | PASS |
| 2 | `test_gulf_chub_normalised` | PASS |
| 3 | `test_gulf_dialect_shinu_normalised` | PASS |
| 4 | `test_gulf_wain_normalised` | PASS |
| 5 | `test_msa_text_unchanged` | PASS |
| 6 | `test_normalisation_preserves_meaning` | PASS |

### TestHindiDetection (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_devanagari_ratio_detection` | PASS |
| 2 | `test_devanagari_text_detected` | PASS |
| 3 | `test_hindi_english_code_switch` | PASS |
| 4 | `test_hindi_job_description` | PASS |
| 5 | `test_hindi_numerals` | PASS |

### TestInputSecurity (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_emoji_in_response` | PASS |
| 2 | `test_null_byte_in_input` | PASS |
| 3 | `test_prompt_injection_in_input` | PASS |
| 4 | `test_rtl_override_character` | PASS |
| 5 | `test_special_unicode_characters` | PASS |
| 6 | `test_sql_injection_in_input` | PASS |
| 7 | `test_very_long_input` | PASS |

### TestNERMultiLanguage (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_job_title_extracted` | PASS |
| 2 | `test_education_entity_extracted` | PASS |
| 3 | `test_english_job_title_extracted` | PASS |
| 4 | `test_location_entity_extracted` | PASS |
| 5 | `test_multiple_entities_same_sentence` | PASS |
| 6 | `test_ner_invalid_json_returns_empty` | PASS |
| 7 | `test_ner_timeout_propagates_or_empty` | PASS |

### TestTagalogDetection (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_tagalog_arabic_mix` | PASS |
| 2 | `test_tagalog_english_mix` | PASS |
| 3 | `test_tagalog_hours_phrase` | PASS |
| 4 | `test_tagalog_job_description` | PASS |
| 5 | `test_tagalog_text_detected` | PASS |

### TestUrduDetection (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_urdu_empty_input` | PASS |
| 2 | `test_urdu_job_title_detected` | PASS |
| 3 | `test_urdu_roman_script` | PASS |
| 4 | `test_urdu_text_returns_ur` | PASS |
| 5 | `test_urdu_with_english_code_switch` | PASS |

---

## NationalityClassifier
**File:** `backend/tests/test_nationality_classifier.py` &nbsp;|&nbsp; **Tests: 45**

### TestArabicInput (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_egyptian` | PASS |
| 2 | `test_arabic_emirati` | PASS |
| 3 | `test_arabic_indian` | PASS |
| 4 | `test_arabic_phrase` | PASS |
| 5 | `test_arabic_saudi` | PASS |

### TestCaseInsensitivity (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_lowercase_country` | PASS |
| 2 | `test_mixed_case` | PASS |
| 3 | `test_uppercase` | PASS |

### TestConversationalInput (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_change_nationality_to_indian` | PASS |
| 2 | `test_from_phrase` | PASS |
| 3 | `test_nationality_should_be_pakistani` | PASS |
| 4 | `test_raw_text_preserved` | PASS |

### TestDataIntegrity (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_alias_index_not_empty` | PASS |
| 2 | `test_all_entries_have_required_keys` | PASS |
| 3 | `test_country_data_not_empty` | PASS |
| 4 | `test_no_duplicate_iso3` | PASS |
| 5 | `test_stateless_multiple_instances` | PASS |

### TestEdgeCases (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_empty_string` | PASS |
| 2 | `test_gibberish` | PASS |
| 3 | `test_returns_dataclass` | PASS |
| 4 | `test_whitespace_only` | PASS |

### TestIso3Lookup (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_are_lowercase` | PASS |
| 2 | `test_gbr` | PASS |
| 3 | `test_ind` | PASS |
| 4 | `test_phl` | PASS |

### TestM49Lookup (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_india_356` | PASS |
| 2 | `test_pakistan_586` | PASS |
| 3 | `test_uae_784` | PASS |

### TestMultiWordCountries (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_saudi_arabia` | PASS |
| 2 | `test_south_africa` | PASS |
| 3 | `test_sri_lanka` | PASS |
| 4 | `test_united_kingdom` | PASS |
| 5 | `test_united_states` | PASS |

### TestOutputFields (2 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_fields_populated_for_match` | PASS |
| 2 | `test_unknown_sentinel_fields` | PASS |

### TestTopUaeNationalities (10 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_bangladeshi` | PASS |
| 2 | `test_british` | PASS |
| 3 | `test_egyptian` | PASS |
| 4 | `test_emirati` | PASS |
| 5 | `test_filipina` | PASS |
| 6 | `test_filipino` | PASS |
| 7 | `test_india` | PASS |
| 8 | `test_indian` | PASS |
| 9 | `test_pakistani` | PASS |
| 10 | `test_uae` | PASS |

---

## PersonRegister
**File:** `backend/tests/test_person_register.py` &nbsp;|&nbsp; **Tests: 10**

### Module-level tests (10 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_as_collected_data_excludes_none_fields` | PASS |
| 2 | `test_as_collected_data_has_all_fields` | PASS |
| 3 | `test_fields_available_returns_non_none_fields` | PASS |
| 4 | `test_inactive_record_not_returned` | PASS |
| 5 | `test_no_record_returns_none` | PASS |
| 6 | `test_reduction_pct_empty` | PASS |
| 7 | `test_reduction_pct_full_prefill` | PASS |
| 8 | `test_reduction_pct_partial_prefill` | PASS |
| 9 | `test_second_update_overwrites_first` | PASS |
| 10 | `test_update_then_prefill_roundtrip` | PASS |

---

## RAGExpert
**File:** `backend/tests/test_rag_expert.py` &nbsp;|&nbsp; **Tests: 46**

### TestBuildHierarchyInfo (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_hierarchy_info_is_pydantic_model` | PASS |
| 2 | `test_level1_only_major_populated` | PASS |
| 3 | `test_level2_has_major_and_sub_major` | PASS |
| 4 | `test_level4_has_both_parent_groups` | PASS |
| 5 | `test_unknown_sub_major_returns_unknown_label` | PASS |

### TestDetectScript (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_text_returns_ar` | PASS |
| 2 | `test_code_switched_returns_mixed` | PASS |
| 3 | `test_digits_only_returns_other` | PASS |
| 4 | `test_empty_string_returns_other` | PASS |
| 5 | `test_english_text_returns_en` | PASS |
| 6 | `test_predominantly_arabic_returns_ar` | PASS |
| 7 | `test_pure_latin_no_arabic_returns_en` | PASS |

### TestExpandHierarchy (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_raw_results_tagged_as_semantic` | PASS |
| 2 | `test_no_duplicate_codes` | PASS |
| 3 | `test_parent_confidence_lower_than_child` | PASS |
| 4 | `test_parent_groups_added_as_hierarchical_expansion` | PASS |
| 5 | `test_returns_capped_at_top_k` | PASS |
| 6 | `test_returns_list_of_triples` | PASS |
| 7 | `test_sorted_by_confidence_descending` | PASS |

### TestParseExplanations (9 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_explanation_ar_populated_from_llm` | PASS |
| 2 | `test_explanation_en_populated_from_llm` | PASS |
| 3 | `test_hierarchy_info_attached_to_candidate` | PASS |
| 4 | `test_malformed_json_uses_fallback` | PASS |
| 5 | `test_missing_code_falls_back_to_auto_explanation` | PASS |
| 6 | `test_rank_starts_at_1` | PASS |
| 7 | `test_retrieval_stage_preserved` | PASS |
| 8 | `test_strips_markdown_fences` | PASS |
| 9 | `test_valid_json_produces_occupation_candidates` | PASS |

### TestRetrieveEdgeCases (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_empty_query_returns_empty_candidates` | PASS |
| 2 | `test_no_store_results_returns_empty_candidates` | PASS |
| 3 | `test_whitespace_only_returns_empty_candidates` | PASS |

### TestRetrieveLanguage (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_query_detected_as_ar` | PASS |
| 2 | `test_english_query_detected_as_en` | PASS |
| 3 | `test_mixed_query_detected_as_mixed` | PASS |

### TestRetrieveMethodFlag (2 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_method_is_hierarchical_when_parents_added` | PASS |
| 2 | `test_method_is_semantic_only_when_no_parents_needed` | PASS |

### TestRetrieveResultShape (8 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_candidate_ranks_are_sequential` | PASS |
| 2 | `test_candidates_are_occupation_candidates` | PASS |
| 3 | `test_candidates_ranked_from_1` | PASS |
| 4 | `test_confidence_scores_in_valid_range` | PASS |
| 5 | `test_method_is_string` | PASS |
| 6 | `test_query_field_echoed_in_result` | PASS |
| 7 | `test_result_is_rag_expert_result` | PASS |
| 8 | `test_total_retrieved_matches_store_output` | PASS |

### TestTopK (2 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_top_k_clamped_to_10` | PASS |
| 2 | `test_top_k_minimum_is_1` | PASS |

---

## ReportGenerator
**File:** `backend/tests/test_report_generator.py` &nbsp;|&nbsp; **Tests: 45**

### TestBuildProfile (9 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_best_isco_by_confidence` | PASS |
| 2 | `test_employment_status_extracted` | PASS |
| 3 | `test_employment_type_extracted` | PASS |
| 4 | `test_empty_responses_returns_all_none` | PASS |
| 5 | `test_hours_per_week_extracted` | PASS |
| 6 | `test_industry_extracted` | PASS |
| 7 | `test_job_title_extracted` | PASS |
| 8 | `test_latest_answer_wins_for_duplicate_field` | PASS |
| 9 | `test_no_isco_when_no_code_in_responses` | PASS |

### TestEmploymentProfileModel (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_optional_defaults_none` | PASS |
| 2 | `test_fields_populated` | PASS |
| 3 | `test_isco_confidence_bounds` | PASS |

### TestFallbackTemplates (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_text_present_in_all_statuses` | PASS |
| 2 | `test_escalated_fallback_mentions_supervisor` | PASS |
| 3 | `test_fail_fallback_mentions_review` | PASS |
| 4 | `test_pass_fallback_has_no_follow_up` | PASS |
| 5 | `test_unknown_fallback_keys_present` | PASS |

### TestGenerateErrors (2 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_raises_for_in_progress_session` | PASS |
| 2 | `test_raises_for_unknown_session` | PASS |

### TestGenerateHappyPath (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_cached_report_returned_on_second_call` | PASS |
| 2 | `test_language_preserved` | PASS |
| 3 | `test_regenerate_forces_new_llm_call` | PASS |
| 4 | `test_report_en_content` | PASS |
| 5 | `test_report_persisted_to_db` | PASS |
| 6 | `test_returns_survey_report` | PASS |

### TestGenerateWithQuality (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_escalated_quality_status` | PASS |
| 2 | `test_failed_quality_status` | PASS |
| 3 | `test_flagged_count_included` | PASS |
| 4 | `test_quality_score_included` | PASS |

### TestGetReportGeneratorSingleton (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_instance_is_none_before_first_call` | PASS |
| 2 | `test_returns_report_generator_instance` | PASS |
| 3 | `test_same_instance_on_repeated_calls` | PASS |

### TestGetSessionReport (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_returns_most_recent_report` | PASS |
| 2 | `test_returns_none_for_unknown_session` | PASS |
| 3 | `test_returns_none_when_no_report_generated` | PASS |
| 4 | `test_returns_report_after_generation` | PASS |

### TestParseNarrative (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_broken_json_falls_back` | PASS |
| 2 | `test_clean_json_parsed` | PASS |
| 3 | `test_escalated_status_fallback` | PASS |
| 4 | `test_fail_status_fallback` | PASS |
| 5 | `test_markdown_fences_stripped` | PASS |
| 6 | `test_missing_keys_falls_back` | PASS |
| 7 | `test_pass_status_fallback` | PASS |

### TestSurveyReportModel (2 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_optional_quality_fields_default_none` | PASS |
| 2 | `test_required_fields` | PASS |

---

## SurveyOrchestrator
**File:** `backend/tests/test_survey_orchestrator.py` &nbsp;|&nbsp; **Tests: 65**

### TestContextManagement (8 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_context_not_present_after_completion` | PASS |
| 2 | `test_context_reused_across_turns` | PASS |
| 3 | `test_drop_context_noop_for_unknown_session` | PASS |
| 4 | `test_drop_context_removes_cache_entry` | PASS |
| 5 | `test_get_conv_context_after_turn` | PASS |
| 6 | `test_get_conv_context_none_before_first_turn` | PASS |
| 7 | `test_new_context_created_after_drop` | PASS |
| 8 | `test_separate_sessions_have_separate_contexts` | PASS |

### TestGetSurveyOrchestratorSingleton (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_instance_is_none_initially` | PASS |
| 2 | `test_returns_survey_orchestrator` | PASS |
| 3 | `test_singleton_same_instance` | PASS |

### TestISCOMatchDataclass (2 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_fields_exist` | PASS |
| 2 | `test_is_dataclass` | PASS |

### TestNeutralFallback (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_action_is_continue` | PASS |
| 2 | `test_arabic_prompt_set` | PASS |
| 3 | `test_raw_text_preserved` | PASS |
| 4 | `test_returns_emotional_analysis` | PASS |
| 5 | `test_state_is_neutral` | PASS |

### TestProcessTurnAuditLogging (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_agent_decision_logged` | PASS |
| 2 | `test_audit_failure_does_not_abort_turn` | PASS |
| 3 | `test_interaction_logged` | PASS |

### TestProcessTurnBasic (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_empty_entities_list` | PASS |
| 2 | `test_no_isco_matches_when_no_job_entities` | PASS |
| 3 | `test_reply_from_conversation_manager` | PASS |
| 4 | `test_session_id_propagated` | PASS |
| 5 | `test_session_not_completed_by_default` | PASS |
| 6 | `test_state_is_string` | PASS |
| 7 | `test_validation_passed_when_no_collected_data` | PASS |

### TestProcessTurnCompletion (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_completion_logs_session_completed_event` | PASS |
| 2 | `test_conv_context_cleared_on_completion` | PASS |
| 3 | `test_hitl_review_called_with_session_id` | PASS |
| 4 | `test_quality_report_none_when_hitl_fails` | PASS |
| 5 | `test_quality_report_none_when_not_completed` | PASS |
| 6 | `test_quality_report_set_on_completion` | PASS |
| 7 | `test_session_completed_flag` | PASS |

### TestProcessTurnContextMemory (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_memory_failure_does_not_abort_turn` | PASS |
| 2 | `test_missing_fields_empty_on_memory_failure` | PASS |
| 3 | `test_missing_fields_returned` | PASS |
| 4 | `test_session_saved_to_memory` | PASS |
| 5 | `test_turn_appended_to_memory` | PASS |

### TestProcessTurnEmotional (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_ei_failure_returns_neutral_fallback` | PASS |
| 2 | `test_frustrated_end_action` | PASS |
| 3 | `test_neutral_emotional_state` | PASS |
| 4 | `test_stressed_emotional_state_propagated` | PASS |
| 5 | `test_survey_action_continue_by_default` | PASS |

### TestProcessTurnEntities (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_entities_serialised_as_dicts` | PASS |
| 2 | `test_isco_match_fields_mapped_correctly` | PASS |
| 3 | `test_job_title_entity_produces_isco_match` | PASS |
| 4 | `test_multiple_job_titles_produce_multiple_matches` | PASS |
| 5 | `test_non_job_entity_skipped` | PASS |

### TestProcessTurnISCOFailure (1 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_isco_failure_skipped_gracefully` | PASS |

### TestProcessTurnLanguageDetection (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_detection_preserved` | PASS |
| 2 | `test_arabic_lang_uses_arabic_adapted_prompt` | PASS |
| 3 | `test_code_switched_flag_propagated` | PASS |
| 4 | `test_context_language_updated_on_detection_change` | PASS |
| 5 | `test_english_detection_preserved` | PASS |
| 6 | `test_english_lang_uses_english_adapted_prompt` | PASS |
| 7 | `test_unknown_lang_falls_back_to_session_language` | PASS |

### TestProcessTurnValidation (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_no_validation_when_empty_collected_data` | PASS |
| 2 | `test_validation_error_falls_back_to_pass` | PASS |
| 3 | `test_validation_failure_propagated` | PASS |
| 4 | `test_validation_pass_propagated` | PASS |

### TestTurnResultDataclass (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_is_dataclass` | PASS |
| 2 | `test_quality_report_none_by_default` | PASS |
| 3 | `test_required_fields` | PASS |

---

## Survey Routes (API)
**File:** `backend/tests/test_survey_routes.py` &nbsp;|&nbsp; **Tests: 29**

### TestCompleteSession (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_already_completed_returns_409` | PASS |
| 2 | `test_marks_session_completed` | PASS |
| 3 | `test_not_found_returns_404` | PASS |

### TestCreateSession (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_missing_language_returns_422` | PASS |
| 2 | `test_no_auth_returns_403` | PASS |
| 3 | `test_returns_201_with_correct_fields` | PASS |
| 4 | `test_sets_user_id_to_current_user` | PASS |

### TestDeleteSession (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_delete_soft_deletes_session_responses_retained` | PASS |
| 2 | `test_deleted_session_returns_404` | PASS |
| 3 | `test_deletes_session` | PASS |
| 4 | `test_not_found_returns_404` | PASS |

### TestGetSession (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_not_found_returns_404` | PASS |
| 2 | `test_other_users_session_returns_404` | PASS |
| 3 | `test_returns_session` | PASS |

### TestListResponses (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_nonexistent_session_returns_404` | PASS |
| 2 | `test_returns_all_responses` | PASS |
| 3 | `test_returns_empty_list` | PASS |

### TestListSessions (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_does_not_return_other_users_sessions` | PASS |
| 2 | `test_returns_all_sessions_for_user` | PASS |
| 3 | `test_returns_empty_list_when_no_sessions` | PASS |
| 4 | `test_returns_sessions_newest_first` | PASS |

### TestSubmitResponse (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_completed_session_returns_409` | PASS |
| 2 | `test_creates_response_with_all_fields` | PASS |
| 3 | `test_creates_response_without_optional_fields` | PASS |
| 4 | `test_missing_required_fields_returns_422` | PASS |
| 5 | `test_nonexistent_session_returns_404` | PASS |

### TestUpdateResponse (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_nonexistent_response_returns_404` | PASS |
| 2 | `test_nonexistent_session_returns_404` | PASS |
| 3 | `test_updates_all_fields` | PASS |

---

## ValidationAgent
**File:** `backend/tests/test_validation_agent.py` &nbsp;|&nbsp; **Tests: 82**

### TestNormalise (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_casts_hours_float_to_int_string` | PASS |
| 2 | `test_lowercases_employment_status` | PASS |
| 3 | `test_lowercases_employment_type` | PASS |
| 4 | `test_none_values_become_empty_string` | PASS |
| 5 | `test_preserves_extra_keys` | PASS |
| 6 | `test_preserves_invalid_hours_as_string` | PASS |
| 7 | `test_strips_whitespace_from_values` | PASS |

### TestParseSemanticResponse (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_explanation_preserved` | PASS |
| 2 | `test_confidence_clamped_to_unit_interval` | PASS |
| 3 | `test_embedded_json_extracted` | PASS |
| 4 | `test_inconsistent_response_sets_is_valid_false` | PASS |
| 5 | `test_malformed_json_falls_back_to_valid` | PASS |
| 6 | `test_strips_markdown_fences` | PASS |
| 7 | `test_valid_json_parsed_correctly` | PASS |

### TestR01RequiredFields (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_fields_present_no_violations` | PASS |
| 2 | `test_employed_missing_job_title_is_error` | PASS |
| 3 | `test_empty_field_value_triggers_error` | PASS |
| 4 | `test_missing_employment_status_is_error` | PASS |
| 5 | `test_missing_industry_is_error` | PASS |
| 6 | `test_unemployed_missing_job_title_is_not_error` | PASS |

### TestR02HoursNumeric (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_absent_field_passes` | PASS |
| 2 | `test_empty_string_passes` | PASS |
| 3 | `test_float_string_passes` | PASS |
| 4 | `test_integer_string_passes` | PASS |
| 5 | `test_non_numeric_string_is_error` | PASS |

### TestR03HoursRange (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_exactly_168_passes` | PASS |
| 2 | `test_exactly_1_passes` | PASS |
| 3 | `test_negative_hours_is_error` | PASS |
| 4 | `test_non_numeric_skipped` | PASS |
| 5 | `test_normal_hours_passes` | PASS |
| 6 | `test_over_168_is_error` | PASS |
| 7 | `test_zero_hours_is_error` | PASS |

### TestR04HoursExtreme (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_absent_field_passes` | PASS |
| 2 | `test_exactly_80_passes` | PASS |
| 3 | `test_normal_hours_passes` | PASS |
| 4 | `test_over_80_is_warning` | PASS |

### TestR05R06UnemployedConsistency (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_employed_with_job_title_no_error` | PASS |
| 2 | `test_non_unemployed_status_skipped` | PASS |
| 3 | `test_unemployed_with_job_title_is_r05_error` | PASS |
| 4 | `test_unemployed_with_positive_hours_is_r06_error` | PASS |
| 5 | `test_unemployed_zero_hours_no_error` | PASS |

### TestR07NilfConsistency (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_employed_status_skipped` | PASS |
| 2 | `test_nilf_with_job_title_is_r07_error` | PASS |
| 3 | `test_nilf_without_job_title_no_error` | PASS |

### TestR08R09R10HoursTypeConsistency (7 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_employed_zero_hours_is_r10_error` | PASS |
| 2 | `test_full_time_40_hours_passes` | PASS |
| 3 | `test_full_time_low_hours_is_r08_warning` | PASS |
| 4 | `test_missing_employment_type_no_violations` | PASS |
| 5 | `test_out_of_range_hours_skipped` | PASS |
| 6 | `test_part_time_high_hours_is_r09_warning` | PASS |
| 7 | `test_part_time_low_hours_passes` | PASS |

### TestRuleViolationModel (2 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_rule_violation_fields_accessible` | PASS |
| 2 | `test_violation_severities_are_valid_strings` | PASS |

### TestValidateConfidence (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_confidence_never_above_one` | PASS |
| 2 | `test_confidence_never_below_zero` | PASS |
| 3 | `test_each_error_reduces_confidence` | PASS |
| 4 | `test_valid_response_has_high_confidence` | PASS |
| 5 | `test_warning_reduces_confidence_less_than_error` | PASS |

### TestValidateErrors (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_employed_zero_hours_fails` | PASS |
| 2 | `test_hours_out_of_range_fails` | PASS |
| 3 | `test_missing_employment_status_fails` | PASS |
| 4 | `test_non_numeric_hours_fails` | PASS |
| 5 | `test_unemployed_with_job_title_fails` | PASS |

### TestValidateLanguage (2 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_language_produces_arabic_explanation` | PASS |
| 2 | `test_english_language_produces_english_explanation` | PASS |

### TestValidateResultShape (9 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_confidence_in_unit_interval` | PASS |
| 2 | `test_explanation_ar_is_non_empty_string` | PASS |
| 3 | `test_explanation_en_is_non_empty_string` | PASS |
| 4 | `test_returns_validation_result` | PASS |
| 5 | `test_rule_violations_is_list` | PASS |
| 6 | `test_semantic_issues_is_list` | PASS |
| 7 | `test_valid_employed_response_is_valid` | PASS |
| 8 | `test_valid_unemployed_response_is_valid` | PASS |
| 9 | `test_validated_data_echoes_input_keys` | PASS |

### TestValidateSemanticPath (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_llm_failure_degrades_confidence` | PASS |
| 2 | `test_llm_inconsistency_sets_is_valid_false` | PASS |
| 3 | `test_llm_issues_appear_in_semantic_issues_list` | PASS |
| 4 | `test_semantic_stage_called_for_clean_responses` | PASS |
| 5 | `test_semantic_stage_skipped_when_errors_present` | PASS |

### TestValidateWarnings (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_extreme_hours_is_warning_not_error` | PASS |
| 2 | `test_full_time_low_hours_is_warning` | PASS |
| 3 | `test_part_time_high_hours_is_warning` | PASS |

---

## ValidationAgent (extended)
**File:** `backend/tests/test_validation_agent_extended.py` &nbsp;|&nbsp; **Tests: 59**

### TestArabicLanguageRules (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_arabic_explanation_returned` | PASS |
| 2 | `test_arabic_rule_violation_message` | PASS |
| 3 | `test_english_explanation_always_present` | PASS |
| 4 | `test_rtl_text_in_explanation` | PASS |

### TestConfidenceScoreValidation (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_confidence_bounded_0_to_1` | PASS |
| 2 | `test_multiple_errors_lower_confidence` | PASS |
| 3 | `test_perfect_data_high_confidence` | PASS |

### TestCrossRuleInteractions (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_error_fails_validation` | PASS |
| 2 | `test_first_error_does_not_hide_second` | PASS |
| 3 | `test_multiple_violations_reported` | PASS |
| 4 | `test_warning_does_not_fail_validation` | PASS |

### TestDataTypeHandling (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_dict_value_handled` | PASS |
| 2 | `test_float_hours_accepted` | PASS |
| 3 | `test_integer_hours_accepted` | PASS |
| 4 | `test_list_value_handled` | PASS |
| 5 | `test_none_hours_handled` | PASS |

### TestR01BoundaryValues (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_fields_empty_strings_fail` | PASS |
| 2 | `test_job_title_with_single_char_passes` | PASS |
| 3 | `test_missing_education_level_passes` | PASS |
| 4 | `test_single_space_job_title_fails` | PASS |

### TestR03BoundaryValues (11 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_hours_boundary[-1-False]` | PASS |
| 2 | `test_hours_boundary[0-False]` | PASS |
| 3 | `test_hours_boundary[0.5-False]` | PASS |
| 4 | `test_hours_boundary[1-True]` | PASS |
| 5 | `test_hours_boundary[167.5-True]` | PASS |
| 6 | `test_hours_boundary[168-True]` | PASS |
| 7 | `test_hours_boundary[168.1-True]` | PASS |
| 8 | `test_hours_boundary[169-False]` | PASS |
| 9 | `test_hours_boundary[200-False]` | PASS |
| 10 | `test_hours_boundary[40-True]` | PASS |
| 11 | `test_hours_boundary[80-True]` | PASS |

### TestR04ExtremeHours (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_extreme_hours_warning[100-True]` | PASS |
| 2 | `test_extreme_hours_warning[168-True]` | PASS |
| 3 | `test_extreme_hours_warning[79-False]` | PASS |
| 4 | `test_extreme_hours_warning[80-False]` | PASS |
| 5 | `test_extreme_hours_warning[81-True]` | PASS |

### TestR05R06UnemployedBoundary (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_unemployed_with_hours_fails` | PASS |
| 2 | `test_unemployed_with_none_hours_ok` | PASS |
| 3 | `test_unemployed_with_zero_hours_ok` | PASS |

### TestR08R09Boundary (9 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_r08_full_time_low_hours[15-full_time-True]` | PASS |
| 2 | `test_r08_full_time_low_hours[19-full_time-True]` | PASS |
| 3 | `test_r08_full_time_low_hours[20-full_time-False]` | PASS |
| 4 | `test_r08_full_time_low_hours[20-part_time-False]` | PASS |
| 5 | `test_r08_full_time_low_hours[40-full_time-False]` | PASS |
| 6 | `test_r09_part_time_high_hours[34-part_time-False]` | PASS |
| 7 | `test_r09_part_time_high_hours[35-part_time-True]` | PASS |
| 8 | `test_r09_part_time_high_hours[40-full_time-False]` | PASS |
| 9 | `test_r09_part_time_high_hours[40-part_time-True]` | PASS |

### TestR10EmployedZeroHours (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_employed_with_positive_hours_ok` | PASS |
| 2 | `test_employed_with_zero_hours_fails` | PASS |
| 3 | `test_self_employed_zero_hours_fails` | PASS |

### TestSemanticValidation (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_llm_inconsistency_reported` | PASS |
| 2 | `test_semantic_check_runs_for_clean_data` | PASS |
| 3 | `test_semantic_check_skipped_when_errors_present` | PASS |
| 4 | `test_semantic_llm_failure_propagates` | PASS |

### TestWageValidation (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_negative_wage_handled` | PASS |
| 2 | `test_valid_wage_range_passes` | PASS |
| 3 | `test_very_high_wage_passes` | PASS |
| 4 | `test_zero_wage_employed` | PASS |

---

## VectorStore
**File:** `backend/tests/test_vector_store.py` &nbsp;|&nbsp; **Tests: 29**

### TestEnsureCollection (3 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_creates_collection_when_absent` | PASS |
| 2 | `test_recreate_deletes_then_creates` | PASS |
| 3 | `test_skips_creation_when_collection_exists` | PASS |

### TestEnsurePopulated (4 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_embed_called_with_all_isco_texts` | PASS |
| 2 | `test_skips_upsert_when_already_populated` | PASS |
| 3 | `test_upserts_all_isco_entries` | PASS |
| 4 | `test_upserts_when_collection_is_empty` | PASS |

### TestIscoDataIntegrity (6 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_codes_unique` | PASS |
| 2 | `test_all_entries_have_required_keys` | PASS |
| 3 | `test_codes_are_non_empty_strings` | PASS |
| 4 | `test_dataset_contains_unit_groups` | PASS |
| 5 | `test_dataset_has_all_major_groups` | PASS |
| 6 | `test_levels_are_valid` | PASS |

### TestSearch (11 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_confidence_clamped_above_one` | PASS |
| 2 | `test_confidence_clamped_below_zero` | PASS |
| 3 | `test_confidence_rounded_to_4_decimal_places` | PASS |
| 4 | `test_default_top_k_is_five` | PASS |
| 5 | `test_empty_query_returns_empty_list` | PASS |
| 6 | `test_multiple_results_returned` | PASS |
| 7 | `test_query_prefixed_with_query_tag` | PASS |
| 8 | `test_result_fields_match_payload` | PASS |
| 9 | `test_returns_occupation_match_objects` | PASS |
| 10 | `test_top_k_passed_to_qdrant` | PASS |
| 11 | `test_whitespace_query_returns_empty_list` | PASS |

### TestStableId (5 tests)

| # | Test | Status |
|---|---|---|
| 1 | `test_all_isco_codes_have_unique_ids` | PASS |
| 2 | `test_different_codes_give_different_ids` | PASS |
| 3 | `test_fits_in_uint64` | PASS |
| 4 | `test_returns_non_negative_integer` | PASS |
| 5 | `test_same_code_gives_same_id` | PASS |

---

## Notes

- All 1,178 collected tests pass with exit code 0 (`1178 passed, 1 deselected, 1 warning in 340.50s`).
- The single warning is `InsecureKeyLengthWarning` from PyJWT on a deliberately short test-fixture HMAC secret (`backend/tests/test_auth_and_api_extended.py::TestJWTEdgeCases::test_jwt_with_wrong_secret_rejected`) — expected for that edge-case test, not a production concern.
- Teardown noise: a harmless `ValueError: I/O operation on closed file` is printed from a CrewAI/colorama `atexit` hook after the run completes. Exit code is still 0 — not a test failure.
- Zero live infrastructure required: PostgreSQL is replaced by SQLite in-memory, Redis by a `FakeRedis` class, and Qdrant/LLM/CrewAI calls are mocked via `unittest.mock`.
- `backend/tests/load_test.py` is excluded by default (`@pytest.mark.slow`, filtered by `pytest.ini`); it requires a running backend at `localhost:8000` and is run explicitly with `pytest backend/tests/load_test.py -m slow -v`.

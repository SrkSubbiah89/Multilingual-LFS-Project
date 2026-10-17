"""Independent canonical questionnaire records for completion regressions.

These examples spell out the current required questionnaire answers. They do
not derive expectations from the production required-field helper. Ordered
optional questions are deliberately omitted.
"""


def required_answers(status="employed", **overrides):
    common = {
        "education_level": "secondary",
        "gender": "male",
        "nationality": "indian",
        "marital_status": "single",
        "emirate": "dubai",
        "main_skills": "communication",
        "platform_work": "no",
        "question_clarity": "5",
        "ai_preference": "prefer_ai",
        "data_confidence": "very_confident",
    }
    paths = {
        "employed": {
            "employment_status": "employed",
            "employment_nature": "paid_employee",
            "employment_sector": "private",
            "job_title": "software engineer",
            "job_duties": "writes and tests application software",
            "industry": "technology",
            "hours_per_week": "40",
            "employment_type": "full_time",
            "monthly_wage_range": "10001_20000",
            "job_satisfaction": "4",
        },
        "unemployed": {
            "employment_status": "unemployed",
            "job_search_active": "yes",
            "available_for_work": "yes",
            "unemployment_duration": "1_to_6_months",
            "ever_worked": "never_worked",
        },
        "not_in_labour_force": {
            "employment_status": "not_in_labour_force",
            "outside_lf_reason": "studying",
            "ever_worked": "never_worked",
        },
    }
    if status not in paths:
        raise ValueError(f"Unsupported questionnaire path: {status}")
    return {**common, **paths[status], **overrides}

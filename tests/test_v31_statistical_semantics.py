from voiceprobe.v3.statistical_semantics import StatisticalIntentScorer


def test_statistical_semantics_core_paraphrases() -> None:
    scorer = StatisticalIntentScorer()
    cases = (
        ("visit_reason_request", "Why are you coming in?", "visit_reason"),
        (
            "visit_reason_request",
            "Can you tell me why you need the appointment?",
            "visit_reason",
        ),
        (
            "visit_reason_request",
            "What specific concern are you being seen for?",
            "visit_reason",
        ),
        (
            "appointment_type_request",
            "What kind of visit are we scheduling?",
            "appointment_type",
        ),
        (
            "insurance_request",
            "Which insurance carrier do you use?",
            "insurance",
        ),
        ("dob_request", "What's your DOB?", "dob"),
        ("last_name_request", "What is your surname?", "identity"),
        (
            "visit_reason_request",
            "What is the reason for your appointment?",
            "visit_reason",
        ),
    )

    for expected, text, stage in cases:
        result = scorer.classify(text, stage=stage)
        assert result.intent == expected, (text, result)
        assert scorer.accepts(result), (text, result)


def test_statistical_semantics_rejects_out_of_domain_requests() -> None:
    scorer = StatisticalIntentScorer()

    for text in (
        "What is your home address?",
        "Which pharmacy do you use?",
        "Do you need directions to the clinic?",
        "I have a question about billing.",
    ):
        result = scorer.classify(text, stage="visit_reason")
        assert result.intent == "unknown", (text, result)

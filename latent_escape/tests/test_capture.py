from latent_escape.capture_activations import _align_generated_ids_at_boundary


def test_boundary_alignment_accepts_only_late_retokenization_drift() -> None:
    aligned, status = _align_generated_ids_at_boundary(
        [10, 11, 12, 99], [10, 11, 12, 98], domain_index=1, special_ids=set()
    )

    assert aligned == [10, 11, 12, 99]
    assert status == "domain_boundary_prefix_verified"


def test_boundary_alignment_rejects_drift_at_domain_token() -> None:
    aligned, status = _align_generated_ids_at_boundary(
        [10, 42, 12], [10, 11, 12], domain_index=1, special_ids=set()
    )

    assert aligned is None
    assert status == "stored_token_ids_do_not_match_boundary_tokenization"


def test_boundary_alignment_strips_terminal_special_tokens() -> None:
    aligned, status = _align_generated_ids_at_boundary(
        [10, 11, 1], [10, 11], domain_index=1, special_ids={1}
    )

    assert aligned == [10, 11]
    assert status == "terminal_specials_stripped"

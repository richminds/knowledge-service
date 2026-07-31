from rag.log_context import bind_request_id, get_request_id, request_id_scope, reset_request_id


def test_get_request_id_defaults_to_empty():
    assert get_request_id() == ""


def test_bind_and_reset_request_id():
    token = bind_request_id("abc123")
    try:
        assert get_request_id() == "abc123"
    finally:
        reset_request_id(token)
    assert get_request_id() == ""


def test_request_id_scope_mints_one_when_absent():
    with request_id_scope() as request_id:
        assert request_id
        assert get_request_id() == request_id
    assert get_request_id() == ""


def test_request_id_scope_uses_given_id():
    with request_id_scope("fixed-id") as request_id:
        assert request_id == "fixed-id"
        assert get_request_id() == "fixed-id"

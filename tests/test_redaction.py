from evalforge.ingestion.redaction import redact


def test_email_caught():
    r = redact("contact me at alice@example.com please")
    assert "[REDACTED_EMAIL]" in r.text
    assert r.counts.get("EMAIL") == 1
    assert "alice@example.com" not in r.text


def test_credit_card_caught():
    r = redact("card 4111-1111-1111-1111 expires soon")
    assert "[REDACTED_CARD]" in r.text
    assert r.counts.get("CARD") == 1


def test_phone_caught():
    r = redact("ring me at (555) 123-4567")
    assert "[REDACTED_PHONE]" in r.text


def test_account_id_caught():
    r = redact("look up ACCT-ABC123XYZ for details")
    assert "[REDACTED_ACCOUNT]" in r.text


def test_invoice_id_caught():
    r = redact("invoice INV-7K3M9P is overdue")
    assert "[REDACTED_INVOICE]" in r.text


def test_clean_text_unchanged():
    original = "the quick brown fox jumps over the lazy dog"
    r = redact(original)
    assert r.text == original
    assert r.counts == {}


def test_multiple_patterns_in_one_string():
    r = redact("email a@b.co and ACCT-XYZ12345 and card 5555 5555 5555 4444")
    assert r.counts.get("EMAIL") == 1
    assert r.counts.get("ACCOUNT") == 1
    assert r.counts.get("CARD") == 1

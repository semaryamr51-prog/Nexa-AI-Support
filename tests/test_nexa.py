import re
from pathlib import Path
from src.agents import route_and_run
from src.classify import _rules, classify
from src.data_gen import generate_splits
from src.knowledge import search
from src.mock_backends import mock_billing_db, mock_calendar


def test_deduplication_and_template_split():
    splits = generate_splits()
    assert all(splits.values())
    seen = set()
    for rows in splits.values():
        for text, label in rows:
            assert text.casefold() not in seen
            seen.add(text.casefold())
            assert label in {'billing', 'technical', 'general', 'booking'}


def test_clear_keyword_routes():
    assert _rules('I need a refund for my invoice')[0] == 'billing'
    assert _rules('The video crashes on login')[0] == 'technical'
    assert _rules('Book a mentoring session')[0] == 'booking'


def test_unknown_invoice_not_invented():
    result = route_and_run('Check invoice INV-9999 please')
    assert 'No demo invoice found' in result['reply']
    assert 'check_invoice' in result['tool_calls']


def test_demo_bookings_cannot_reserve():
    result = route_and_run('Can I book office hours next Monday?')
    assert result['agent'] == 'booking'
    assert 'NOT live' in result['reply']
    assert 'check_availability' in result['tool_calls']


def test_knowledge_returns_grounded_sources():
    hits = search('refund policy', 'billing')
    assert hits and any('billing_faq.md' in p.source for p in hits)


def test_safe_no_model_needed(monkeypatch, tmp_path):
    import src.classify as classify_module
    monkeypatch.setattr(classify_module, 'MODEL_PATH', tmp_path / 'missing.pt')
    label, conf = classify_module.classify('Why is my payment delayed?')
    assert label == 'billing' and 0 < conf <= 1


def test_no_old_branding_in_interface():
    app = Path(__file__).resolve().parent.parent.joinpath('app.py').read_text(encoding='utf-8')
    for forbidden in ('🎓', 'Edges Training', 'EdgeCourse', 'edges training'):
        assert forbidden not in app

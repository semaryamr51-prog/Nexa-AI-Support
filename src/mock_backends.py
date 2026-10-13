"""Predictable fictional backends. Never creates a real charge or booking."""
from __future__ import annotations

_INVOICES = {
    "INV-2091": {"customer": "Amr", "amount": "$49.99", "status": "paid", "course": "Intro to Python"},
    "INV-3345": {"customer": "Sara", "amount": "$120.00", "status": "refunded", "course": "Machine Learning 101"},
    "INV-8820": {"customer": "Youssef", "amount": "$15.00", "status": "unpaid", "course": "Advanced SQL"},
    "INV-1187": {"customer": "Mona", "amount": "$249.00", "status": "paid", "course": "Deep Learning Foundations"},
    "INV-5502": {"customer": "Karim", "amount": "$9.99", "status": "paid", "course": "React from Scratch"},
}


class MockBillingDB:
    def get_invoice(self, invoice_id: str) -> str:
        invoice_id = invoice_id.strip().upper()
        record = _INVOICES.get(invoice_id)
        if not record:
            return f"No demo invoice found for {invoice_id}. No real account was queried."
        return (f"Demo invoice {invoice_id}: course={record['course']}, "
                f"amount={record['amount']}, status={record['status']}. "
                "This is simulated data, not a real billing account.")


class MockCalendar:
    def check(self, date: str) -> str:
        key = date.strip().lower()
        if not key:
            return "Please specify a day/date to check demo availability. No booking was created."
        if any(v in key for v in ("monday", "friday")):
            return (f"Demo calendar for {date}: marked fully booked. Example alternatives: "
                    "Tuesday 3 PM or Wednesday 11 AM. These are NOT live slots and cannot be reserved here.")
        return (f"Demo calendar for {date}: example times 10 AM, 1 PM, 4 PM. "
                "These are NOT live slots and no reservation has been made.")


mock_billing_db = MockBillingDB()
mock_calendar = MockCalendar()

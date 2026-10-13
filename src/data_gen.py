"""
data_gen.py
-----------
Generates a synthetic, domain-specific labeled dataset for the Nexa Learning Demo
support-bot intent classifier (billing / technical / booking / general).

Why synthetic instead of Banking77/CLINC150?
This project targets an *online course platform* helpdesk, which those public
datasets don't model well out of the box. Per the project brief (section 5.2),
LLM/template-generated domain examples are an explicitly recommended add-on.
Here they're the primary source (fully offline, reproducible, no API key
needed), which also keeps `python -m src.data_gen` runnable with zero
dependencies beyond the standard library.

Run:
    python -m src.data_gen
Produces:
    data/intents_dataset.csv  (all examples, shuffled)
    data/train.csv  data/val.csv  data/test.csv   (template-grouped 70/15/15 split)
"""

import csv
import random
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COURSES = ["Intro to Python", "Data Structures", "Machine Learning 101",
           "Web Development Bootcamp", "Cloud Computing Basics",
           "Advanced SQL", "React from Scratch", "Deep Learning Foundations"]
NAMES = ["Amr", "Sara", "Youssef", "Mona", "Karim", "Laila", "Omar", "Nourhan"]
INVOICES = ["INV-2091", "INV-3345", "INV-8820", "INV-1187", "INV-5502"]
AMOUNTS = ["$49.99", "$120", "$15", "$249", "$9.99"]
DATES = ["next Monday", "March 14th", "this Friday", "tomorrow at 5 PM",
         "the 22nd", "next week", "Thursday morning"]
ERRORS = ["Error 403", "a blank white screen", "'video failed to load'",
          "the page freezing on submit", "a 500 server error",
          "the quiz not saving my answers", "the app crashing on login"]
DEVICES = ["Chrome on Windows", "my iPhone", "Safari on Mac", "the Android app",
           "an iPad", "Firefox"]

# ---------------------------------------------------------------------------
# BILLING intent
# ---------------------------------------------------------------------------
billing_templates = [
    "I was charged {amount} but I never confirmed the purchase, can you check invoice {inv}?",
    "My card was billed twice for {course}, invoice number {inv}.",
    "Can you tell me why invoice {inv} shows {amount} instead of the price I saw?",
    "I need a refund for {course}, order {inv}.",
    "How do I update the credit card on file for my subscription?",
    "When does my subscription renew and how much will it cost?",
    "I cancelled last month but I was still charged {amount}.",
    "Can I get a receipt for invoice {inv}?",
    "Is there a student discount available for {course}?",
    "My payment for {course} failed, what should I do?",
    "I want to downgrade my plan, will I get a partial refund?",
    "Why is my invoice {inv} still marked as unpaid?",
    "Do you offer installment payments for {course}?",
    "I was double-charged this month, please check my account.",
    "Can you apply my coupon code retroactively to invoice {inv}?",
    "What payment methods do you accept?",
    "I need an official invoice for my company reimbursement, order {inv}.",
    "The price for {course} changed after I enrolled, can you explain?",
    "How do I cancel my subscription before the next billing cycle?",
    "Please check the status of my refund request for {amount}.",
]

# ---------------------------------------------------------------------------
# TECHNICAL intent
# ---------------------------------------------------------------------------
technical_templates = [
    "I'm getting {error} when I try to open {course} on {device}.",
    "The videos in {course} won't play, I just see {error}.",
    "I can't log in to my account from {device}, it keeps showing {error}.",
    "My progress in {course} isn't saving, is this a known bug?",
    "The code editor for {course} is stuck loading on {device}.",
    "I keep getting {error} whenever I submit an assignment.",
    "The mobile app crashes every time I open {course} on {device}.",
    "How do I reset my password? The reset email never arrives.",
    "Downloads for {course} materials are failing with {error}.",
    "The certificate for {course} won't generate after I finished it.",
    "I'm seeing {error} on the checkout page, but this isn't about payment.",
    "Two-factor authentication isn't sending me a code on {device}.",
    "The captions are out of sync in {course} videos.",
    "My screen just shows {error} whenever I try to join a live session.",
    "The quiz timer freezes on {device} during {course}.",
    "I can't upload my project file, it says {error}.",
    "Is there a known issue with {device} and the course player?",
    "The dark mode setting keeps resetting every time I reopen {device}.",
    "My account shows the wrong course list after I switched to {device}.",
    "Live chat support widget won't open, I only get {error}.",
]

# ---------------------------------------------------------------------------
# BOOKING intent
# ---------------------------------------------------------------------------
booking_templates = [
    "Can I book office hours with the instructor for {course} {date}?",
    "I'd like to schedule a 1:1 mentoring session {date}.",
    "Is there a live Q&A slot available {date} for {course}?",
    "Can you check if {date} is open for a tutoring session?",
    "I need to reschedule my session that was set for {date}.",
    "How do I book a seat in the {course} live workshop?",
    "Are there any open slots for career coaching {date}?",
    "I want to cancel my booking for {date}, is that possible?",
    "Can you confirm my appointment with the TA {date}?",
    "What times are available this week for {course} help sessions?",
    "I'd like to move my mentoring call to {date} instead.",
    "Is the study group for {course} meeting {date}?",
    "Please check availability for a make-up exam session {date}.",
    "Can I reserve a spot in the {date} cohort kickoff call?",
    "I missed my slot, can we rebook for {date}?",
    "How far in advance do I need to book office hours?",
    "Is {date} still free for a portfolio review session?",
    "I want to schedule my capstone presentation for {date}.",
    "Can you check if the instructor is available {date}?",
    "Book me a reminder call before {course} starts {date}.",
]

# ---------------------------------------------------------------------------
# GENERAL intent
# ---------------------------------------------------------------------------
general_templates = [
    "What topics are covered in {course}?",
    "How long does it typically take to finish {course}?",
    "Do I get a certificate after completing {course}?",
    "Is {course} suitable for complete beginners?",
    "What are the prerequisites for {course}?",
    "Can I access {course} materials after the course ends?",
    "Who should I contact for accessibility accommodations?",
    "Do you have a mobile app for offline learning?",
    "What's the difference between the free and paid version of {course}?",
    "Can I get feedback on my assignments from a real instructor?",
    "How do I contact support if none of these options fit?",
    "Is there a community forum for {course} students?",
    "What's your policy on course completion deadlines?",
    "Can I switch from {course} to a different course later?",
    "Do you offer team or corporate plans?",
    "How do I update my email address on file?",
    "What languages is {course} available in?",
    "Can I get a transcript of my course history?",
    "Is my data private, and what's your privacy policy?",
    "Just wanted to say thanks, the {course} content was great!",
]


def fill(template: str, rng: random.Random) -> str:
    return template.format(
        course=rng.choice(COURSES), name=rng.choice(NAMES),
        inv=rng.choice(INVOICES), amount=rng.choice(AMOUNTS),
        date=rng.choice(DATES), error=rng.choice(ERRORS),
        device=rng.choice(DEVICES),
    )


def generate_splits(n_per_template: int = 18, seed: int = 42):
    """Group by SOURCE TEMPLATE first: no near-identical pattern in multiple splits.

    Removes exact text duplicates globally. Kept examples are synthetic; tests
    measure held-out template families, not a representative real-world sample.
    """
    rng = random.Random(seed)
    labels_and_templates = [
        ("billing", billing_templates), ("technical", technical_templates),
        ("booking", booking_templates), ("general", general_templates),
    ]
    output = {"train": [], "val": [], "test": []}
    seen = set()
    for label, templates in labels_and_templates:
        templates = list(templates)
        rng.shuffle(templates)
        n_train = int(len(templates) * 0.70)
        n_val = int(len(templates) * 0.15)
        partitions = {"train": templates[:n_train],
                      "val": templates[n_train:n_train + n_val],
                      "test": templates[n_train + n_val:]}
        for split, selected in partitions.items():
            for template in selected:
                local = set()
                for _ in range(n_per_template * 12):
                    if len(local) >= n_per_template:
                        break
                    sentence = fill(template, rng)
                    norm = " ".join(sentence.casefold().split())
                    if norm in seen:
                        continue
                    seen.add(norm)
                    local.add(sentence)
                output[split].extend((sentence, label) for sentence in sorted(local))
    for rows in output.values():
        rng.shuffle(rows)
    return output


def write_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    splits = generate_splits()
    for split, rows in splits.items():
        write_csv(DATA_DIR / f"{split}.csv", rows)
    all_rows = [row for rows in splits.values() for row in rows]
    write_csv(DATA_DIR / "intents_dataset.csv", all_rows)
    from collections import Counter
    print(f"Total unique messages: {len(all_rows)}")
    for split, rows in splits.items():
        print(f"{split}: {len(rows)} | {dict(Counter(label for _, label in rows))}")
    assert len({text.casefold() for text, _ in all_rows}) == len(all_rows), "Duplicate leakage"


if __name__ == "__main__":
    main()

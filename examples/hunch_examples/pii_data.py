"""PII windows from gretel (synthetic_pii_finance_multilingual): each window with its PII values, a twin with every
value replaced by a phrase naming its type, and one with the values masked."""

import json
import re

import pyarrow.parquet as pq

from . import GRETEL, find

# label -> (category, phrase that names the type without giving a value)
GRETEL_TYPES = {
    "name": ("person", "the customer"), "first_name": ("person", "the customer"), "last_name": ("person", "the customer"),
    "email": ("contact", "the customer's email address"), "phone_number": ("contact", "the customer's phone number"),
    "street_address": ("address", "the customer's home address"),
    "iban": ("financial", "the customer's account number"), "bban": ("financial", "the customer's account number"),
    "credit_card_number": ("financial", "the customer's card number"),
    "credit_card_security_code": ("financial", "the card's security code"),
    "bank_routing_number": ("financial", "the bank's routing number"), "account_pin": ("financial", "the account PIN"),
    "ssn": ("gov_id", "the customer's social security number"), "passport_number": ("gov_id", "the customer's passport number"),
    "driver_license_number": ("gov_id", "the customer's driver's license number"),
    "password": ("credential", "the user's password"), "api_key": ("credential", "the API key"),
    "user_name": ("credential", "the user's login name"), "ipv4": ("credential", "the server's IP address"),
    "ipv6": ("credential", "the server's IP address"),
}
GRETEL_AMBIGUOUS = {"customer_id", "employee_id", "date_of_birth", "local_latlng"}
LOOKS_LIKE_PII = re.compile(r"[\w.+-]+@[\w-]+\.\w+|\+?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}")


def windows(text, maxlen=320, minlen=40):
    """(start, end) of windows made of whole lines, long lines split at sentence ends."""
    pieces = []
    for m in re.finditer(r"[^\n]+", text):
        a = m.start()
        for s in re.split(r"(?<=[.!?])\s+", m.group()):
            i = text.index(s, a)
            pieces.append((i, i + len(s)))
            a = i + len(s)
    out, cur = [], None
    for a, b in pieces:
        if b - a > maxlen:
            if cur:
                out.append(cur)
            cur = None
        elif cur and b - cur[0] <= maxlen:
            cur = (cur[0], b)
        else:
            if cur:
                out.append(cur)
            cur = (a, b)
    if cur:
        out.append(cur)
    return [(a, b) for a, b in out if len(text[a:b].strip()) >= minlen]


def units(docs, types, ambiguous):
    """docs: (doc_id, group, text, [(start, end, label)]) -> one row per usable window."""
    rows = []
    for doc, group, text, spans in docs:
        for a, b in windows(text):
            if any(s < b and e > a and not (a <= s and e <= b) for s, e, _ in spans):
                continue  # a span crosses the window edge
            inside = [(s, e, lab) for s, e, lab in spans if a <= s and e <= b and text[s:e].strip()]
            if any(lab in ambiguous for _, _, lab in inside):
                continue
            pii = sorted([(s, e, lab) for s, e, lab in inside if lab in types])
            w = text[a:b]
            typed = neutral = w
            for s, e, lab in reversed(pii):
                typed = typed[: s - a] + types[lab][1] + typed[e - a:]
                neutral = neutral[: s - a] + "***" + neutral[e - a:]
            rows.append({"doc": doc, "group": group, "text": w, "typed": typed, "neutral": neutral,
                         "cats": sorted({types[lab][0] for _, _, lab in pii}), "n_pii": len(pii),
                         "spans": [(s - a, e - a, types[lab][0]) for s, e, lab in pii]})
    return rows


def load_gretel():
    docs = []
    for i, r in enumerate(pq.read_table(find(GRETEL, "English_test-00000-of-00001.parquet")).to_pylist()):
        spans = [(s["start"], s["end"], s["label"]) for s in json.loads(r["pii_spans"])]
        docs.append((f"g{i}", r["document_type"], r["generated_text"], spans))
    return units(docs, GRETEL_TYPES, GRETEL_AMBIGUOUS)



HARD_NEGATIVES = [  # mention PII fields, or entity values that are not personal data, but contain no PII value
    "Please confirm the customer's email address and phone number before closing the ticket.",
    "Counterparty: Goldman Sachs International. Trade date: 2023-04-12. Settlement: T+2.",
    "Product identifier: HY-123. CAS No.: 123456789. Store in a cool, dry place.",
    "Never share your password, PIN or social security number with anyone, including bank staff.",
    "Field 59: Beneficiary Customer. Field 70: Remittance Information. Field 71A: Details of Charges.",
    "The account number must be 8 to 12 digits; the routing number is always 9 digits.",
]

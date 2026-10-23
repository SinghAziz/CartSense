"""Deterministic running bill from accepted cart contents (integer cents). No payments."""
def compute_bill(session, catalog, unit_price_cents=None):
    lines, subtotal, warnings = [], 0, []
    for sku, qty in sorted(session.items.items()):
        p = catalog.get(sku)
        if p is None:
            warnings.append(f"{sku}: not in catalogue, unpriced")
            lines.append(dict(sku_id=sku, name="(unknown SKU)", qty=qty, unit_cents=None, line_cents=None))
            continue
        unit_cents = unit_price_cents if unit_price_cents is not None else p.price_cents
        lt = unit_cents * qty
        subtotal += lt
        lines.append(dict(sku_id=sku, name=p.name, qty=qty, unit_cents=unit_cents, line_cents=lt))
    pend = session.pending()
    if pend:
        warnings.append(f"{len(pend)} event(s) need review - not included in bill")
    if catalog.is_mock:
        warnings.append("Catalogue prices are MOCK sample data")
    unpriced = any(l["line_cents"] is None for l in lines)
    status = "PROVISIONAL" if (pend or unpriced) else "CONFIRMED_UNPAID"
    return dict(lines=lines, subtotal_cents=subtotal, currency=catalog.meta.get("currency", ""),
                status=status, warnings=warnings, payment_processed=False,
                unresolved_events=[e.event_id for e in pend])

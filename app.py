"""Minimal CartSense dashboard:  streamlit run app.py"""
import os, tempfile, cv2, numpy as np, pandas as pd, streamlit as st
from src.setup import build, load_config
from src.pricing.billing import compute_bill

st.set_page_config(page_title="CartSense prototype", layout="wide")
st.title("CartSense ML prototype")
st.warning("Prototype. Generic COCO detector (not SKU recognition). Catalogue prices are MOCK data. No payment is processed.")

@st.cache_resource
def get_pipe():
    return build(load_config())
pipe, cat, sess = get_pipe()
if "last" not in st.session_state: st.session_state.last = None

with st.sidebar:
    st.header("Input")
    up = st.file_uploader("Image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"])
    cam = st.camera_input("...or webcam snapshot (single image)")
    run = st.button("Run inference", type="primary")
    if st.button("Reset demo session"):
        sess.reset(); st.session_state.last = None; st.rerun()
    st.caption(f"Session {sess.session_id} | providers {pipe.det.providers} | previous sessions: {len(sess.history)}")

if run and (up or cam):
    with st.spinner("Processing..."):
        if cam and not up:
            img = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            st.session_state.last = ("image", pipe.process_image(img))
        elif up.type.startswith("image"):
            img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            st.session_state.last = ("image", pipe.process_image(img))
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]); tmp.write(up.getvalue()); tmp.close()
            out = os.path.join("outputs", "annotated_ui.mp4"); os.makedirs("outputs", exist_ok=True)
            try:
                st.session_state.last = ("video", pipe.process_video(tmp.name, out))
            finally:
                os.unlink(tmp.name)

c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Annotated output")
    if st.session_state.last:
        kind, r = st.session_state.last
        if kind == "image":
            st.image(cv2.cvtColor(r["annotated"], cv2.COLOR_BGR2RGB))
            st.caption("Single image: identification only. Cart events need temporal evidence (video).")
            st.dataframe(pd.DataFrame([dict(label=d.label, det_conf=round(d.confidence, 3), status=i.status, sku=i.sku_id, reason=i.reason)
                                       for d, i in zip(r["detections"], r["identifications"])]))
            for bc, i in r["orphan_barcodes"]: st.write(f"Barcode `{bc.value}` ({bc.symbology}) -> {i.status} {i.sku_id or ''} - {i.reason}")
            st.write("Latency ms:", {k: round(v, 1) for k, v in r["latency_ms"].items()})
        else:
            st.write(f"{r['frames']} frames processed. Annotated video saved to `{r['out_path']}` (open it locally).")
            with st.expander("Per-instance inventory diagnostics"):
                st.json(r.get("inventory", {"status": "not available"}))
            tot = r["latency_ms"]["total"]
            if tot: st.write(f"Per-frame latency ms: median {np.median(tot):.1f}, p95 {np.percentile(tot, 95):.1f} (this machine)")
    else:
        st.info("Load an image/video and press Run inference.")
    st.subheader("Manual add (e.g. from a single image)")
    m1, m2, m3 = st.columns(3)
    sku = m1.selectbox("SKU", list(cat.products), format_func=lambda s: f"{s} {cat.get(s).name}")
    typ = m2.selectbox("Type", ["ADD", "REMOVE"]); why = m3.text_input("Reason", "operator confirmed visually")
    if st.button("Create manual event"):
        sess.manual_event(typ, sku, why); st.rerun()
with c2:
    st.subheader("Digital cart")
    b = compute_bill(sess, cat)
    st.dataframe(pd.DataFrame(b["lines"]) if b["lines"] else pd.DataFrame(columns=["sku_id", "name", "qty", "unit_cents", "line_cents"]))
    st.metric("Subtotal" + (" (PROVISIONAL)" if b["status"] == "PROVISIONAL" else ""), f"{b['subtotal_cents']/100:.2f} {b['currency']}")
    for w in b["warnings"]: st.warning(w)
    st.subheader("Needs review")
    for e in sess.pending():
        with st.expander(f"{e.event_id} {e.event_type} track={e.track_id} sku={e.sku_id}"):
            st.write(e.evidence)
            s2 = st.selectbox("SKU", list(cat.products), key="s" + e.event_id, index=list(cat.products).index(e.sku_id) if e.sku_id in cat.products else 0)
            t2 = st.selectbox("Type", ["ADD", "REMOVE"], key="t" + e.event_id, index=1 if e.event_type == "REMOVE" else 0)
            rs = st.text_input("Reason (required)", key="r" + e.event_id)
            a, bb = st.columns(2)
            if a.button("Confirm / correct", key="c" + e.event_id) and rs: sess.confirm(e.event_id, rs, s2, t2); st.rerun()
            if bb.button("Reject", key="x" + e.event_id) and rs: sess.reject(e.event_id, rs); st.rerun()

st.subheader("Events")
st.dataframe(pd.DataFrame([{k: v for k, v in e.to_dict().items() if k not in ("state_history", "dedupe_key")} for e in sess.events.values()]))
with st.expander("Audit log"): st.dataframe(pd.DataFrame(sess.audit))

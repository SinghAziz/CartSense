"""CartSense Streamlit dashboard. Launch with ``python -m streamlit run app.py``."""

import os, tempfile, cv2, numpy as np, pandas as pd, streamlit as st
from src.setup import build, load_config
from src.pricing.billing import compute_bill

st.set_page_config(page_title="CartSense prototype", layout="wide")
st.title("CartSense ML prototype")
st.warning(
    "Prototype. Generic COCO detector (not SKU recognition). Catalogue prices are MOCK data. No payment is processed."
)


@st.cache_resource
def get_pipeline():
    return build(load_config())


pipeline, catalog, cart_session = get_pipeline()
if "last" not in st.session_state:
    st.session_state.last = None

with st.sidebar:
    st.header("Input")
    uploaded_file = st.file_uploader(
        "Image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv"]
    )
    camera_snapshot = st.camera_input("...or webcam snapshot (single image)")
    run_inference = st.button("Run inference", type="primary")
    if st.button("Reset demo session"):
        cart_session.reset()
        st.session_state.last = None
        st.rerun()
    st.caption(
        f"Session {cart_session.session_id} | providers {pipeline.detector.providers} | previous sessions: {len(cart_session.history)}"
    )

if run_inference and (uploaded_file or camera_snapshot):
    with st.spinner("Processing..."):
        if camera_snapshot and not uploaded_file:
            image = cv2.imdecode(
                np.frombuffer(camera_snapshot.getvalue(), np.uint8), cv2.IMREAD_COLOR
            )
            st.session_state.last = ("image", pipeline.process_image(image))
        elif uploaded_file.type.startswith("image"):
            image = cv2.imdecode(
                np.frombuffer(uploaded_file.getvalue(), np.uint8), cv2.IMREAD_COLOR
            )
            st.session_state.last = ("image", pipeline.process_image(image))
        else:
            temporary_video_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
            )
            temporary_video_file.write(uploaded_file.getvalue())
            temporary_video_file.close()
            annotated_video_path = os.path.join("outputs", "annotated_ui.mp4")
            os.makedirs("outputs", exist_ok=True)
            try:
                st.session_state.last = (
                    "video",
                    pipeline.process_video(
                        temporary_video_file.name, annotated_video_path
                    ),
                )
            finally:
                os.unlink(temporary_video_file.name)

main_content_column, cart_column = st.columns([3, 2])
with main_content_column:
    st.subheader("Annotated output")
    if st.session_state.last:
        kind, result = st.session_state.last
        if kind == "image":
            st.image(cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB))
            st.caption(
                "Single image: identification only. Cart events need temporal evidence (video)."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        dict(
                            label=detection.label,
                            det_conf=round(detection.confidence, 3),
                            status=identification.status,
                            sku=identification.sku_id,
                            reason=identification.reason,
                        )
                        for detection, identification in zip(
                            result["detections"], result["identifications"]
                        )
                    ]
                )
            )
            for barcode, identification in result["orphan_barcodes"]:
                st.write(
                    f"Barcode `{barcode.value}` ({barcode.symbology}) -> "
                    f"{identification.status} {identification.sku_id or ''} - "
                    f"{identification.reason}"
                )
            st.write(
                "Latency ms:",
                {
                    stage: round(milliseconds, 1)
                    for stage, milliseconds in result["latency_ms"].items()
                },
            )
        else:
            st.write(
                f"{result['frames']} frames processed. Annotated video saved to "
                f"`{result['out_path']}` (open it locally)."
            )
            with st.expander("Per-instance inventory diagnostics"):
                st.json(result.get("inventory", {"status": "not available"}))
            total_latency = result["latency_ms"]["total"]
            if total_latency:
                st.write(
                    f"Per-frame latency ms: median {np.median(total_latency):.1f}, p95 {np.percentile(total_latency, 95):.1f} (this machine)"
                )
    else:
        st.info("Load an image/video and press Run inference.")
    st.subheader("Manual add (e.g. from a single image)")
    sku_column, event_type_column, reason_column = st.columns(3)
    selected_sku = sku_column.selectbox(
        "SKU",
        list(catalog.products),
        format_func=lambda sku_id: f"{sku_id} {catalog.get(sku_id).name}",
    )
    event_type = event_type_column.selectbox("Type", ["ADD", "REMOVE"])
    reason = reason_column.text_input("Reason", "operator confirmed visually")
    if st.button("Create manual event"):
        cart_session.manual_event(event_type, selected_sku, reason)
        st.rerun()
with cart_column:
    st.subheader("Digital cart")
    bill = compute_bill(cart_session, catalog)
    st.dataframe(
        pd.DataFrame(bill["lines"])
        if bill["lines"]
        else pd.DataFrame(columns=["sku_id", "name", "qty", "unit_cents", "line_cents"])
    )
    st.metric(
        "Subtotal" + (" (PROVISIONAL)" if bill["status"] == "PROVISIONAL" else ""),
        f"{bill['subtotal_cents'] / 100:.2f} {bill['currency']}",
    )
    for warning in bill["warnings"]:
        st.warning(warning)
    st.subheader("Needs review")
    for event in cart_session.pending():
        with st.expander(
            f"{event.event_id} {event.event_type} track={event.track_id} sku={event.sku_id}"
        ):
            st.write(event.evidence)
            selected_sku = st.selectbox(
                "SKU",
                list(catalog.products),
                key="s" + event.event_id,
                index=(
                    list(catalog.products).index(event.sku_id)
                    if event.sku_id in catalog.products
                    else 0
                ),
            )
            selected_event_type = st.selectbox(
                "Type",
                ["ADD", "REMOVE"],
                key="t" + event.event_id,
                index=1 if event.event_type == "REMOVE" else 0,
            )
            review_reason = st.text_input("Reason (required)", key="r" + event.event_id)
            confirm_column, reject_column = st.columns(2)
            if (
                confirm_column.button("Confirm / correct", key="c" + event.event_id)
                and review_reason
            ):
                cart_session.confirm(
                    event.event_id,
                    review_reason,
                    selected_sku,
                    selected_event_type,
                )
                st.rerun()
            if (
                reject_column.button("Reject", key="x" + event.event_id)
                and review_reason
            ):
                cart_session.reject(event.event_id, review_reason)
                st.rerun()

st.subheader("Events")
st.dataframe(
    pd.DataFrame(
        [
            {
                key: value
                for key, value in event.to_dict().items()
                if key not in ("state_history", "dedupe_key")
            }
            for event in cart_session.events.values()
        ]
    )
)
with st.expander("Audit log"):
    st.dataframe(pd.DataFrame(cart_session.audit))

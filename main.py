import streamlit as st
from PIL import Image

from app.analyzer import analyze_image, compare_images
from app.visualizer import (
    plot_color_palette,
    plot_brightness_heatmap,
    plot_edge_heatmap,
    draw_ui_regions,
)

st.set_page_config(
    page_title="Game Screenshot Analyzer",
    page_icon="🎮",
    layout="wide",
)

st.markdown(
    """
    <style>
    .yoxawork-watermark {
        position: fixed;
        bottom: 12px;
        right: 18px;
        color: rgba(255, 255, 255, 0.45);
        font-size: 12px;
        font-family: Arial, sans-serif;
        z-index: 999999;
        pointer-events: none;
    }
    </style>

    <div class="yoxawork-watermark">
        Powered by YoxaWork
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🎮 Game Screenshot Analyzer")
st.caption(
    "Computer vision toolkit for analyzing visual characteristics and UI composition "
    "in game screenshots."
)

with st.sidebar:
    st.header("Analysis")
    mode = st.radio(
        "Mode",
        ["Single Screenshot", "Compare Screenshots"],
    )
    st.divider()
    st.info(
        "V1 uses classical computer vision and heuristics. "
        "It does not require an AI model download."
    )

if mode == "Single Screenshot":
    uploaded = st.file_uploader(
        "Upload a game screenshot",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        result = analyze_image(image)

        col_img, col_info = st.columns([1.25, 1])

        with col_img:
            st.subheader("Screenshot")
            st.image(image, use_container_width=True)

        with col_info:
            st.subheader("Technical Metrics")
            m = result["metrics"]

            c1, c2 = st.columns(2)
            c1.metric("Resolution", f"{m['width']} × {m['height']}")
            c2.metric("Aspect Ratio", m["aspect_ratio"])

            c1, c2 = st.columns(2)
            c1.metric("Brightness", f"{m['brightness']:.1f}%")
            c2.metric("Contrast", f"{m['contrast']:.1f}%")

            c1, c2 = st.columns(2)
            c1.metric("Saturation", f"{m['saturation']:.1f}%")
            c2.metric("Sharpness", f"{m['sharpness']:.1f}")

            c1, c2 = st.columns(2)
            c1.metric("Edge Density", f"{m['edge_density']:.1f}%")
            c2.metric("Quality Score", f"{m['quality_score']:.0f}/100")

        st.divider()

        ui = result["ui"]
        st.subheader("HUD / UI Analysis")

        c1, c2, c3 = st.columns(3)
        c1.metric("UI Detected", "Yes" if ui["detected"] else "No")
        c2.metric("Estimated UI Coverage", f"{ui['coverage']:.1f}%")
        c3.metric("Detected Regions", str(len(ui["regions"])))

        st.caption(
            "UI detection is heuristic-based and estimates screen regions that "
            "look like HUD overlays. It is not a semantic game-UI detector."
        )

        overlay = draw_ui_regions(image, ui["regions"])
        st.image(
            overlay,
            caption="Estimated HUD/UI regions",
            use_container_width=True,
        )

        st.divider()
        st.subheader("Visual Analysis")

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_color_palette(result["palette"]), clear_figure=True)
        with c2:
            st.pyplot(plot_brightness_heatmap(result["gray"]), clear_figure=True)

        st.pyplot(plot_edge_heatmap(result["edges"]), clear_figure=True)

    else:
        st.info("Upload a PNG, JPG, JPEG, or WEBP game screenshot to begin.")

else:
    col1, col2 = st.columns(2)
    with col1:
        a_file = st.file_uploader(
            "Screenshot A",
            type=["png", "jpg", "jpeg", "webp"],
            key="a",
        )
    with col2:
        b_file = st.file_uploader(
            "Screenshot B",
            type=["png", "jpg", "jpeg", "webp"],
            key="b",
        )

    if a_file and b_file:
        image_a = Image.open(a_file).convert("RGB")
        image_b = Image.open(b_file).convert("RGB")
        comparison = compare_images(image_a, image_b)

        st.subheader("Comparison")

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Brightness Difference",
            f"{comparison['brightness_difference']:.1f} pts",
        )
        c2.metric(
            "Structural Difference",
            f"{comparison['difference_score']:.1f}%",
        )
        c3.metric(
            "Similarity",
            f"{comparison['similarity']:.1f}%",
        )

        a, b = st.columns(2)
        with a:
            st.image(image_a, caption="Screenshot A", use_container_width=True)
        with b:
            st.image(image_b, caption="Screenshot B", use_container_width=True)

        st.subheader("Metric Comparison")
        st.dataframe(
            comparison["table"],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Upload two screenshots to compare them.")

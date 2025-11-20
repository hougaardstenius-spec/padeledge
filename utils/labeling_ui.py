import os
import tempfile
from typing import List

import streamlit as st

from utils.dataset_manager import (
    get_dataset_overview,
    save_labeled_clip,
    list_uncertain_clips,
)


def _get_known_categories_and_shots():
    df = get_dataset_overview()
    cats = sorted(df["Category"].unique()) if not df.empty else []
    shots = sorted(df["Shot Type"].unique()) if not df.empty else []
    return cats, shots


def render_labeling_ui():
    """
    UI til:
    - at uploade nye klip og placere dem korrekt i data/samples/<category>/<shot_type>/
    - at gennemgå "uncertain" klip (data/uncertain) og flytte dem ind i dataset.
    """
    st.subheader("🎬 Tilføj og label nye træningsklip")

    known_categories, known_shots = _get_known_categories_and_shots()

    tab_upload, tab_uncertain = st.tabs(["Upload nye klip", "Review 'uncertain' klip"])

    with tab_upload:
        st.write("Upload et nyt videoklip og vælg kategori + slag-type.")

        up_file = st.file_uploader(
            "Upload træningsklip (mp4/mov/avi)",
            type=["mp4", "mov", "avi"],
            key="label_upload",
        )

        col1, col2 = st.columns(2)
        with col1:
            new_category = st.text_input(
                "Kategori (fx 'overhead', 'groundstrokes', 'volley')",
                value=known_categories[0] if known_categories else "",
            )
        with col2:
            new_shot = st.text_input(
                "Shot-type (fx 'bandeja', 'vibora', 'smash')",
                value=known_shots[0] if known_shots else "",
            )

        if up_file and new_category.strip() and new_shot.strip():
            st.video(up_file)

            if st.button("💾 Gem klip i dataset", type="primary"):
                # skriv midlertidigt til disk
                tmp_dir = tempfile.mkdtemp()
                tmp_path = os.path.join(tmp_dir, up_file.name)
                with open(tmp_path, "wb") as f:
                    f.write(up_file.read())

                final_path = save_labeled_clip(
                    tmp_path,
                    new_category.strip(),
                    new_shot.strip(),
                    filename=up_file.name,
                )
                st.success(f"Klip gemt som træningsdata: {final_path}")
        elif up_file:
            st.warning("Udfyld både kategori og shot-type for at gemme klippet.")

    with tab_uncertain:
        st.write(
            "Her kan du se klip som senere kan blive markeret som 'uncertain' af modellen "
            "(fx data/uncertain). Lige nu forventes videoer i mappen `data/uncertain`."
        )

        uncertain_files = list_uncertain_clips()
        if not uncertain_files:
            st.info("Ingen 'uncertain' klip fundet endnu.")
            return

        selected = st.selectbox(
            "Vælg et klip til review",
            options=uncertain_files,
        )

        if selected:
            st.video(selected)

            col1, col2 = st.columns(2)
            with col1:
                cat = st.text_input(
                    "Kategori for dette klip",
                    value=known_categories[0] if known_categories else "",
                    key="uncertain_cat",
                )
            with col2:
                shot = st.text_input(
                    "Shot-type for dette klip",
                    value=known_shots[0] if known_shots else "",
                    key="uncertain_shot",
                )

            if st.button("✅ Label og flyt klip til dataset"):
                final_path = save_labeled_clip(
                    selected, cat.strip(), shot.strip(), filename=os.path.basename(selected)
                )
                st.success(f"Klip flyttet til: {final_path}")
                st.info("Filen er fjernet fra 'uncertain'-mappen ved flytningen.")

import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Needle Check (Refs + 4 Tests)", page_icon="🧵")
st.title("🧵 Needle Deformation Checks (Reference-based)")

st.markdown("""    This app takes 4 **reference** images (ref1..ref4) and 4 **test** images of the same needle,
then runs the MATLAB-equivalent logic to check:
- **Breakage** (Test #1 ↔ Ref1)
- **Bending**  (Test #3 ↔ Ref2)
- **Wear**     (Test #2 ↔ Ref1)
""")

# (Functions omitted here for brevity but identical to previous assistant message)
# In real code you'd include the full helper functions and analysis logic.

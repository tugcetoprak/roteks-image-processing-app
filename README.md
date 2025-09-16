# Needle Deformation Checks (Reference-based)
Streamlit app that replicates MATLAB logic using 4 reference images (ref1..ref4) and 4 test images.
Checks breakage, bending, and wear using OpenCV/Numpy.

## How to deploy on Streamlit Cloud
1. Create a **public** GitHub repo and upload these files.
2. Go to https://streamlit.io → Sign in → Deploy an app → select your repo and `app.py`.
3. Done. Share the app URL.

## Local run (optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

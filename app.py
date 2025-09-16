import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Needle Check (Refs + 4 Tests)", page_icon="🧵")
st.title("🧵 Needle Deformation Checks (Reference-based)")

st.markdown("""
Bu uygulama referans görüntüleri ve aynı iğneye ait **4 test görüntüsünü** alır, ardından
**Kırılma**, **Eğilme** ve **Aşınma** kontrollerini yapar.
""")

# ---------- Yardımcılar ----------

def to_gray01(img_rgb: np.ndarray) -> np.ndarray:
    """RGB -> grayscale float32 [0,1]."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0
    return gray

def imbinarize_invert_like_matlab(img_rgb: np.ndarray, thr: float=0.2) -> np.ndarray:
    """
    MATLAB: L = ~imbinarize(img, 0.2)
    Burada önce griye çevirip 0.2 eşik ile binarize ediyoruz, sonra invert (~).
    """
    g = to_gray01(img_rgb)
    bin_ = (g > thr).astype(np.uint8)
    inv = 1 - bin_  # ~
    return inv

def fill_holes(binary: np.ndarray) -> np.ndarray:
    """
    MATLAB imfill(...,'holes') eşleniği. binary: 0/1.
    OpenCV floodFill ile arka planı doldurup tersliyoruz.
    """
    h, w = binary.shape
    # Kenarlardan flood-fill için 0/255'e çevir
    mask = np.zeros((h + 2, w + 2), np.uint8)
    img = (binary * 255).astype(np.uint8)
    # Arka planı (0) dıştan içe doldur
    flood = img.copy()
    cv2.floodFill(flood, mask, (0, 0), 255)
    # flood doldurulan yerler 255; içerdeki boşlukları doldurmak için:
    flood_inv = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(img, flood_inv)
    return (filled > 0).astype(np.uint8)

def pad_and_fill_like_matlab(binary: np.ndarray) -> np.ndarray:
    """
    MATLAB:
      bw_c = padarray(bw,[1 1],1,'post');
      bw_c_filled = imfill(bw_c,'holes');
      bw_filled = bw_c_filled(1:end-1,1:end-1);
    """
    h, w = binary.shape
    bw_c = np.ones((h + 1, w + 1), dtype=np.uint8)
    bw_c[:h, :w] = binary
    bw_c_filled = fill_holes(bw_c)
    bw_filled = bw_c_filled[:h, :w]
    return bw_filled

def erode_disk(binary: np.ndarray, radius: int = 4) -> np.ndarray:
    """MATLAB strel('disk',4) yaklaşık karşılığı: eliptik kernel (2r+1)."""
    k = 2 * radius + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode((binary * 255).astype(np.uint8), se)
    return (eroded > 0).astype(np.uint8)

def row_first_loc_from_bottom(binary: np.ndarray) -> int:
    """
    MATLAB:
        firstloc = find(erodeddefimg(end,:),1,'first');
    Yani alt satırda ilk 1 görülen kolon indeksi. Python'da 0-based döndürüyoruz.
    """
    last_row = binary[-1, :]
    idx = np.argmax(last_row > 0)
    if (last_row > 0).any():
        return int(idx)
    return -1  # bulunamadı

def longest_nonempty_height(binary: np.ndarray) -> int:
    """
    MATLAB:
      [~,v] = max(binary,[],2); v(v==1) = [];
      length(v) referans/test boyu gibi kullanılmış.
    Bizim eşleniğimiz: bir satırda en az bir '1' varsa say.
    """
    return int(np.sum(binary.max(axis=1) > 0))

def slope_percent(binary: np.ndarray) -> float:
    """
    MATLAB eğilme:
      v = max(...,[],2) -> satır bazında en az bir 1 olan satırlar
      ilk ve son dolu satırların '1' bölgesinin orta kolonları arasındaki eğim * 100
    """
    rows = np.where(binary.max(axis=1) > 0)[0]
    if len(rows) < 2:
        return 0.0
    first_r, last_r = int(rows[0]), int(rows[-1])

    def middle_col(r):
        row = binary[r, :]
        cols = np.where(row > 0)[0]
        if len(cols) == 0:
            return None
        mid = int(cols[(len(cols) - 1) // 2])  # yaklaşık orta
        return mid

    m1 = middle_col(first_r)
    m2 = middle_col(last_r)
    if m1 is None or m2 is None or last_r == first_r:
        return 0.0
    slope = abs((m2 - m1) / (last_r - first_r)) * 100.0
    return float(slope)

def smooth_moving(arr: np.ndarray, span: int = 5) -> np.ndarray:
    """MATLAB smooth(x,'moving') ~ pencere ortalaması (default ~5)."""
    if arr.size == 0:
        return arr
    k = max(1, int(span))
    kernel = np.ones(k, dtype=np.float32) / k
    sm = np.convolve(arr.astype(np.float32), kernel, mode="same")
    return sm

def preprocess_pipeline(img_rgb: np.ndarray) -> np.ndarray:
    """
    MATLAB zinciri:
      L = ~imbinarize(img,0.2) -> gerekirse invert
      bw = imfill(logical(L),'holes')
      pad + fill + crop
      erode with disk(4)
    """
    L = imbinarize_invert_like_matlab(img_rgb, thr=0.2)

    # Sütunlarda 1 sayısı çoğunlukta mı kontrolü (sum(any(L)) > sum(any(~L))) eşleniği:
    cols_L = np.sum(np.any(L > 0, axis=0))
    cols_notL = np.sum(np.any((1 - L) > 0, axis=0))
    if cols_L > cols_notL:
        L = 1 - L

    bwdef = fill_holes(L)
    bw_filled = pad_and_fill_like_matlab(bwdef)
    eroded = erode_disk(bw_filled, radius=4)
    return eroded

# ---------- Yüklemeler ----------

st.subheader("1) Referans Görüntüler (Ref1..Ref4)")
ref_files = st.file_uploader("Ref görüntülerini sırasıyla yükleyin (4 adet)", type=["jpg","jpeg","png"],
                             accept_multiple_files=True, key="refs")

if ref_files and len(ref_files) != 4:
    st.warning("Lütfen **tam olarak 4** referans görüntü yükleyin (Ref1, Ref2, Ref3, Ref4).")

if ref_files and len(ref_files) == 4:
    ref_imgs = [Image.open(f).convert("RGB") for f in ref_files]
    cols = st.columns(4)
    for i, (c, im) in enumerate(zip(cols, ref_imgs), start=1):
        with c:
            c.image(im, caption=f"Ref{i}", use_container_width=True)

st.subheader("2) Test Görüntüleri (Aynı iğneye ait 4 adet)")
test_files = st.file_uploader("Test görüntülerini yükleyin (4 adet)", type=["jpg","jpeg","png"],
                              accept_multiple_files=True, key="tests")

if test_files and len(test_files) != 4:
    st.warning("Lütfen **tam olarak 4** test görüntüsü yükleyin (aynı iğneye ait).")

if (ref_files and len(ref_files) == 4) and (test_files and len(test_files) == 4):
    test_imgs = [Image.open(f).convert("RGB") for f in test_files]

    st.markdown("**Önizleme (Testler):**")
    cols = st.columns(4)
    for i, (c, im) in enumerate(zip(cols, test_imgs), start=1):
        with c:
            c.image(im, caption=f"Test{i}", use_container_width=True)

    if st.button("Analizi Çalıştır"):
        # Ref görüntülerini numpy olarak al
        ref_np = [np.array(im) for im in ref_imgs]
        # Test görüntülerini numpy olarak al
        tst_np = [np.array(im) for im in test_imgs]

        # --- KIRILMA (Test #1 vs Ref1) ---
        st.markdown("### ⭐ Kırılma Kontrolü (Test1 ↔ Ref1)")
        # Resimleri ref1 boyutuna resize
        defimg = cv2.resize(tst_np[0], (ref_np[0].shape[1], ref_np[0].shape[0]))
        eroded_def = preprocess_pipeline(defimg)

        refimg = ref_np[0]
        eroded_ref = preprocess_pipeline(refimg)

        vdef_h = longest_nonempty_height(eroded_def)
        vref_h = longest_nonempty_height(eroded_ref)

        st.write(f"Referans iğnenin boyu (piksel satır): {vref_h}")
        st.write(f"Test iğnesinin boyu (piksel satır): {vdef_h}")

        if vref_h > 0:
            defperc = abs((vdef_h - vref_h) / vref_h) * 100.0
        else:
            defperc = 0.0
        st.write(f"Deformasyon oranı (yüzde): {defperc:.3f}")

        broken = int(defperc > 35.0)  # eşik
        if broken:
            st.success("Test iğnesinin **kırık** olduğu tespit edilmiştir.")
        else:
            st.info("Test iğnesinin **kırık olmadığı** tespit edilmiştir. Eğilme ve aşınma kontrolüne geçiliyor.")

        # --- EĞİLME (Test #3 ↔ Ref2) ---
        st.markdown("---")
        st.markdown("### ⭐ Eğilme Kontrolü (Test3 ↔ Ref2)")
        defimg = cv2.resize(tst_np[2], (ref_np[1].shape[1], ref_np[1].shape[0]))
        eroded_def = preprocess_pipeline(defimg)

        refimg = ref_np[1]
        eroded_ref = preprocess_pipeline(refimg)

        refslope = slope_percent(eroded_ref)
        defslope = slope_percent(eroded_def)

        st.write(f"Referans iğnenin olası eğikliği: {refslope:.3f}")
        st.write(f"Test iğnesinin eğikliği: {defslope:.3f}")

        bent = int(defslope > 5.0)  # eşik
        if bent:
            st.success("Test iğnesinde **eğiklik** tespit edilmiştir.")
        else:
            st.info("Test iğnesinde **eğiklik tespit edilmemiştir**. Aşınma kontrolüne geçiliyor.")

        # --- AŞINMA (Test #2 ↔ Ref1) ---
        st.markdown("---")
        st.markdown("### ⭐ Aşınma Kontrolü (Test2 ↔ Ref1)")
        defimg = cv2.resize(tst_np[1], (ref_np[0].shape[1], ref_np[0].shape[0]))
        eroded_def = preprocess_pipeline(defimg)

        refimg = ref_np[0]
        eroded_ref = preprocess_pipeline(refimg)

        # grafikleri çıkar: her satırdaki 1 sayısı
        def row_ones_count(binary: np.ndarray) -> np.ndarray:
            return np.sum(binary > 0, axis=1).astype(np.float32)

        graph_ref = row_ones_count(eroded_ref)
        graph_def = row_ones_count(eroded_def)

        # sıfırları at
        graph_ref = graph_ref[graph_ref > 0]
        graph_def = graph_def[graph_def > 0]

        # smooth (moving)
        graph_ref_s = smooth_moving(graph_ref, span=5)
        graph_def_s = smooth_moving(graph_def, span=5)

        # ilk 10 farkların ortalaması (veri kısa ise elde olan kadarını al)
        n = min(10, len(graph_ref_s) - 1, len(graph_def_s) - 1)
        if n <= 1:
            finalcal = 0.0
        else:
            dref = np.diff(graph_ref_s[: n + 1])
            ddef = np.diff(graph_def_s[: n + 1])
            # ortalamalar (0 bölünme koruması)
            mref = float(np.mean(dref)) if np.any(np.isfinite(dref)) else 1e-6
            mdef = float(np.mean(ddef)) if np.any(np.isfinite(ddef)) else 0.0
            if abs(mref) < 1e-6:
                mref = 1e-6
            finalcal = mdef / mref

        st.write(f"Aşınma miktarı (yüzde değişim oranı benzeri): {finalcal:.3f}")

        worn = int(finalcal > 1.5)
        if worn:
            st.success("Test iğnesinin **aşınma miktarı yüksek** bulunmuştur.")
        else:
            st.info("Test iğnesinin **aşınma miktarı yüksek değildir**.")

        # Özet tablo
        st.markdown("---")
        st.subheader("Özet")
        st.write(f"Kırılma: {'Pozitif' if broken else 'Negatif'}")
        st.write(f"Eğilme:  {'Pozitif' if bent else 'Negatif'}")
        st.write(f"Aşınma:  {'Pozitif' if worn else 'Negatif'}")

else:
    st.info("Lütfen önce 4 referans ve 4 test görüntüsü yükleyin.")

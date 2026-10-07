# ============================================================
#  WeedScan — Streamlit app
#  ระบบตรวจจับวัชพืชในนาข้าวด้วยโดรนและการเรียนรู้เชิงลึก
# ============================================================
import glob, os, time
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import weedscan_core as core

st.set_page_config(page_title='WeedScan · ตรวจจับวัชพืชในนาข้าว', page_icon='🌾', layout='wide')

# ---------------- สไตล์ ----------------
ST_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, label, button, input, p, span, div {
  font-family: 'IBM Plex Sans Thai', sans-serif; }
[data-testid="stAppViewContainer"] {
  background-color: #070b09;
  background-image:
    linear-gradient(rgba(182,242,58,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(182,242,58,.035) 1px, transparent 1px);
  background-size: 40px 40px; }
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stDecoration"] { visibility: hidden; height: 0; }
.block-container { max-width: 1320px; padding-top: 1.6rem; padding-bottom: 2rem; }
[data-testid="stVerticalBlockBorderWrapper"] { background: #0e1512; border-color: #1f2d27 !important; border-radius: 16px !important; }
.stButton > button { width: 100%; height: 52px; border-radius: 12px; border: none;
  background: #b6f23a; color: #0a0f0d; font-weight: 700; font-size: 17px;
  box-shadow: 0 8px 30px -10px rgba(182,242,58,.6); transition: transform .15s ease, box-shadow .15s ease; }
.stButton > button:hover { transform: translateY(-1px); color: #0a0f0d; background: #c4f75a;
  box-shadow: 0 12px 34px -10px rgba(182,242,58,.8); }
.stButton > button:focus:not(:active) { color: #0a0f0d; border: none; }
[data-testid="stFileUploaderDropzone"] { background: #121c18; border: 1px dashed #2a3b33; border-radius: 12px; }
[data-testid="stImage"] img { border-radius: 10px; }
[data-testid="stImageCaption"], [data-testid="caption"] { color: #8a9a91 !important; }
.img-label { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #b6f23a; margin: 0 0 6px; }
"""
st.markdown(f'<style>{core.CSS}\n{ST_CSS}</style>', unsafe_allow_html=True)


@st.cache_resource(show_spinner='กำลังโหลดโมเดล…')
def get_models():
    return core.load_models()


def load_image(src):
    im = ImageOps.exif_transpose(Image.open(src)).convert('RGB')
    if max(im.size) > core.MAX_SIDE:
        im.thumbnail((core.MAX_SIDE, core.MAX_SIDE), Image.LANCZOS)
    return np.asarray(im)


st.markdown(core.hero_html(), unsafe_allow_html=True)

examples = sorted(glob.glob(os.path.join(core.EXAMPLE_DIR, '*.[jJpP][pPnN]*[gG]')))
on_gpu = False
try:
    import torch
    on_gpu = torch.cuda.is_available()
except Exception:
    pass

left, right = st.columns([4, 8], gap='medium')

with left:
    with st.container(border=True):
        st.markdown('<div class="section-label">01 · Input</div>', unsafe_allow_html=True)
        up = st.file_uploader('อัปโหลดภาพจากโดรน', type=['jpg', 'jpeg', 'png'])
        choice = None
        if examples:
            names = ['— ใช้ภาพที่อัปโหลด —'] + [os.path.basename(p) for p in examples]
            pick = st.selectbox('หรือเลือกภาพตัวอย่างจากแปลงนาไทย', names, index=0 if up else 1)
            if pick != names[0]:
                choice = examples[names.index(pick) - 1]
        src = up if (up is not None and choice is None) else choice
        img = load_image(src) if src is not None else None
        if img is not None:
            st.image(img, caption=f'{img.shape[1]}×{img.shape[0]} px')

        use_sahi = st.toggle('ใช้ SAHI — จับวัชพืชต้นเล็กได้ดีขึ้น', value=on_gpu,
                             help=None if on_gpu else 'บนเซิร์ฟเวอร์ CPU ฟรี ใช้เวลาประมาณ 10–30 วินาทีต่อภาพ')
        conf = st.slider('Confidence threshold', 0.10, 0.90, 0.25, 0.05)
        grid = st.slider('ความละเอียดกริดแผนที่ (คอลัมน์)', 6, 30, 12, 1)
        go = st.button('สแกนวัชพืช  →', type='primary', disabled=img is None)

if go and img is not None:
    get_models()
    with st.spinner('กำลังตรวจจับวัชพืช…' + (' (SAHI)' if use_sahi else '')):
        t0 = time.time()
        dets = core.detect(img, use_sahi, conf)
        secs = time.time() - t0
        det_img = core.draw_detections(img, dets)
        pmap, level = core.prescription_map(img, dets, int(grid))
    st.session_state['result'] = dict(det=det_img, map=pmap,
                                      html=core.results_html(dets, level, secs, use_sahi))

res = st.session_state.get('result')

with right:
    with st.container(border=True):
        st.markdown('<div class="section-label">02 · Detection &amp; Prescription Map</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="img-label">ผลการตรวจจับ</div>', unsafe_allow_html=True)
            if res: st.image(res['det'])
            else: st.markdown('<div id="results"><div class="empty">ยังไม่มีผล</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="img-label">แผนที่กำหนดการพ่น</div>', unsafe_allow_html=True)
            if res: st.image(res['map'])
            else: st.markdown('<div id="results"><div class="empty">ยังไม่มีผล</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label" style="margin-top:14px">03 · Summary</div>', unsafe_allow_html=True)
        st.markdown(res['html'] if res else core.results_html(None, None, 0, use_sahi), unsafe_allow_html=True)

st.markdown('<div id="credit">ปริญญานิพนธ์ <b>การพัฒนาระบบตรวจจับวัชพืชในนาข้าวด้วยโดรนและการเรียนรู้เชิงลึก</b> · '
            'นายอรรถวิทย์ หมัดสอิ๊ด · อาจารย์ที่ปรึกษา ผศ.ดร.วสุ อุดมเพทายกุล · สจล.</div>', unsafe_allow_html=True)

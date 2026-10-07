# ============================================================
#  WeedScan core — detection, prescription map, HTML widgets
#  (UI อยู่ใน streamlit_app.py)
# ============================================================
import glob, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------- ตั้งค่า ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
# Spaces: วางไฟล์ weed_best.pt ไว้ข้าง app.py | Colab: ตั้ง os.environ['MODEL_PATH'] ก่อน import
MODEL_PATH  = os.environ.get('MODEL_PATH', os.path.join(HERE, 'weed_best.pt'))
EXAMPLE_DIR = os.environ.get('EXAMPLE_DIR', os.path.join(HERE, 'examples'))
IMGSZ       = int(os.environ.get('IMGSZ', 640))      # ต้องตรงกับตอนเทรน (weed_only_v2 = 640)
MAX_SIDE    = int(os.environ.get('MAX_SIDE', 2048))  # ย่อภาพใหญ่ก่อนประมวลผล เพื่อไม่ให้ SAHI บน CPU ช้าเกินไป

# ตัวเลขบนหัวเว็บ — ผลบน test set (แก้ให้ตรงกับโมเดลที่ใช้จริง)
MODEL_NAME  = 'YOLOv8s · weed_only_v2'
MODEL_STATS = [('mAP50 (test)', '82.8%'), ('Precision', '79.7%'), ('Recall', '83.2%'), ('Speed (T4)', '13.0 ms')]

# เกณฑ์ตามตารางที่ 4.8 ในเล่ม
LEVELS = [  # (ชื่อ, conf ต่ำสุด, สีกรอบ, สีบนแผนที่, การดำเนินการ)
    ('สูง',  0.70, (220, 38, 38),  (217, 48, 37),  'พ่นสารทันที'),
    ('กลาง', 0.50, (245, 140, 0),  (242, 139, 32), 'เฝ้าระวัง'),
    ('ต่ำ',  0.00, (250, 204, 21), (46, 158, 68),  'ตรวจสอบเพิ่มเติม'),
]

def level_of(conf):
    for i, (_, lo, *_rest) in enumerate(LEVELS):
        if conf >= lo:
            return i
    return len(LEVELS) - 1

# ---------------- ฟอนต์ไทย ----------------
def thai_font(size):
    for pat in ['/usr/share/fonts/**/Loma.ttf', '/usr/share/fonts/**/Loma.otf',
                '/usr/share/fonts/**/Sarabun*.ttf', '/usr/share/fonts/**/Garuda.ttf']:
        found = glob.glob(pat, recursive=True)
        if found:
            return ImageFont.truetype(found[0], size)
    return ImageFont.load_default()

# ---------------- ตรวจจับ ----------------
_yolo = _sahi = None

def load_models():
    global _yolo, _sahi
    if _yolo is None:
        from ultralytics import YOLO
        from sahi import AutoDetectionModel
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        _yolo = YOLO(MODEL_PATH)
        for mt in ('ultralytics', 'yolov8'):        # SAHI รุ่นใหม่ใช้ 'ultralytics', รุ่นเก่าใช้ 'yolov8'
            try:
                _sahi = AutoDetectionModel.from_pretrained(
                    model_type=mt, model_path=MODEL_PATH, image_size=IMGSZ,
                    confidence_threshold=0.1, device=device)   # กรอง conf จริงทีหลัง
                break
            except Exception as e:
                last_err = e
        else:
            raise last_err
    return _yolo, _sahi

def detect(img_rgb, use_sahi, conf_th, slice_size=320, overlap=0.2):
    """คืนค่า list ของ (x1, y1, x2, y2, conf) — img_rgb เป็น numpy RGB"""
    yolo, sahi_model = load_models()
    if use_sahi:
        from sahi.predict import get_sliced_prediction
        res = get_sliced_prediction(img_rgb, sahi_model,
                                    slice_height=slice_size, slice_width=slice_size,
                                    overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                                    verbose=0)
        dets = [(p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy, p.score.value)
                for p in res.object_prediction_list]
    else:
        # ส่งเป็น PIL เพื่อให้ ultralytics รู้ว่าเป็น RGB (numpy จะถูกมองเป็น BGR)
        r = yolo.predict(Image.fromarray(img_rgb), conf=0.1, imgsz=IMGSZ, verbose=False)[0]
        dets = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes]
    return [d for d in dets if d[4] >= conf_th]

# ---------------- วาดผล ----------------
def draw_detections(img_rgb, dets):
    im = Image.fromarray(img_rgb).convert('RGB')
    W, H = im.size
    lw = max(2, round(min(W, H) / 250))
    font = thai_font(max(14, round(min(W, H) / 40)))
    dr = ImageDraw.Draw(im)
    for x1, y1, x2, y2, c in sorted(dets, key=lambda d: d[4]):   # กรอบ conf สูงวาดทับทีหลัง
        color = LEVELS[level_of(c)][2]
        dr.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        text = f'{c:.2f}'
        tb = dr.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0] + 8, tb[3] - tb[1] + 8
        ty = y1 - th if y1 - th >= 0 else y1          # ชิดขอบบน -> ย้ายป้ายเข้าในกรอบ
        tx = min(x1, W - tw)
        dr.rectangle([tx, ty, tx + tw, ty + th], fill=color)
        dr.text((tx + 4, ty + 4 - tb[1]), text, font=font, fill='black')
    return im

def prescription_map(img_rgb, dets, grid_cols=12):
    """แบ่งภาพเป็นกริด แต่ละช่องได้ระดับสูงสุดของกรอบที่ทับช่องนั้น"""
    H, W = img_rgb.shape[:2]
    cell = W / grid_cols
    grid_rows = max(1, round(H / cell))
    cell_h = H / grid_rows
    level = np.full((grid_rows, grid_cols), -1)        # -1 = ไม่พบวัชพืช
    for x1, y1, x2, y2, c in dets:
        lv = level_of(c)
        c0, c1 = int(x1 // cell), int(min(x2, W - 1) // cell)
        r0, r1 = int(y1 // cell_h), int(min(y2, H - 1) // cell_h)
        for r in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if level[r, cc] == -1 or lv < level[r, cc]:   # index น้อย = ระดับสูงกว่า
                    level[r, cc] = lv
    base = Image.fromarray(img_rgb).convert('RGBA')
    over = Image.new('RGBA', base.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(over)
    for r in range(grid_rows):
        for cc in range(grid_cols):
            box = [cc * cell, r * cell_h, (cc + 1) * cell, (r + 1) * cell_h]
            if level[r, cc] >= 0:
                dr.rectangle(box, fill=LEVELS[level[r, cc]][3] + (130,))
            dr.rectangle(box, outline=(255, 255, 255, 70), width=1)
    return Image.alpha_composite(base, over).convert('RGB'), level

def summary(dets, level):
    total = level.size
    rows = []
    for i, (name, lo, _, _, action) in enumerate(LEVELS):
        n = sum(1 for d in dets if level_of(d[4]) == i)
        cells = int((level == i).sum())
        rows.append([name, action, n, f'{100 * cells / total:.1f}%'])
    confs = [d[4] for d in dets]
    spray_now = (level == 0).sum() / total * 100
    spray_any = ((level == 0) | (level == 1)).sum() / total * 100
    md = (f'**พบวัชพืช {len(dets)} จุด** · conf สูงสุด {max(confs, default=0):.2f} · '
          f'เฉลี่ย {np.mean(confs) if confs else 0:.2f}\n\n'
          f'พื้นที่ที่ต้องพ่นทันที **{spray_now:.1f}%** ของภาพ '
          f'(รวมระดับเฝ้าระวัง {spray_any:.1f}%)\n\n'
          f'<sub>สัดส่วนพื้นที่คิดจากช่องกริดในภาพนี้เท่านั้น ยังไม่ใช่ปริมาณสารที่ลดได้จริงในแปลง</sub>')
    return rows, md

# ================= หน้าตาแอป (UI) =================
import time


CSS = r"""
:root, .dark {
  --bg: #070b09; --panel: #0e1512; --panel-2: #121c18; --line: #1f2d27;
  --ink: #e8f0ea; --muted: #8a9a91; --lime: #b6f23a; --lime-dim: #6f9a1f;
  --red: #ff5a4e; --amber: #ffad33; --green: #3ddc84;
}
body, gradio-app, .gradio-container {
  background: var(--bg) !important;
  background-image:
    linear-gradient(rgba(182,242,58,.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(182,242,58,.035) 1px, transparent 1px) !important;
  background-size: 40px 40px !important;
}
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* ---------- hero ---------- */
#hero { position: relative; overflow: hidden; border: 1px solid var(--line); border-radius: 18px;
  padding: 28px 32px; margin-bottom: 8px;
  background: radial-gradient(1200px 300px at 85% -40%, rgba(182,242,58,.18), transparent 60%),
              linear-gradient(135deg, #0f1a15 0%, #0a100d 100%); }
#hero .tag { font-family: 'IBM Plex Mono', monospace; font-size: 12px; letter-spacing: .18em;
  color: var(--lime); text-transform: uppercase; display: flex; align-items: center; gap: 10px; }
#hero .tag .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--lime);
  box-shadow: 0 0 0 0 rgba(182,242,58,.7); animation: pulse 2s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(182,242,58,.6); } 70% { box-shadow: 0 0 0 10px rgba(182,242,58,0); } 100% { box-shadow: 0 0 0 0 rgba(182,242,58,0); } }
#hero h1 { font-size: 40px; line-height: 1.1; margin: 12px 0 6px; color: var(--ink); font-weight: 700; letter-spacing: -.01em; }
#hero h1 span { color: var(--lime); }
#hero p { color: var(--muted); margin: 0; font-size: 16px; }
#hero .stats { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
#hero .stat { border: 1px solid var(--line); background: rgba(255,255,255,.02); border-radius: 12px; padding: 10px 14px; min-width: 120px; }
#hero .stat b { display: block; font-family: 'IBM Plex Mono', monospace; font-size: 22px; color: var(--ink); }
#hero .stat small { color: var(--muted); font-size: 12px; letter-spacing: .06em; text-transform: uppercase; }
#hero .drone { position: absolute; right: 28px; top: 22px; width: 150px; opacity: .9; }

/* ---------- panels ---------- */
.panel { border: 1px solid var(--line) !important; border-radius: 16px !important;
  background: var(--panel) !important; padding: 14px !important; }
.section-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: .16em;
  color: var(--muted); text-transform: uppercase; margin: 2px 0 6px; }
#run-btn { background: var(--lime) !important; color: #0a0f0d !important; font-weight: 700 !important;
  font-size: 17px !important; border: none !important; border-radius: 12px !important; height: 52px;
  box-shadow: 0 8px 30px -10px rgba(182,242,58,.6); transition: transform .15s ease, box-shadow .15s ease; }
#run-btn:hover { transform: translateY(-1px); box-shadow: 0 12px 34px -10px rgba(182,242,58,.8); }
.img-out img { border-radius: 10px; }

/* ---------- results ---------- */
#results .empty { color: var(--muted); text-align: center; padding: 26px; border: 1px dashed var(--line); border-radius: 14px; }
#results .kpis { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; }
#results .kpi { background: var(--panel-2); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; }
#results .kpi small { color: var(--muted); font-size: 12px; }
#results .kpi b { display: block; font-family: 'IBM Plex Mono', monospace; font-size: 30px; color: var(--ink); margin-top: 2px; }
#results .kpi.hot b { color: var(--red); }
#results .kpi em { font-style: normal; font-size: 12px; color: var(--muted); }
#results .bar { display: flex; height: 12px; border-radius: 999px; overflow: hidden; background: #1a2621; margin: 16px 0 10px; }
#results .bar i { display: block; height: 100%; }
#results .levels { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 10px; }
#results .lv { display: flex; align-items: center; gap: 10px; background: var(--panel-2); border: 1px solid var(--line); border-radius: 12px; padding: 10px 12px; }
#results .lv .sw { width: 12px; height: 34px; border-radius: 4px; flex: none; }
#results .lv b { color: var(--ink); font-size: 15px; display: block; }
#results .lv span { color: var(--muted); font-size: 12px; }
#results .lv .n { margin-left: auto; font-family: 'IBM Plex Mono', monospace; color: var(--ink); font-size: 20px; text-align: right; }
#results .note { color: var(--muted); font-size: 12px; margin-top: 10px; }
#credit { color: var(--muted); font-size: 13px; text-align: center; padding: 10px 0 4px; }
#credit b { color: var(--ink); font-weight: 600; }
@media (max-width: 760px) {
  #hero { padding: 22px 18px; } #hero h1 { font-size: 30px; } #hero .drone { display: none; }
  #results .kpis { grid-template-columns: repeat(2, minmax(0,1fr)); } #results .levels { grid-template-columns: 1fr; }
}
"""

DRONE_SVG = """<svg class="drone" viewBox="0 0 160 110" fill="none" stroke="#b6f23a" stroke-width="2">
 <ellipse cx="30" cy="22" rx="24" ry="5" opacity=".55"/><ellipse cx="130" cy="22" rx="24" ry="5" opacity=".55"/>
 <path d="M30 27v8M130 27v8M30 35h100" /><rect x="60" y="30" width="40" height="18" rx="6" fill="#0e1512"/>
 <circle cx="80" cy="56" r="7"/><path d="M68 48l-10 18M92 48l10 18M52 66h14M94 66h14"/>
 <path d="M80 63l-34 44M80 63l34 44" stroke-dasharray="3 5" opacity=".45"/></svg>"""

def hero_html():
    stats = ''.join(f'<div class="stat"><b>{v}</b><small>{k}</small></div>' for k, v in MODEL_STATS)
    return f"""<div id="hero">{DRONE_SVG}
      <div class="tag"><span class="dot"></span>KMITL · Precision Agriculture · {MODEL_NAME} + SAHI</div>
      <h1>Weed<span>Scan</span> — ตรวจจับวัชพืชในนาข้าว</h1>
      <p>อัปโหลดภาพจากโดรน แล้วระบบจะหาตำแหน่งวัชพืชและสร้างแผนที่กำหนดการพ่นแบบเฉพาะจุด</p>
      <div class="stats">{stats}</div></div>"""

def rgb(c): return f'rgb({c[0]},{c[1]},{c[2]})'

def results_html(dets, level, seconds, use_sahi):
    if dets is None:
        return '<div id="results"><div class="empty">ผลการวิเคราะห์จะแสดงที่นี่ · เลือกภาพแล้วกด “สแกนวัชพืช”</div></div>'
    total = level.size
    confs = [d[4] for d in dets]
    pct = [100 * (level == i).sum() / total for i in range(len(LEVELS))]
    counts = [sum(1 for d in dets if level_of(d[4]) == i) for i in range(len(LEVELS))]
    kpis = [
        ('จุดที่พบ', f'{len(dets)}', 'ตำแหน่งวัชพืช', ''),
        ('ต้องพ่นทันที', f'{pct[0]:.1f}%', 'ของพื้นที่ภาพ', 'hot'),
        ('Conf สูงสุด', f'{max(confs, default=0):.2f}', f'เฉลี่ย {np.mean(confs) if confs else 0:.2f}', ''),
        ('เวลาประมวลผล', f'{seconds:.1f}s', 'SAHI' if use_sahi else 'YOLOv8s ปกติ', ''),
    ]
    k = ''.join(f'<div class="kpi {c}"><small>{a}</small><b>{b}</b><em>{e}</em></div>' for a, b, e, c in kpis)
    bar = ''.join(f'<i style="width:{p:.2f}%;background:{rgb(LEVELS[i][3])}"></i>' for i, p in enumerate(pct))
    lv = ''.join(
        f'<div class="lv"><div class="sw" style="background:{rgb(L[3])}"></div>'
        f'<div><b>{L[4]}</b><span>ระดับ{L[0]} · กรอบ <span style="color:{rgb(L[2])}">■</span> · พื้นที่ {pct[i]:.1f}%</span></div>'
        f'<div class="n">{counts[i]}</div></div>' for i, L in enumerate(LEVELS))
    return (f'<div id="results"><div class="kpis">{k}</div><div class="bar">{bar}</div>'
            f'<div class="levels">{lv}</div>'
            f'<div class="note">ระดับ: สูง conf ≥ 0.70 · กลาง 0.50–0.69 · ต่ำ &lt; 0.50 — สัดส่วนพื้นที่คิดจากช่องกริดในภาพนี้เท่านั้น ยังไม่ใช่ปริมาณสารที่ลดได้จริงในแปลง</div></div>')

def run(img, use_sahi, conf_th, grid_cols):
    if img is None:
        return None, None, results_html(None, None, 0, use_sahi)
    img = np.asarray(img)[:, :, :3]
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        k = MAX_SIDE / max(h, w)
        img = np.asarray(Image.fromarray(img).resize((round(w * k), round(h * k)), Image.LANCZOS))
    t0 = time.time()
    dets = detect(img, use_sahi, conf_th)
    seconds = time.time() - t0
    det_img = draw_detections(img, dets)
    pmap, level = prescription_map(img, dets, int(grid_cols))
    return det_img, pmap, results_html(dets, level, seconds, use_sahi)

def make_theme(gr):
    t = gr.themes.Base(
        primary_hue=gr.themes.colors.lime, neutral_hue=gr.themes.colors.zinc,
        font=[gr.themes.GoogleFont('IBM Plex Sans Thai'), 'sans-serif'],
        font_mono=[gr.themes.GoogleFont('IBM Plex Mono'), 'monospace'],
        radius_size=gr.themes.sizes.radius_lg)
    v = dict(
        body_background_fill='#070b09', body_text_color='#e8f0ea', body_text_color_subdued='#8a9a91',
        background_fill_primary='#0e1512', background_fill_secondary='#121c18',
        block_background_fill='#0e1512', block_border_color='#1f2d27', block_label_background_fill='#121c18',
        block_label_text_color='#b6f23a', block_title_text_color='#e8f0ea', border_color_primary='#1f2d27',
        input_background_fill='#121c18', input_border_color='#1f2d27',
        slider_color='#b6f23a', checkbox_background_color_selected='#b6f23a', checkbox_border_color_selected='#b6f23a',
        color_accent_soft='#1a2621', button_secondary_background_fill='#121c18', button_secondary_text_color='#e8f0ea')
    v.update({f'{key}_dark': val for key, val in v.items()})
    return t.set(**v)

def gradio_major():
    import gradio as gr
    return int(gr.__version__.split('.')[0])

def style_kwargs():
    """Gradio 6 ย้าย theme/css จาก gr.Blocks() ไปไว้ที่ launch()"""
    import gradio as gr
    return dict(theme=make_theme(gr), css=CSS)

def build_app():
    import gradio as gr
    examples = sorted(glob.glob(os.path.join(EXAMPLE_DIR, '*.[jJpP][pPnN]*[gG]')))[:6]
    on_gpu = False
    try:
        import torch; on_gpu = torch.cuda.is_available()
    except Exception:
        pass
    blocks_kw = style_kwargs() if gradio_major() < 6 else {}
    with gr.Blocks(title='WeedScan · ตรวจจับวัชพืชในนาข้าว', **blocks_kw) as app:
        gr.HTML(hero_html())
        with gr.Row(equal_height=False):
            with gr.Column(scale=4, elem_classes='panel'):
                gr.HTML('<div class="section-label">01 · Input</div>')
                inp = gr.Image(label='ภาพจากโดรน', type='numpy', height=300)
                use_sahi = gr.Checkbox(value=on_gpu, label='ใช้ SAHI — จับวัชพืชต้นเล็กได้ดีขึ้น' + ('' if on_gpu else ' (บน CPU ใช้เวลา 10–30 วินาที)'))
                conf = gr.Slider(0.10, 0.90, value=0.25, step=0.05, label='Confidence threshold')
                grid = gr.Slider(6, 30, value=12, step=1, label='ความละเอียดกริดแผนที่ (คอลัมน์)')
                btn = gr.Button('สแกนวัชพืช  →', elem_id='run-btn')
                if examples:
                    gr.Examples(examples, inputs=inp, label='ภาพตัวอย่างจากแปลงนาไทย')
            with gr.Column(scale=8, elem_classes='panel'):
                gr.HTML('<div class="section-label">02 · Detection & Prescription Map</div>')
                with gr.Row():
                    out_det = gr.Image(label='ผลการตรวจจับ', elem_classes='img-out', height=330)
                    out_map = gr.Image(label='แผนที่กำหนดการพ่น', elem_classes='img-out', height=330)
                gr.HTML('<div class="section-label">03 · Summary</div>')
                out_html = gr.HTML(results_html(None, None, 0, True))
        gr.HTML('<div id="credit">ปริญญานิพนธ์ <b>การพัฒนาระบบตรวจจับวัชพืชในนาข้าวด้วยโดรนและการเรียนรู้เชิงลึก</b> · '
                'นายอรรถวิทย์ หมัดสอิ๊ด · อาจารย์ที่ปรึกษา ผศ.ดร.วสุ อุดมเพทายกุล · สจล.</div>')
        btn.click(run, [inp, use_sahi, conf, grid], [out_det, out_map, out_html])
    return app

def launch(**kw):
    """เรียกใช้แทน build_app().launch() — จัดการความต่างของ Gradio 5/6 ให้"""
    app = build_app()
    if gradio_major() >= 6:
        kw = {**style_kwargs(), **kw}
    app.queue(default_concurrency_limit=1).launch(**kw)

if __name__ == '__main__':
    launch()   # Hugging Face Spaces

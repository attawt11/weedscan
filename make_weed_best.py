# ย่อไฟล์โมเดล v7 ให้เล็กลง (ตัด optimizer ออก ~55 MB → ~19 MB) แล้วตั้งชื่อเป็น weed_best.pt สำหรับเว็บเดโม
# รันใน Kaggle/Colab หรือเครื่องที่ติดตั้ง ultralytics >= 8.4:  python make_weed_best.py v7_best_sofar.pt
import sys, shutil, os
from ultralytics.utils.torch_utils import strip_optimizer
src = sys.argv[1] if len(sys.argv) > 1 else 'v7_best_sofar.pt'
shutil.copy(src, 'weed_best.pt'); strip_optimizer('weed_best.pt')
print('weed_best.pt', round(os.path.getsize('weed_best.pt') / 1e6, 1), 'MB')
from ultralytics import YOLO; m = YOLO('weed_best.pt'); print('โหลดได้ ✅', m.names)

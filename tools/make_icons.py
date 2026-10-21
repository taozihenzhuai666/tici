# 生成桌面图标：python tools/make_icons.py
# 画面铺满（像数学岛的小水獭图标）：顶上是前置镜头，下面几行台词，正在读的一行最亮、带红箭头。
# 换图后把文件名里的版本号加一（同时改 index.html 和 manifest.webmanifest），苹果按网址缓存图标，不换名会一直用旧图。
import pathlib
from playwright.sync_api import sync_playwright

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 512 坐标系；k = 画面缩放（maskable 会被安卓裁成圆，安全区约 80%）
SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="{s}" height="{s}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#ff7a45"/><stop offset="1" stop-color="#ff3d5a"/>
    </linearGradient>
  </defs>
  <rect width="512" height="512" fill="url(#bg)"/>
  <g transform="translate(256 256) scale({k}) translate(-256 -256)">
    <!-- front camera -->
    <circle cx="256" cy="58" r="30" fill="#1a1d26"/>
    <circle cx="256" cy="58" r="12" fill="#3b4256"/>
    <!-- script lines: read, current, next, next -->
    <rect x="40" y="128" width="432" height="58" rx="29" fill="#fff" opacity=".45"/>
    <rect x="40" y="218" width="432" height="76" rx="38" fill="#fff"/>
    <rect x="40" y="326" width="360" height="58" rx="29" fill="#fff" opacity=".7"/>
    <rect x="40" y="416" width="250" height="58" rx="29" fill="#fff" opacity=".45"/>
    <!-- reading arrow on the current line -->
    <path d="M70 230 L118 256 L70 282 Z" fill="#ff3d5a"/>
  </g>
</svg>"""

ICONS = [('tici-icon-v1-180.png', 180, 1.0), ('tici-icon-v1-192.png', 192, 1.0),
         ('tici-icon-v1-512.png', 512, 1.0), ('tici-icon-v1-maskable.png', 512, .78)]

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    for name, size, k in ICONS:
        pg.set_viewport_size({'width': size, 'height': size})
        pg.set_content(f"<style>html,body{{margin:0;overflow:hidden}}svg{{display:block}}</style>{SVG.format(s=size, k=k)}")
        pg.screenshot(path=str(ROOT / name))
        print(name, size)
    b.close()

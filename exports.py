from io import BytesIO
import base64
import html
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def png_bytes(array):
    buf = BytesIO()
    Image.fromarray(array).save(buf, format='PNG')
    return buf.getvalue()


def color_dashboard(result):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout='constrained')
    values = list(result.roi_colors.values())
    axes[0].bar(['Red', 'Dark', 'Other'], values, color=['#f06470','#475569','#2dd4bf'])
    axes[0].set(title='Colors within ROI (exclusive groups)', ylabel='Pixels (%)', ylim=(0,100))
    axes[1].plot(range(180), result.hue_hist, color='#0d9488')
    axes[1].set(title='Hue inside selected candidate', xlabel='OpenCV Hue (0–179)', ylabel='Pixels')
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=160); plt.close(fig)
    return buf.getvalue()


def trend_png(records):
    fig, ax = plt.subplots(figsize=(9, 3.6), layout='constrained')
    ax.plot(range(1,len(records)+1), [r['area_px2'] for r in records], 'o-', color='#0d9488')
    ax.set(xlabel='Record order', ylabel='Candidate area (px²)', title='Pixel area trend (not a healing rate)')
    ax.set_xticks(range(1,len(records)+1)); ax.grid(alpha=.2)
    buf = BytesIO(); fig.savefig(buf, format='png', dpi=160); plt.close(fig)
    return buf.getvalue()


def html_report(result, name):
    m = result.metrics
    rows = ''.join(f'<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>' for k,v in m.items())
    encoded = base64.b64encode(png_bytes(result.overlay)).decode()
    dashboard = base64.b64encode(color_dashboard(result)).decode()
    return f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Wound Analyzer</title>
<style>body{{font:16px sans-serif;max-width:1000px;margin:40px auto;padding:20px;color:#123}}
th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}</style>
<h1>Wound Analyzer 분석 보고서</h1><p>{html.escape(name)}</p>
<p>교육용 색상 기반 후보 분석입니다. px²는 처리 영상의 픽셀 면적이며 실제 면적이나 진단 결과가 아닙니다.</p>
<p>면적은 가장 큰 외곽선의 기하학적 면적입니다. 내부 구멍은 포함됩니다. 색상 비율은 픽셀 개수 기준입니다.</p>
<img src="data:image/png;base64,{encoded}"><table>{rows}</table><h2>색상 분석</h2>
<img src="data:image/png;base64,{dashboard}"><p>{html.escape(' / '.join(result.warnings))}</p></html>'''

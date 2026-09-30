"""Streamlit recreation of the Wound Analyzer described in the assignment report."""
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from wound_processing import load_image, resize_image, analyze_wound, sample_image, compare_results
from exports import png_bytes, color_dashboard, trend_png, html_report

st.set_page_config(page_title="Wound Analyzer", page_icon="🩹", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#07111f;color:#edf3ff}
[data-testid="stSidebar"]{background:#0a1427;border-right:1px solid #203152}
.block-container{max-width:1600px;padding-top:.8rem;padding-bottom:2rem}
.wa-title{font-size:1.08rem;font-weight:800;margin-bottom:.2rem}
.wa-sub{font-size:.72rem;color:#9fb0cc;margin-bottom:.45rem}
.wa-guide{background:#132855;border:1px solid #31508e;border-radius:5px;padding:8px 12px;color:#c9d8f8;font-size:.78rem;margin-bottom:.6rem}
.wa-section{font-size:.82rem;font-weight:800;color:#dfe9ff;margin:.55rem 0 .35rem;padding-bottom:.25rem;border-bottom:1px solid #24365c}
.wa-card{background:#0d1830;border:1px solid #263a63;border-radius:7px;padding:12px 14px;min-height:92px}
.wa-label{color:#8fa7d0;font-size:.72rem;margin-bottom:5px}
.wa-value{color:white;font-size:1.1rem;font-weight:800}
.wa-small{color:#9fb0cc;font-size:.69rem;margin-top:5px;line-height:1.4}
.wa-log{background:#050b15;border:1px solid #22314f;border-radius:6px;padding:10px 12px;font:12px Consolas,monospace;color:#c8d5ed;white-space:pre-wrap}
[data-testid="stFileUploaderDropzone"]{background:#0d1830;border:1px dashed #365282;padding:.45rem}
[data-testid="stMetric"]{background:#0d1830;border:1px solid #263a63;padding:.7rem;border-radius:7px}
div[data-testid="stButton"]>button,div[data-testid="stDownloadButton"]>button{border-radius:5px;border:1px solid #4779ec;background:#2764e8;color:white;min-height:2.15rem;font-weight:700}
button[data-baseweb="tab"]{font-size:.78rem}
</style>
""", unsafe_allow_html=True)

for k,v in {
    "result":None, "previous_result":None, "comparison":None, "records":[],
    "logs":["[SYSTEM] Wound Analyzer 시작"], "roi_mode":False, "roi_reset":0,
    "last_current_hash":None, "last_previous_hash":None
}.items():
    st.session_state.setdefault(k,v)

def add_log(msg):
    stamp=datetime.now(ZoneInfo("Asia/Seoul")).strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{stamp}] {msg}")
    st.session_state.logs=st.session_state.logs[-100:]

def card(label,value,sub=""):
    st.markdown(f"<div class='wa-card'><div class='wa-label'>{label}</div><div class='wa-value'>{value}</div><div class='wa-small'>{sub}</div></div>",unsafe_allow_html=True)

def confidence_for(r):
    m=r.metrics
    if not m["detected"]: return "Low","후보 영역이 검출되지 않았습니다."
    score=0; reasons=[]
    if .4 <= m["image_ratio_pct"] <= 30: score+=1
    else: reasons.append("후보 면적 비율이 비정상적")
    if m["compactness"] >= .25: score+=1
    else: reasons.append("경계 형태가 불규칙")
    if m["candidate_count"] == 1: score+=1
    else: reasons.append("복수 후보 검출")
    return ("Medium" if score>=2 else "Low"), ("HSV·형태 기반 휴리스틱" + (" · "+", ".join(reasons) if reasons else ""))

def contour_view(result):
    rgb=result.original.copy()
    mask=(result.candidate_mask>0).astype(np.uint8)*255
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c=max(contours,key=cv2.contourArea)
        cv2.drawContours(rgb,[c],-1,(255,220,0),3)
    return rgb

def safe_csv(records):
    if not records: return b""
    df=pd.DataFrame(records).copy()
    if "image" in df:
        df["image"]=df["image"].map(lambda x:"'"+x if str(x).startswith(("=","+","-","@","\t","\r")) else x)
    return df.to_csv(index=False).encode("utf-8-sig")

with st.sidebar:
    st.markdown("<div class='wa-title'>Wound Analyzer</div>",unsafe_allow_html=True)
    st.markdown("<div class='wa-sub'>PyQt + OpenCV 과제 프로그램의 Streamlit 재구현</div>",unsafe_allow_html=True)

    st.markdown("<div class='wa-section'>1단계 사진 선택</div>",unsafe_allow_html=True)
    up_cur=st.file_uploader("현재 사진 불러오기",type=["png","jpg","jpeg","bmp","webp"],key="cur")
    up_prev=st.file_uploader("비교용 이전 사진 불러오기",type=["png","jpg","jpeg","bmp","webp"],key="prev")
    with st.expander("시연용 샘플 이미지",expanded=up_cur is None):
        demo=st.toggle("샘플 이미지 사용",value=up_cur is None,key="demo")
        demo_cur=st.selectbox("현재 샘플",["현재","이전","그림자","어두운 후보","배경 오검출","작은 후보"])
        demo_prev=st.selectbox("이전 샘플",["이전","현재","그림자","어두운 후보","배경 오검출","작은 후보"])

    st.markdown("<div class='wa-section'>2단계 분석 범위 보정</div>",unsafe_allow_html=True)
    roi_mode=st.toggle("분석 범위 선택 켜기",key="roi_mode")
    if st.button("선택 범위 해제",use_container_width=True):
        st.session_state.roi_mode=False; st.session_state.roi_reset+=1
        add_log("ROI 선택 범위를 해제했습니다."); st.rerun()

    st.markdown("<div class='wa-section'>3단계 자동 검출 조절</div>",unsafe_allow_html=True)
    sensitivity=st.slider("색상 민감도",0,100,55)
    min_area=st.number_input("작은 점 무시",0,200000,300,50)
    kernel=st.select_slider("잡음 정리 강도",options=[1,3,5,7,9],value=5)

def resolve(upload,use_demo,kind):
    if upload is not None:
        b=upload.getvalue()
        return load_image(b),upload.name,hashlib.sha256(b).hexdigest()[:16]
    if use_demo:
        im=sample_image(kind)
        return im,f"sample_{kind}.png",hashlib.sha256(im.tobytes()).hexdigest()[:16]
    return None,None,None

try:
    cur,cur_name,cur_hash=resolve(up_cur,demo,demo_cur)
    prev,prev_name,prev_hash=resolve(up_prev,(up_prev is None and demo),demo_prev)
except Exception as e:
    st.error(f"이미지를 불러오지 못했습니다: {e}"); st.stop()

if cur_hash!=st.session_state.last_current_hash:
    st.session_state.last_current_hash=cur_hash; st.session_state.result=None; st.session_state.comparison=None; st.session_state.roi_reset+=1
    if cur_name: add_log(f"현재 사진 로드: {cur_name}")
if prev_hash!=st.session_state.last_previous_hash:
    st.session_state.last_previous_hash=prev_hash; st.session_state.previous_result=None; st.session_state.comparison=None
    if prev_name: add_log(f"이전 사진 로드: {prev_name}")

st.markdown("<div class='wa-title'>Wound Analyzer</div>",unsafe_allow_html=True)
st.markdown("<div class='wa-guide'>사용 흐름: 현재 사진 불러오기 → 필요 시 ROI 선택 → 색상 민감도/작은 점 무시 조절 → 현재 사진 분석하기 → 결과 확인 및 저장</div>",unsafe_allow_html=True)

tab_summary,tab_photo,tab_process,tab_color,tab_record,tab_log=st.tabs(
    ["결과 요약","사진 결과","분석 과정","색상 그래프","기록/그래프","처리 로그"]
)

roi=None
analysis_rgb=resize_image(cur) if cur is not None else None
with tab_photo:
    st.markdown("#### 원본 / 분석 범위")
    if analysis_rgb is None:
        st.info("왼쪽에서 현재 사진을 불러와 주세요.")
    elif roi_mode:
        st.caption("파란 사각형의 모서리 또는 내부를 드래그하여 분석 범위를 지정하세요.")
        h,w=analysis_rgb.shape[:2]
        default=(int(w*.2),int(w*.8),int(h*.2),int(h*.8))
        box=st_cropper(Image.fromarray(analysis_rgb),realtime_update=True,default_coords=default,box_color="#25a7ff",aspect_ratio=None,return_type="box",stroke_width=3,key=f"crop_{cur_hash}_{st.session_state.roi_reset}")
        x=int(box["left"]); y=int(box["top"]); bw=int(box["width"]); bh=int(box["height"])
        roi=(x,y,x+bw,y+bh)
        st.caption(f"선택 ROI: x={x}~{x+bw}, y={y}~{y+bh} / {bw}×{bh}px")
    else:
        st.image(analysis_rgb,caption=cur_name or "현재 사진",use_container_width=True)
        st.caption("ROI 선택이 꺼져 있어 사진 전체를 분석합니다.")

with st.sidebar:
    st.markdown("<div class='wa-section'>4단계 분석 / 저장</div>",unsafe_allow_html=True)
    do_analyze=st.button("현재 사진 분석하기",type="primary",use_container_width=True,disabled=analysis_rgb is None)
    do_compare=st.button("이전 사진과 크기 변화 비교",use_container_width=True,disabled=analysis_rgb is None or prev is None)
    do_add=st.button("현재 결과를 기록표에 추가",use_container_width=True,disabled=st.session_state.result is None)

    r=st.session_state.result
    recs=st.session_state.records
    result_png=png_bytes(r.overlay) if r else b""
    dash_png=color_dashboard(r) if r else b""
    report=html_report(r,cur_name or "image") if r else ""
    st.download_button("결과 사진 저장",result_png,"wound_result.png","image/png",use_container_width=True,disabled=r is None)
    st.download_button("쉬운 해석 보고서 저장",report.encode("utf-8") if isinstance(report,str) else report,"wound_report.html","text/html",use_container_width=True,disabled=r is None)
    st.download_button("기록표 CSV 저장",safe_csv(recs),"wound_records.csv","text/csv",use_container_width=True,disabled=not recs)
    area_graph=trend_png(recs) if len(recs)>=2 else b""
    st.download_button("크기 변화 그래프 저장",area_graph,"wound_area_trend.png","image/png",use_container_width=True,disabled=len(recs)<2)
    st.download_button("색상 분석 그래프 저장",dash_png,"wound_color_dashboard.png","image/png",use_container_width=True,disabled=r is None)
    st.markdown("<div class='wa-small'>면적은 px² 단위이며 실제 cm²가 아닙니다.</div>",unsafe_allow_html=True)

if do_analyze:
    try:
        rr=analyze_wound(analysis_rgb,int(sensitivity),int(min_area),int(kernel),roi)
        st.session_state.result=rr; st.session_state.comparison=None
        add_log(f"현재 사진 분석 완료: {cur_name} / {rr.metrics['seconds']:.4f} sec"); st.rerun()
    except Exception as e:
        add_log(f"분석 오류: {e}"); st.error(f"분석 중 오류가 발생했습니다: {e}")

if do_compare:
    try:
        rr=st.session_state.result or analyze_wound(analysis_rgb,int(sensitivity),int(min_area),int(kernel),roi)
        prev_rgb=resize_image(prev)
        proi=None
        if roi is not None:
            ch,cw=analysis_rgb.shape[:2]; ph,pw=prev_rgb.shape[:2]
            proi=(round(roi[0]*pw/cw),round(roi[1]*ph/ch),round(roi[2]*pw/cw),round(roi[3]*ph/ch))
        pr=analyze_wound(prev_rgb,int(sensitivity),int(min_area),int(kernel),proi)
        st.session_state.result=rr; st.session_state.previous_result=pr; st.session_state.comparison=compare_results(pr,rr)
        add_log(f"이전/현재 크기 비교 완료: {prev_name} → {cur_name}"); st.rerun()
    except Exception as e:
        add_log(f"비교 오류: {e}"); st.error(f"비교 중 오류가 발생했습니다: {e}")

if do_add and st.session_state.result is not None:
    rr=st.session_state.result; m=rr.metrics
    conf,_=confidence_for(rr)
    st.session_state.records.append({
        "time":datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        "image":cur_name or "image",
        "area_px2":m["area_px2"],"perimeter_px":m["perimeter_px"],"image_ratio_pct":m["image_ratio_pct"],
        "analysis_red_pct":rr.roi_colors["red"],"analysis_dark_pct":rr.roi_colors["dark"],
        "candidate_red_pct":rr.candidate_colors["red"],"confidence":conf,"seconds":m["seconds"]
    })
    add_log(f"기록표 추가: {cur_name} / {m['area_px2']:.1f} px²"); st.rerun()

r=st.session_state.result
with tab_summary:
    st.markdown("#### 최종 분석 결과")
    if r is None:
        st.info("왼쪽의 '현재 사진 분석하기' 버튼을 누르면 결과가 표시됩니다.")
    else:
        m=r.metrics; conf,why=confidence_for(r)
        c=st.columns(4)
        with c[0]: card("후보 영역 크기",f"{m['area_px2']:,.1f} px²","가장 큰 유효 후보 contour")
        with c[1]: card("사진 내 비율",f"{m['image_ratio_pct']:.2f}%","전체 처리 이미지 기준")
        with c[2]: card("분석 범위 내 붉은색",f"{r.roi_colors['red']:.2f}%","ROI 또는 전체 사진 기준")
        with c[3]: card("자동 검출 믿음도",conf,why)
        q=st.columns(4)
        q[0].metric("경계선 길이",f"{m['perimeter_px']:,.1f} px")
        q[1].metric("어두운 부분 비율",f"{r.roi_colors['dark']:.2f}%")
        q[2].metric("후보 내부 붉은색",f"{r.candidate_colors['red']:.2f}%")
        q[3].metric("처리 시간",f"{m['seconds']:.4f} sec")
        for w in r.warnings: st.warning(w)
        if st.session_state.comparison is not None:
            st.divider(); st.markdown("#### 이전 사진과 크기 변화 비교")
            st.metric("후보 영역 크기 변화율",f"{st.session_state.comparison:+.2f}%","감소" if st.session_state.comparison<0 else "증가")
            st.caption("픽셀 면적 변화율이며 회복률이 아닙니다.")

with tab_photo:
    if r is not None:
        st.divider(); st.markdown("#### 사진 결과")
        a,b,c=st.tabs(["원본 이미지","최종 결과 이미지","경계선 확인 이미지"])
        with a: st.image(r.original,use_container_width=True)
        with b: st.image(r.overlay,use_container_width=True)
        with c: st.image(contour_view(r),use_container_width=True)
        if st.session_state.previous_result is not None:
            x,y=st.columns(2)
            x.image(st.session_state.previous_result.overlay,caption="이전 사진",use_container_width=True)
            y.image(r.overlay,caption="현재 사진",use_container_width=True)

with tab_process:
    st.markdown("#### 분석 과정")
    if r is None:
        st.info("분석을 실행하면 ROI → 색상 Mask → Morphology → Contour 결과를 단계별로 확인할 수 있습니다.")
    else:
        st.caption("BGR/RGB → HSV → 색상 기반 Mask → Morphology Open/Close → Contour → 가장 큰 유효 후보")
        a,b,c,d=st.tabs(["ROI","색으로 찾기 Mask","잡음 정리 Mask","경계선 결과"])
        with a: st.image(r.preview,use_container_width=True)
        with b: st.image(r.raw_mask,clamp=True,use_container_width=True)
        with c: st.image(r.clean_mask,clamp=True,use_container_width=True)
        with d: st.image(contour_view(r),use_container_width=True)

with tab_color:
    st.markdown("#### 색상 분석 대시보드")
    if r is None:
        st.info("분석 결과가 생성되면 색상 비율과 Hue 분포가 표시됩니다.")
    else:
        st.image(color_dashboard(r),use_container_width=True)
        m=r.metrics
        st.markdown(f"""- 분석 범위 내 붉은색 비율: **{r.roi_colors['red']:.2f}%**
- 분석 범위 내 어두운 부분 비율: **{r.roi_colors['dark']:.2f}%**
- 후보 영역 내부 붉은색 비율: **{r.candidate_colors['red']:.2f}%**
- 후보 영역 내부 어두운 부분 비율: **{r.candidate_colors['dark']:.2f}%**
- 평균 HSV: **{('-' if m['mean_h'] is None else f"H {m['mean_h']:.1f} / S {m['mean_s']:.1f} / V {m['mean_v']:.1f}")}**""")

with tab_record:
    st.markdown("#### 누적 분석 기록")
    if not st.session_state.records:
        st.info("'현재 결과를 기록표에 추가' 버튼을 누르면 기록이 누적됩니다.")
    else:
        st.dataframe(pd.DataFrame(st.session_state.records),use_container_width=True,hide_index=True)
        if len(st.session_state.records)>=2:
            st.image(trend_png(st.session_state.records),use_container_width=True)
            st.line_chart(pd.DataFrame(st.session_state.records)[["analysis_red_pct","analysis_dark_pct"]])
        if st.button("기록 전체 지우기"):
            st.session_state.records=[]; add_log("누적 기록을 모두 지웠습니다."); st.rerun()

with tab_log:
    logs="\n".join(st.session_state.logs).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    st.markdown("#### 처리 로그")
    st.markdown(f"<div class='wa-log'>{logs}</div>",unsafe_allow_html=True)

st.caption("Wound Analyzer · OpenCV HSV 기반 상처 후보 영역 분석 · 교육/발표용 프로그램")

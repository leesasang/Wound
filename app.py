"""Streamlit entry point: streamlit run app.py"""
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from wound_processing import load_image, resize_image, analyze_wound, sample_image, compare_results
from exports import png_bytes, color_dashboard, trend_png, html_report

st.set_page_config(page_title='Wound Analyzer', page_icon='🩹', layout='wide')
st.markdown('''<style>.block-container{padding-top:2rem;max-width:1450px}h1{letter-spacing:-1.5px}
[data-testid="stMetric"]{background:white;border:1px solid #dce7ed;border-radius:14px;padding:18px}
[data-testid="stSidebar"]{border-right:1px solid #dce7ed}</style>''', unsafe_allow_html=True)
st.caption('AIDEV · COMPUTER VISION LAB')
st.title('Wound Analyzer')
st.write('사진 속 상처 후보 영역을 찾고, 크기와 색상의 변화를 살펴보세요.')
st.caption('교육·발표용 / OpenCV 색상 기반 분석 / 임상 검증되지 않은 후보 영역이며 진단·치료 판단용이 아닙니다.')

with st.sidebar:
    st.header('01 · 사진 선택')
    source = st.radio('입력 방식', ['시연 이미지', '사진 업로드'])
    if source == '시연 이미지':
        kind = st.selectbox('합성 시연 이미지', ['현재','이전','그림자','어두운 후보','배경 오검출','작은 후보'])
        original, name = sample_image(kind), f'합성 시연 · {kind}'
        st.caption('도형으로 만든 합성 이미지입니다. 실제 상처나 성능 검증 자료가 아닙니다.')
    else:
        upload = st.file_uploader('현재 사진', type=['png','jpg','jpeg','webp'])
        if upload is None:
            st.info('현재 사진을 업로드해 주세요.'); st.stop()
        try:
            original, name = load_image(upload.getvalue()), upload.name
        except Exception as exc:
            st.error(f'이미지를 읽을 수 없습니다: {exc}'); st.stop()
    st.header('02 · 검출 설정')
    sensitivity = st.slider('색상 민감도', 0, 100, 55, help='값이 클수록 붉은색으로 인정하는 색상 범위를 넓히고 최소 채도를 낮춥니다.')
    min_area = st.number_input('작은 후보 무시 (px²)', 0, 1000000, 300, 50)
    kernel = st.select_slider('잡음 제거 커널', options=[1,3,5,7,9], value=5)
    st.caption('모든 면적은 최대 변 1,200px로 축소한 처리 영상 기준입니다.')
    compare = st.toggle('이전 사진과 비교')
    previous_original = None
    if compare:
        if source == '시연 이미지':
            previous_original, previous_name = sample_image('이전'), '합성 시연 · 이전'
        else:
            previous_upload = st.file_uploader('이전 사진', type=['png','jpg','jpeg','webp'])
            if previous_upload:
                try:
                    previous_original, previous_name = load_image(previous_upload.getvalue()), previous_upload.name
                except Exception as exc:
                    st.error(f'이전 사진을 읽을 수 없습니다: {exc}')

rgb = resize_image(original)
identity = hashlib.sha256(rgb.tobytes()).hexdigest()[:16]

def roi_control(im, key, title):
    h, w = im.shape[:2]
    st.subheader(title)
    use_roi = st.checkbox('분석 범위 제한', key=f'{key}_use')
    roi = None
    if use_roi:
        xs = st.slider('가로 범위 (%)', 0,100,(25,75), key=f'{key}_x')
        ys = st.slider('세로 범위 (%)', 0,100,(20,80), key=f'{key}_y')
        roi = (round(w*xs[0]/100),round(h*ys[0]/100),round(w*xs[1]/100),round(h*ys[1]/100))
    from PIL import Image, ImageDraw
    preview = Image.fromarray(im)
    if roi and roi[2]>roi[0] and roi[3]>roi[1]:
        ImageDraw.Draw(preview).rectangle((roi[0],roi[1],roi[2]-1,roi[3]-1),outline='#14b8a6',width=3)
    st.image(preview, caption=f'처리 해상도 {w} × {h}px', width='stretch')
    return roi

st.markdown('### 분석 범위 설정')
cols = st.columns(2 if previous_original is not None else 1)
with cols[0]:
    roi = roi_control(rgb, identity, '현재 사진')
previous_rgb = None
previous_roi = None
if previous_original is not None:
    previous_rgb = resize_image(previous_original)
    with cols[1]:
        previous_roi = roi_control(previous_rgb, 'previous_'+hashlib.sha256(previous_rgb.tobytes()).hexdigest()[:16], '이전 사진')

signature = hashlib.sha256(json.dumps([identity,roi,sensitivity,min_area,kernel,
    hashlib.sha256(previous_rgb.tobytes()).hexdigest() if previous_rgb is not None else None,previous_roi]).encode()).hexdigest()
if st.session_state.get('signature') != signature:
    st.session_state.pop('result',None)
    st.session_state.pop('previous_result',None)
if st.button('상처 후보 분석하기', type='primary', width='stretch'):
    try:
        with st.spinner('색상 분리와 경계선 분석 중…'):
            current = analyze_wound(rgb,sensitivity,min_area,kernel,roi)
            prior = analyze_wound(previous_rgb,sensitivity,min_area,kernel,previous_roi) if previous_rgb is not None else None
        st.session_state.update(result=current, previous_result=prior, signature=signature)
    except ValueError as exc:
        st.error(str(exc))

result = st.session_state.get('result')
st.session_state.setdefault('records',[])
if result:
    m = result.metrics
    st.markdown('### 분석 결과')
    cards = st.columns(4)
    for col, label, value in zip(cards, ['후보 면적','경계선 길이','사진 내 면적 비율','처리 시간'],
        [f"{m['area_px2']:,.1f} px²",f"{m['perimeter_px']:,.1f} px",f"{m['image_ratio_pct']:.2f}%",f"{m['seconds']*1000:.1f} ms"]):
        col.metric(label,value if m['detected'] or label=='처리 시간' else '미검출')
    for warning in result.warnings: st.warning(warning)
    prior = st.session_state.get('previous_result')
    if prior:
        change = compare_results(prior,result)
        st.info('픽셀 면적 비교입니다. 촬영 거리·각도·해상도·조명이 같아야 의미 있는 비교가 가능합니다. 감소율은 회복률이 아닙니다.')
        if change is None:
            st.warning('두 사진 모두 후보가 검출되어야 변화율을 계산할 수 있습니다.')
        else:
            st.metric('이전 대비 후보 면적 변화', f'{change:+.2f}%')
        pc = st.columns(2)
        pc[0].image(prior.overlay,caption='이전 분석',width='stretch')
        pc[1].image(result.overlay,caption='현재 분석',width='stretch')
    tabs = st.tabs(['사진 결과','분석 과정','색상 분석','결과 저장','처리 정보'])
    with tabs[0]:
        c = st.columns(2)
        c[0].image(result.original,caption='원본 · 처리 해상도',width='stretch')
        c[1].image(result.overlay,caption='가장 큰 후보 영역 · 경계선',width='stretch')
        st.caption('초록색으로 채워진 부분은 색상 조건을 만족하는 후보입니다. 실제 상처 영역과 다를 수 있습니다.')
    with tabs[1]:
        st.write('RGB → HSV → 색상 임계값 → Opening → Closing → Contour → 가장 큰 유효 후보')
        for col, data, caption in zip(st.columns(4),[result.preview,result.raw_mask,result.clean_mask,result.candidate_mask],['① ROI','② 색상 Mask','③ 잡음 제거','④ 선택한 후보']):
            col.image(data,caption=caption,width='stretch')
        st.caption('Opening은 작은 잡음을 제거하고, Closing은 작은 틈을 메웁니다. 가장 큰 외곽선 내부를 측정하므로 내부 구멍은 면적에 포함됩니다.')
    with tabs[2]:
        st.image(color_dashboard(result),width='stretch')
        st.dataframe(pd.DataFrame([result.roi_colors,result.candidate_colors],index=['ROI 전체 (%)','후보 내부 (%)']).rename(columns={'red':'붉은색','dark':'어두운색','other':'기타'}),width='stretch')
        st.caption('색상 그룹은 서로 겹치지 않습니다. 색은 조직 종류·감염 여부를 판정하지 않습니다. Hue 평균은 OpenCV 값의 산술평균입니다.')
    with tabs[3]:
        c=st.columns(3)
        c[0].download_button('결과 이미지 PNG',png_bytes(result.overlay),'wound_result.png','image/png',width='stretch')
        c[1].download_button('색상 대시보드 PNG',color_dashboard(result),'color_dashboard.png','image/png',width='stretch')
        c[2].download_button('분석 보고서 HTML',html_report(result,name),'wound_report.html','text/html',width='stretch')
        if st.button('현재 결과를 기록에 추가',disabled=not m['detected']):
            if any(r['analysis_id']==signature for r in st.session_state.records):
                st.info('동일한 사진과 설정의 결과가 이미 기록되어 있습니다.')
            else:
                st.session_state.records.append(dict(analysis_id=signature,time=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds'),image=name,**m,
                    roi_red_pct=result.roi_colors['red'],roi_dark_pct=result.roi_colors['dark']))
                st.success('기록에 추가했습니다.')
    with tabs[4]:
        st.json(m)
        st.write('모양 지표(원형도) = 4π × 면적 / 둘레². 분할 정확도나 임상적 신뢰도를 뜻하지 않습니다.')
        st.caption(f'입력 원본: {original.shape[1]} × {original.shape[0]}px / 분석 영상: {rgb.shape[1]} × {rgb.shape[0]}px')
else:
    st.info('사진과 분석 범위를 확인한 뒤 분석 버튼을 눌러 주세요.')

with st.expander('분석 기록 및 변화 그래프',expanded=bool(st.session_state.records)):
    records=st.session_state.records
    st.caption('기록은 현재 세션에만 유지됩니다. 종료하기 전에 CSV를 다운로드하세요. 서로 다른 사진의 px²는 촬영 조건에 영향을 받습니다.')
    if records:
        frame=pd.DataFrame(records)
        st.dataframe(frame.drop(columns=['analysis_id']),width='stretch')
        # Escape spreadsheet formula prefixes in uploaded filenames.
        safe=frame.copy()
        safe['image']=safe['image'].map(lambda x: "'"+x if x.startswith(('=','+','-','@','\t','\r')) else x)
        st.download_button('기록 CSV 다운로드',safe.to_csv(index=False).encode('utf-8-sig'),'wound_records.csv','text/csv')
        if len(records)>1:
            chart=trend_png(records)
            st.image(chart,width='stretch')
            st.download_button('면적 변화 그래프 PNG',chart,'area_trend.png','image/png')
            st.line_chart(frame[['roi_red_pct','roi_dark_pct']])
        if st.button('기록 전체 지우기'):
            st.session_state.records=[]; st.rerun()
    else:
        st.write('아직 추가한 기록이 없습니다.')
with st.expander('프로그램 원리와 사용 안내'):
    st.markdown('''1. 사진을 업로드하거나 합성 시연 이미지를 선택합니다.
2. 배경이 포함되면 가로·세로 슬라이더로 ROI를 좁힙니다.
3. 색상 민감도와 최소 면적을 조정하고 분석합니다.
4. 실제 사진과 후보 경계를 비교한 뒤 결과와 기록을 저장합니다.

PDF의 PyQt 프로그램을 Streamlit으로 재구현했습니다. 원본 코드 복구본은 아니며, 임계값·합성 이미지·UI는 새로 작성했습니다.
물리적 크기 보정은 포함하지 않습니다. 신뢰도 확률 대신 경계 접촉·다중 후보 상태와 모양 지표를 표시합니다.''')

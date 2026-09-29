from streamlit.testing.v1 import AppTest


def test_demo_comparison_and_state_reset():
    app = AppTest.from_file('app.py', default_timeout=30).run()
    app.button[0].click().run()
    assert not app.exception
    assert float(app.metric[0].value.split()[0].replace(',', '')) < 30000
    app.toggle[0].set_value(True).run()
    assert not app.metric  # Previous result is invalid after changing inputs.
    app.button[0].click().run()
    assert not app.exception
    assert any(m.label == '이전 대비 후보 면적 변화' and m.value.startswith('-') for m in app.metric)
    next(b for b in app.button if b.label == '현재 결과를 기록에 추가').click().run()
    assert len(app.session_state['records']) == 1
    app.sidebar.selectbox[0].select('작은 후보').run()
    app.button[0].click().run()
    next(b for b in app.button if b.label == '현재 결과를 기록에 추가').click().run()
    assert len(app.session_state['records']) == 2
    assert not app.exception

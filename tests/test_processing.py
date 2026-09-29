import numpy as np
import pytest
from wound_processing import analyze_wound, compare_results, sample_image, load_image, resize_image
from exports import png_bytes, html_report


def test_rectangle_geometry():
    image=np.zeros((100,100,3),np.uint8)
    image[20:61,30:81]=(200,20,20)
    result=analyze_wound(image,min_area=1,kernel_size=1)
    assert result.metrics['area_px2']==2000
    assert result.metrics['perimeter_px']==180
    assert sum(result.roi_colors.values())==pytest.approx(100)


def test_roi_excludes_larger_distractor():
    image=sample_image('배경 오검출')
    whole=analyze_wound(image)
    roi=analyze_wound(image,roi=(250,100,600,430))
    assert whole.metrics['area_px2'] > roi.metrics['area_px2']*3
    assert roi.metrics['detected']
    assert np.array_equal(roi.overlay[:100],image[:100])


def test_empty_is_not_recovery():
    blank=analyze_wound(np.zeros((100,100,3),np.uint8))
    assert not blank.metrics['detected']
    assert compare_results(analyze_wound(sample_image()),blank) is None


def test_comparison_and_resize():
    assert compare_results(analyze_wound(sample_image('이전')),analyze_wound(sample_image())) < 0
    assert resize_image(np.zeros((2000,1000,3),np.uint8)).shape==(1200,600,3)


def test_invalid_roi_and_decode():
    with pytest.raises(ValueError): analyze_wound(sample_image(),roi=(20,20,20,30))
    with pytest.raises(Exception): load_image(b'not an image')
    assert load_image(png_bytes(sample_image())).shape==(540,800,3)


def test_report_escapes_filename():
    report=html_report(analyze_wound(sample_image()),'<script>evil</script>')
    assert '<script>' not in report
    assert '&lt;script&gt;' in report

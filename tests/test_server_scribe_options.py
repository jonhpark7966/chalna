import pytest
from fastapi import HTTPException

from chalna.server import _make_scribe_options


def test_make_scribe_options_from_api_form_values():
    options = _make_scribe_options(
        diarize=False,
        tag_audio_events=True,
        num_speakers=3,
    )

    assert options.diarize is False
    assert options.tag_audio_events is True
    assert options.num_speakers == 3


def test_make_scribe_options_rejects_invalid_num_speakers():
    with pytest.raises(HTTPException) as exc_info:
        _make_scribe_options(
            diarize=True,
            tag_audio_events=True,
            num_speakers=33,
        )

    assert exc_info.value.status_code == 400

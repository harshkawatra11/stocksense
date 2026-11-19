from __future__ import annotations

import pytest

from stocksense.core.gpu import gpu_lock


def test_second_holder_gets_gpu_busy_and_release_frees_it(tmp_data_root):
    with gpu_lock(tmp_data_root), pytest.raises(RuntimeError, match="gpu busy"), gpu_lock(tmp_data_root):
        pass
    with gpu_lock(tmp_data_root):  # free again
        pass


def test_lock_is_released_when_the_body_raises(tmp_data_root):
    with pytest.raises(ValueError), gpu_lock(tmp_data_root):
        raise ValueError("boom")
    with gpu_lock(tmp_data_root):
        pass


def test_lock_file_lives_under_state(tmp_data_root):
    with gpu_lock(tmp_data_root):
        pass
    assert (tmp_data_root.paths.data_root / "state" / "gpu.lock").is_file()

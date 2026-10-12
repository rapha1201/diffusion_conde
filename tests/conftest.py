import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.make_fake_h5 import make_fake_h5  # noqa: E402


@pytest.fixture(scope="session")
def fake_h5(tmp_path_factory):
    """Small file with the fault_aware.h5 layout: 7 models x 40 sections, chunks of 8."""
    path = tmp_path_factory.mktemp("data") / "fake.h5"
    return str(make_fake_h5(str(path), n_slices=40, chunk=8, seed=0, empty_prob=0.25))

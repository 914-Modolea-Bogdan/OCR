import os
from pathlib import Path

import pytest

from server import app as flask_app


@pytest.fixture
def app():
    flask_app.config.update({"TESTING": True})
    yield flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def sample_image_path() -> Path:
    """
    Path to a sample CNAS certificate for integration tests.

    You can either:
      * set the SAMPLE_CERT_PATH env var to point to a file, OR
      * place a file at tests/data/sample_certificate.png
    """
    env = os.getenv("SAMPLE_CERT_PATH")
    if env:
        p = Path(env)
        if not p.is_file():
            pytest.skip(f"SAMPLE_CERT_PATH={p} does not exist, skipping integration test.")
        return p

    default_path = Path(__file__).parent / "data" / "sample_certificate.png"
    if not default_path.is_file():
        pytest.skip(
            "tests/data/sample_certificate.png missing; "
            "set SAMPLE_CERT_PATH or add a sample image."
        )
    return default_path

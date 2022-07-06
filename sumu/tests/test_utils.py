import io
from contextlib import redirect_stdout

import sumu


def test_cite():
    with io.StringIO() as buf, redirect_stdout(buf):
        sumu.utils.utils.cite(sumu.gadget)
        assert "NeurIPS 2020" in buf.getvalue()

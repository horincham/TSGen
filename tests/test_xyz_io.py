from pathlib import Path
import numpy as np
from tsgen.data.xyz import Frame, write_xyz_multiframe, read_xyz_multiframe

def test_xyz_roundtrip(tmp_path: Path):
    f = Frame(symbols=["H","H"], coords=np.array([[0,0,0],[0,0,1.0]]), comment="id=a")
    p = tmp_path/"a.xyz"
    write_xyz_multiframe(p, [f])
    fs = read_xyz_multiframe(p)
    assert fs[0].get_id() == "a"
    assert fs[0].coords.shape == (2,3)

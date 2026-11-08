import numpy as np
from tsgen.data.xyz import Frame
from tsgen.filters.physics import PhysicsFilter

def test_filter_min_distance():
    fr = Frame(["H","H"], np.array([[0,0,0],[0,0,0.1]]), "id=x")
    pf = PhysicsFilter(min_distance_angstrom=0.75, angle_limits_deg=(45,170))
    kept, flags = pf.apply([fr])
    assert len(kept) == 0
    assert flags[0]["keep"] is False

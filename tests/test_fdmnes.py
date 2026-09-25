from pathlib import Path

import yaml

from larixite import get_amcsd
from larixite.utils import get_logger
from larixite.struct import get_structure
from larixite.fdmnes import FdmnesXasInput

logger = get_logger("larixite.test")


def test_fdmnes():
    db = get_amcsd()
    cifids = {
        4438: ("S", "Fe"),
        4820: ("Ti", "Fe"),
        143: ("Fe", "O"),
        2400: ("Fe", "O"),
        2762: ("Fe", "O"),
    }

    for cifid, atoms in cifids.items():
        cif = db.get_cif(cifid)
        outfile = cif.write_cif(verbose=True)
        for abs in atoms:
            logger.info(f"[{cif.label}] {abs}")
            sg = get_structure(outfile, abs)
            f = FdmnesXasInput(sg, absorber=abs)
            text = f.get_input()
            assert len(text) > 700  # TODO: find a better test
            #: test the inputs writes correctly to disk into a temporary directory
            outdir = f.write_input()
            assert outdir.exists()


def test_green_scf_sync():
    db = get_amcsd()
    cif = db.get_cif(4438)
    outfile = cif.write_cif(verbose=True)
    sg = get_structure(outfile, "Fe")

    f = FdmnesXasInput(sg, absorber="Fe", green=False, scf=True, optimize=False)
    assert f.green is False
    assert f.scf is True
    assert f.params["Green"] is False
    assert f.params["SCF"] is True

    f.green = True
    f.scf = False
    assert f.params["Green"] is True
    assert f.params["SCF"] is False


CIF_ZNO = Path(__file__).parent / "structs" / "ZnO_mp-2133.cif"


def _input_body(fdm):
    """FDMNES input text without the header lines (timestamp and versions)"""
    return fdm.get_input().split("\n")[3:]


def test_from_yaml_roundtrip_defaults(tmp_path):
    f = FdmnesXasInput(CIF_ZNO, absorber="Zn", outdir=tmp_path)
    yamlpath = f.dump_params(tmp_path / "job_params.yaml")
    params = yaml.safe_load(yamlpath.read_text())
    for key in ("vmax", "ecut", "gamma_hole", "gaussian", "estart"):
        assert params[key] is None, key
    g = FdmnesXasInput.from_yaml(yamlpath)
    assert g.params["Vmax"] is False
    assert _input_body(g) == _input_body(f)


def test_from_yaml_roundtrip_custom(tmp_path):
    f = FdmnesXasInput(
        CIF_ZNO,
        absorber="Zn",
        outdir=tmp_path,
        green=False,
        scf=True,
        vmax=-6.0,
        ecut=-2.5,
        gamma_hole=1.2,
        gaussian=0.8,
        estart=-15.0,
    )
    g = FdmnesXasInput.from_yaml(f.dump_params(tmp_path / "job_params.yaml"))
    assert g.green is False and g.scf is True
    assert _input_body(g) == _input_body(f)


def test_from_yaml_legacy_placeholder(tmp_path):
    f = FdmnesXasInput(CIF_ZNO, absorber="Zn", outdir=tmp_path)
    yamlpath = f.dump_params(tmp_path / "job_params.yaml")
    params = yaml.safe_load(yamlpath.read_text())
    params["vmax"] = "!"  #: as written by older larixite versions
    yamlpath.write_text(yaml.dump(params))
    g = FdmnesXasInput.from_yaml(yamlpath)
    assert g.params["Vmax"] is False
    assert _input_body(g) == _input_body(f)


if __name__ == "__main__":
    test_fdmnes()
    test_green_scf_sync()

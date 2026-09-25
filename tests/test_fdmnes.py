import hashlib
import subprocess
from pathlib import Path
from unittest import mock

import yaml

from larixite import get_amcsd, __version__ as larixite_version
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


PROVENANCE_KEYS = {
    "schema": int,
    "created": str,
    "structure_file": str,
    "structure_path": str,
    "structure_sha256": str,
    "code": str,
    "code_version": type(None),
    "larixite_version": str,
}


def _write_zno_job(tmp_path, structpath=CIF_ZNO):
    f = FdmnesXasInput(structpath, absorber="Zn", outdir=tmp_path)
    return f, f.write_input()


def test_provenance_written(tmp_path):
    _, jobdir = _write_zno_job(tmp_path)
    provpath = jobdir / "provenance.yaml"
    assert provpath.exists()
    prov = yaml.safe_load(provpath.read_text())
    assert set(prov) == set(PROVENANCE_KEYS)
    for key, typ in PROVENANCE_KEYS.items():
        assert isinstance(prov[key], typ), key
    assert prov["schema"] == 1
    assert prov["code"] == "fdmnes"
    assert prov["larixite_version"] == larixite_version
    assert prov["structure_file"] == CIF_ZNO.name
    assert prov["structure_path"] == str(CIF_ZNO.resolve())
    assert prov["structure_sha256"] == hashlib.sha256(CIF_ZNO.read_bytes()).hexdigest()


def test_provenance_from_xasstructure(tmp_path):
    xs = get_structure(CIF_ZNO, "Zn")
    _, jobdir = _write_zno_job(tmp_path, structpath=xs)
    prov = yaml.safe_load((jobdir / "provenance.yaml").read_text())
    assert prov["structure_sha256"] == hashlib.sha256(CIF_ZNO.read_bytes()).hexdigest()


def test_provenance_no_file(tmp_path):
    f, _ = _write_zno_job(tmp_path)
    f._xs.filepath = tmp_path / "missing.cif"
    prov = yaml.safe_load(f.dump_provenance(tmp_path).read_text())
    assert prov["structure_file"] == "missing.cif"
    assert prov["structure_sha256"] is None


def test_provenance_unchanged_by_sbatch(tmp_path):
    f, jobdir = _write_zno_job(tmp_path)
    provpath = jobdir / "provenance.yaml"
    before = provpath.read_bytes()
    f.write_sbatch(jobdir)
    f.dump_params()
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="12345\n")
    with mock.patch("larixite.fdmnes.input.subprocess.run", return_value=completed):
        assert f.run_sbatch(jobdir) == "12345"
    assert yaml.safe_load((jobdir / "status.yaml").read_text())["slurm_job_id"] == "12345"
    assert provpath.read_bytes() == before


def test_params_without_provenance(tmp_path):
    f, jobdir = _write_zno_job(tmp_path)
    params = yaml.safe_load((jobdir / f"{f.fileout_prefix}_params.yaml").read_text())
    assert not set(params) & (set(PROVENANCE_KEYS) - {"structure_path"})
    FdmnesXasInput.from_yaml(jobdir / f"{f.fileout_prefix}_params.yaml")


if __name__ == "__main__":
    test_fdmnes()
    test_green_scf_sync()

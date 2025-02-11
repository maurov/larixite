#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generating FDMNES input files
==============================

[FDMNES](http://fdmnes.neel.cnrs.fr/) is a program for calculating X-ray
spectroscopy (XAS, XES, RIXS) from the atomic structures

"""

from dataclasses import dataclass
from typing import Union, Literal
from pathlib import Path
from pymatgen.core import __version__ as pymatgen_version, Element
from larixite.struct import get_structure
from larixite.struct.xas import XasStructure
from larixite.utils import get_logger, strict_ascii, isotime
from larixite.version import __version__ as larixite_version

logger = get_logger("larixite.fdmnes")

TEMPLATE_FOLDER = Path(Path(__file__).parent, "templates")
LiteralStructTypes = Literal[("crystal",)]  # TODO "molecule", "cif_file"]

FDMNES_DEFAULT_PARAMS = {
    "Energpho": False,
    "Quadrupole": False,
    "Density": False,
    "Density_all": False,
    "SCF": False,
    "Green": True,
    "Memory_save": True,
    "Relativism": False,
    "Spinorbit": None,
    "SCFexc": False,
    "SCFexcv": False,
    "Screening": False,
    "Full_atom": False,
    "TDDFT": False,
    "PBE96": False,
    "Atom_conf": False,  #: preferred over `Atom` (permits to keep atomic number in the list of atoms) -> TODO
    "COOP": False,
    "Convolution": True,
}


@dataclass
class FdmnesXasInput:
    """Input generator for a XAS calculation with FDMNES"""

    structpath: Union[
        str, Path, XasStructure
    ]  #: path to the structural file or XasStructure
    absorber: Union[
        str, int, Element
    ]  #: atomic symbol or number of the absorbing element
    absorber_idx: Union[
        int, None
    ] = None  #: index of the absorbing atom in the pymatgen structure
    frame: int = 0  #: index of the frame inside the structure
    edge: Union[str, None] = None  #: edge for calculation
    radius: float = 7  #: radius of the calulation
    struct_type: Union[str, None] = None  #: type of the structure
    vmax: Union[float, None] = None  #: maximum potential value for molecules
    erange: Union[str, None] = "-20.0 0.1 70.0 1.0 100.0"  #: energy range
    tmplpath: Union[str, Path, None] = None  #: path to the FDMNES input template
    params: Union[dict, None] = None  #: parameters for FDMNES

    def __post_init__(self):
        """Validate and adjust attributes"""
        #: absorber
        if isinstance(self.absorber, str):
            self.absorber = Element(self.absorber)
        elif isinstance(self.absorber, int):
            self.absorber = Element.from_Z(self.absorber)
        #: init structure object and type
        if isinstance(self.structpath, str):
            self.structpath = Path(self.structpath)
        if isinstance(self.structpath, XasStructure):
            self.xs = self.structpath
            self.structpath = self.xs.filepath
        else:
            self.xs = get_structure(self.structpath, absorber=self.absorber)
        if self.struct_type is None:
            self.struct_type = self.xs.struct_type
        #: template
        if self.tmplpath is None:
            self.tmplpath = Path(TEMPLATE_FOLDER, "fdmnes_xas.tmpl")
        if isinstance(self.tmplpath, str):
            self.tmplpath = Path(self.tmplpath)
        #: absorption edge
        self.validate_edge()
        #: optimize params
        if self.params is None:
            self.params = FDMNES_DEFAULT_PARAMS
            self.params = self.optimize_params()

    def validate_edge(self):
        """Validates and adjusts the edge attribute"""

        valid_edges = [
            "K",
            "L1",
            "L2",
            "L3",
            "L23",
            "M1",
            "M2",
            "M3",
            "M23",
            "M4",
            "M5",
            "M45",
            "N1",
            "N2",
            "N3",
            "N23",
            "N4",
            "N5",
            "N45",
        ]
        if self.edge is None:
            self.edge = "K" if self.absorber.Z < 58 else "L"
        if self.edge == "L":
            self.edge = "L23"
            logger.warning("Edge 'L' changed to 'L23'")
        if self.edge == "M":
            self.edge = "M45"
            logger.warning("Edge 'M' changed to 'M45'")
        if self.edge not in valid_edges:
            bad_edge = self.edge
            self.edge = "K" if self.absorber.Z < 58 else "L"
            logger.error(f"Edge {bad_edge} not valid -> changed to {self.edge}")

    def optimize_params(self) -> dict:
        """Optimize the given input parameters"""
        params = self.params.copy()
        atoms_z = [species.Z for species in self.xs.struct.types_of_species]
        abs_z = self.absorber.Z
        transition_metals = [range(21, 31), range(39, 49), range(57, 81)]

        if any(abs_z in r for r in transition_metals):
            params["Quadrupole"] = True

        if self.edge == "L23" and abs_z in range(21, 26):
            params["TDDFT"] = True

        if any(z > 36 for z in atoms_z):
            params["Relativism"] = True

        if any(z > 50 for z in atoms_z):
            params["Spinorbit"] = True
            logger.info(
                "Spinorbit enabled. **NOTE**: the simulations are typically 4 to 8 times longer and need 2 times more memory space"
            )

        if 8 in atoms_z:
            params["Full_atom"] = True

        if "mol" in self.struct_type.lower():
            self.vmax = -6

        return params

    def get_structure(self, struct_type: Union[str, None] = None) -> str:
        """Get the structure section of the input

        Parameters
        ------------

        struct_type: None | str [None -> self.struct_type]
            type of the structure -> see Notes

        Notes
        -----

        FDMNES supports various structure types:
            - Crystal  -> Implemented (default)
            - Molecule -> TODO
            - Film  -> Not implemented yet
            - Surface  -> Not implemented yet
            - Interface  -> Not implemented yet
            - Pdb_file  -> Not implemented yet
            - Film_Pdb_file  -> Not implemented yet
            - Cif_file  -> TODO
            - Film_Cif_file  -> Not implemented yet

        """
        if struct_type is not None:
            self.struct_type = struct_type
            self.optimize_params()
        else:
            struct_type = self.struct_type
        logger.debug(f"Generating structure section for {struct_type}")
        structout = [f"!<structure description start: {struct_type}>"]
        if "crys" in struct_type.lower():
            structout.append("Spgroup")
            structout.append(f"   {self.xs.space_group}")
            structout.append("Occupancy")
            structout.append("Crystal")
            lattice = self.xs.sym_struct.lattice
            structout.append(
                f"   {lattice.a} {lattice.b} {lattice.c} {lattice.alpha} {lattice.beta} {lattice.gamma}"
            )
            for (
                idx,
                site,
                site_index,
                occupancy,
                len_sites,
                wyckoff,
            ) in self.xs.unique_sites:
                zelems = [elem.Z for elem in site.species.elements]
                if not len(set(zelems)) == 1:
                    logger.warning(
                        f"[{self.xs.label}] site {idx} has species with different Z -> {site.species_string}"
                    )
                for elem, elstr in zip(
                    site.species.elements, site.species_string.split(", ")
                ):
                    sitestr = f"{elem.Z:>3d} {site.a:15.10f} {site.b:15.10f} {site.c:15.10f} {occupancy:>5.2f} !{site.label:>4s} {wyckoff:>4s} {elstr:>4s}"
                    structout.append(sitestr)
        elif "mol" in struct_type.lower():
            structout.append("Molecule")
            structout.append("   1.0 1.0 1.0 90.0 90.0 90.0")
            for (
                idx,
                site,
                site_index,
                occupancy,
                len_sites,
                wyckoff,
            ) in self.xs.unique_sites:
                zelems = [elem.Z for elem in site.species.elements]
                if not len(set(zelems)) == 1:
                    logger.warning(
                        f"[{self.xs.label}] site {idx} has species with different Z -> {site.species_string}"
                    )
                for elem, elstr in zip(
                    site.species.elements, site.species_string.split(", ")
                ):
                    sitestr = f"{elem.Z:>3d} {site.a:15.10f} {site.b:15.10f} {site.c:15.10f} {occupancy:>5.2f} !{site.label:>4s} {wyckoff:>4s} {elstr:>4s}"
                    structout.append(sitestr)
        else:
            errmsg = f"Structure type `{struct_type}` not supported"
            logger.error(errmsg)
            raise AttributeError(errmsg)
        structout.append("!</structure description end>")
        return "\n".join(structout)

    def get_atbsorber(self) -> str:
        """Get the absorber section of the input

        TODO:
        - add the possibility to use the `Absorber` (number of atom/s in the list of atoms) card instead of `Z_absorber`
        """
        absout = ["Z_absorber"]
        absout.append(f"   {self.absorber.Z}")
        return "\n".join(absout)

    def get_vmax(self) -> str:
        """Get the vmax section of the input"""
        if self.vmax is not None:
            vmax = ["Vmax"]
            vmax.append("   -6")
            return "\n".join(vmax)
        else:
            return "! Vmax"

    def get_input(self, comment: str = "", struct_type: str = None) -> str:
        params = self.params.copy()
        template = open(self.tmplpath, "r").read()

        comment = (
            f"   {self.xs.name}: {self.absorber.symbol} ({self.absorber.Z}) {self.edge} edge"
            + comment
        )
        #: fill the template
        vers = larixite_version[:]
        if ".post" in vers:
            vers = vers.split(".post")[0]
        conf = {
            "timestamp": isotime(),
            "version": vers,
            "pymatgen_version": pymatgen_version,
            "comment": comment,
            "edge": self.edge,
            "radius": f"{self.radius:.2f}",
            "erange": self.erange,
            "vmax": self.get_vmax(),
            "absorber": self.get_atbsorber(),
            "structure": self.get_structure(struct_type=struct_type),
        }
        for parkey, parval in params.items():
            conf[parkey] = str(parkey) if parval is True else f"! {parkey}"

        return strict_ascii(template.format(**conf))

    def write_input(
        self, inputtext: Union[str, None] = None, outdir: Union[str, Path, None] = None
    ) -> None:
        """Write the FDMNES input text to disk."""
        if inputtext is None:
            inputtext = self.get_input()
        if outdir is None:
            import tempfile

            outdir = (
                Path(tempfile.gettempdir()) / "larixite" / "fdmnes" / str(self.xs.name)
            )
            outdir = tempfile.mkdtemp(dir=outdir, prefix="job_")
        if isinstance(outdir, str):
            outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fnout = outdir / "job_inp.txt"
        with open(fnout, "w") as fp:
            fp.write(inputtext)
        with open(outdir / "fdmfile.txt", "w") as fp:
            fp.write("1\njob_inp.txt")
        logger.info(f"written `{fnout}`")
        return

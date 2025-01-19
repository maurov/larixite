#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Wrapper on top of pymatgen to handle atomic structures for XAS calculations
============================================================================
"""

import numpy as np
from pathlib import Path
from typing import Union
from pymatgen.io.xyz import XYZ
from pymatgen.io.cif import CifParser
from pymatgen.core import Molecule, Structure, Element, Lattice, Site
from larixite.struct.xas import XasStructure
from larixite.struct.xas_cif import XasStructureCif
from larixite.struct.xas_xyz import XasStructureXyz
from larixite.utils import get_logger, fcompact
from larixite.amcsd_utils import PMG_CIF_OPTS

logger = get_logger("larixite.struct")

if logger.level != 10:
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")


def site_label(site: Site) -> str:
    """
    return a string label for a pymatgen Site object,
    using the species string and fractional coordinates

    Parameters
    ----------
    site : pymatgen Site object

    Returns
    -------
    str
    """
    coords = ",".join([fcompact(s) for s in site.frac_coords])
    return f"{site.species_string}[{coords}]"


def mol2struct(molecule: Molecule) -> Structure:
    """Convert a pymatgen Molecule to Structure"""
    # extend the lattice
    alat, blat, clat = np.max(molecule.cart_coords, axis=0)
    lattice = Lattice.from_parameters(
        a=alat, b=blat, c=clat, alpha=90, beta=90, gamma=90
    )
    # Create a list of species
    species = [Element(sym) for sym in molecule.species]
    # Create a list of coordinates
    coords = molecule.cart_coords
    # Create the Structure object
    struct = Structure(lattice, species, coords, coords_are_cartesian=True)
    return struct


def get_structure(
    filepath: Union[str, Path], absorber: str, frame: int = 0
) -> XasStructure:
    """
    Get a XasStructure from a structural file.

    Parameters
    ----------
    filepath : str or Path
        Filepath to CIF/XYZ file.
    absorber : str
        Atomic symbol of the absorbing element.
    frame : int, optional
        Index of the structure in the CIF/XYZ file.

    Returns
    -------
    XasStructure
        The XAS structure group for the specified file and absorber.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    if not filepath.exists():
        errmsg = f"{filepath} not found"
        logger.error(errmsg)
        raise FileNotFoundError(errmsg)
    #: CIF
    if filepath.suffix == ".cif":
        try:
            structs = CifParser(filepath, **PMG_CIF_OPTS)
        except Exception:
            raise ValueError(f"could not parse text of CIF from {filepath}")
        try:
            struct = structs.parse_structures()[frame]
        except Exception:
            raise ValueError(f"could not get structure {frame} from text of CIF")
        mol = Molecule.from_dict(struct.as_dict())
        file_format = "cif"
        logger.debug("structure created from a CIF file")
        return XasStructureCif(
            name=filepath.name,
            label=filepath.stem,
            filepath=filepath,
            file_format=file_format,
            struct=struct,
            mol=mol,
            absorber=Element(absorber),
            absorber_idx=None,
        )
    #: XYZ
    if filepath.suffix == ".xyz":
        xyz = XYZ.from_file(filepath)
        molecules = xyz.all_molecules
        mol = molecules[frame]
        struct = mol2struct(mol)
        file_format = "xyz"
        logger.debug("structure created from a XYZ file")
        return XasStructureXyz(
            name=filepath.name,
            label=filepath.stem,
            filepath=filepath,
            file_format=file_format,
            struct=struct,
            mol=mol,
            absorber=Element(absorber),
            absorber_idx=None,
        )

    #: UNSUPPORTED
    raise ValueError(f"File type {filepath.suffix} not supported yet")


def get_structs_from_dir(
    structsdir: Union[str, Path],
    absorbers: Union[list[str], str],
    globstr: str = "*",
    exclude_names: list[str] = None,
    **kwargs,
) -> list[XasStructure]:
    """Get a list of XasStructure from a directory containing structural files

    Parameters
    ----------
    structsdir : str or Path
        directory containing the structural files
    absorbers : list of str or str
        list of atomic symbols of the absorbing elements for each file or a single symbol for all
    globstr : str, optional
        string to filter the files in the directory
    exclude_names : list of str, optional
        list of filenames to exclude
    **kwargs : dict, optional
        additional keyword arguments to pass to `get_structure`

    Returns
    -------
    list of XasStructure
        list of XasStructure objects

    Examples
    --------

    from pathlib import Path
    curdir = Path().cwd()
    basedir = curdir.parent
    testdir = basedir / "test"
    structsdir = testdir / "structs"
    abs = "Fe"
    structs = get_structs_from_dir(structsdir, abs, globstr=f"*{abs}*", exclude_names=["NAMING.tmpl"])

    """
    if isinstance(structsdir, str):
        structsdir = Path(structsdir)
    structs_paths = list(structsdir.glob(globstr))
    if exclude_names is not None:
        structs_paths = [
            struct for struct in structs_paths if struct.name not in exclude_names
        ]
    if isinstance(absorbers, str):
        absorbers = [absorbers] * len(structs_paths)
    assert (
        len(structs_paths) == len(absorbers)
    ), f"number of structures ({len(structs_paths)}) != number of absorbers ({len(absorbers)})"
    structs = []
    for istruct, struct_path in enumerate(structs_paths):
        struct = get_structure(struct_path, absorbers[istruct], **kwargs)
        logger.info(f"{istruct}: {struct.name}")
        structs.append(struct)
    return structs


def build_cluster(
    xsg: XasStructure,
    absorber_site=None,
    radius=None,
):
    if absorber_site not in xsg.atom_sites[xsg.absorber]:
        raise ValueError(
            f"invalid site for absorber {absorber}: must be in {xsg.atom_sites[xsg.absorber]}"
        )
    if radius is not None:
        xsg.radius = radius
    cluster_size = xsg.cluster_size
    csize2 = cluster_size**2
    site_atoms = {}  # map xtal site with list of atoms occupying that site
    site_tags = {}

    for i, site in enumerate(xsg.struct.sites):
        label = site_label(site)
        s_unique = xsg.unique_map.get(label, 0)
        site_species = [e.symbol for e in site.species]
        if len(site_species) > 1:
            s_els = [s.symbol for s in site.species.keys()]

            s_wts = [s for s in site.species.values()]
            site_atoms[i] = rng.choices(s_els, weights=s_wts, k=1000)
            site_tags[i] = f"({site.species_string:s})_{s_unique:d}"
        else:
            site_atoms[i] = [site_species[0]] * 1000
            site_tags[i] = f"{site.species_string:s}_{s_unique:d}"

    # atom0 = xsg.struct[a_index]
    atom0 = xsg.unique_sites[absorber_site - 1][0]
    sphere = xsg.struct.get_neighbors(atom0, xsg.cluster_size)

    xsg.symbols = [xsg.absorber]
    xsg.coords = [[0, 0, 0]]
    site0_species = [e.symbol for e in atom0.species]
    if len(site0_species) > 1:
        xsg.tags = [f"({atom0.species_string})_{absorber_site:d}"]
    else:
        xsg.tags = [f"{atom0.species_string}_{absorber_site:d}"]

    for i, site_dist in enumerate(sphere):
        s_index = site_dist[0].index
        site_symbol = site_atoms[s_index].pop()

        coords = site_dist[0].coords - atom0.coords
        if (coords[0] ** 2 + coords[1] ** 2 + coords[2] ** 2) < csize2:
            xsg.tags.append(site_tags[s_index])
            xsg.symbols.append(site_symbol)
            xsg.coords.append(coords)

    xsg.molecule = Molecule(xsg.symbols, xsg.coords)

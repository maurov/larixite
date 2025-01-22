#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Specialize XasStructure to handle structures from XYZ files
=============================================================
"""

from dataclasses import dataclass
from larixite.struct.xas import XasStructure
from larixite.utils import get_logger

logger = get_logger("larixite.struct")


@dataclass
class XasStructureXyz(XasStructure):

    @property
    def sga(self):
        raise AttributeError("SpacegroupAnalyzer fails for XYZ files")

    @property
    def space_group(self):
        return "P1"

    @property
    def sym_struct(self):
        """No symmetrized structure for XYZ, simply return the structure"""
        return self.struct

    @property
    def equivalent_sites(self):
        return [[site] for site in self.struct.sites]

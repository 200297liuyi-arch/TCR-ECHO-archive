"""Atom-level graph featurizer for peptide sequences.

Stripped-down version of deepAntigen's MolGraphConvFeaturizer,
with deepchem dependency removed. Converts RDKit Mol objects
into atom-level graphs with node (atom) and edge (bond) features.
"""

import numpy as np
import os
import logging
from typing import List, Union, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ── type stubs (replacing deepchem.typing) ──────────────────────────
RDKitAtom = object
RDKitBond = object
RDKitMol = object


@dataclass
class GraphData:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray


# ── allowable sets ──────────────────────────────────────────────────
DEFAULT_ATOM_TYPE_SET = ["C", "O", "N", "S"]
DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3"]
DEFAULT_TOTAL_NUM_Hs_SET = [0, 1, 2, 3, 4]
DEFAULT_TOTAL_DEGREE_SET = [0, 1, 2, 3, 4, 5]
DEFAULT_BOND_TYPE_SET = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
DEFAULT_BOND_STEREO_SET = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]


# ── RDKit chemical features singleton ───────────────────────────────
class _ChemicalFeaturesFactory:
    _instance = None

    @classmethod
    def get_instance(cls):
        try:
            from rdkit import RDConfig
            from rdkit.Chem import ChemicalFeatures
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        if not cls._instance:
            fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
            cls._instance = ChemicalFeatures.BuildFeatureFactory(fdefName)
        return cls._instance


# ── one-hot utility ─────────────────────────────────────────────────
def one_hot_encode(
    val: Union[int, str],
    allowable_set: Union[List[str], List[int]],
    include_unknown_set: bool = False,
) -> List[float]:
    if include_unknown_set is False and val not in allowable_set:
        logger.info(
            "input {0} not in allowable set {1}:".format(val, allowable_set)
        )

    length = len(allowable_set) + 1 if include_unknown_set else len(allowable_set)
    one_hot = [0.0 for _ in range(length)]
    try:
        one_hot[allowable_set.index(val)] = 1.0
    except ValueError:
        if include_unknown_set:
            one_hot[-1] = 1.0
    return one_hot


# ══════════════════════════════════════════════════════════════════════
#  atom (node) featurization
# ══════════════════════════════════════════════════════════════════════

def get_atom_type_one_hot(
    atom,
    allowable_set: List[str] = DEFAULT_ATOM_TYPE_SET,
    include_unknown_set: bool = True,
) -> List[float]:
    return one_hot_encode(atom.GetSymbol(), allowable_set, include_unknown_set)


def construct_hydrogen_bonding_info(mol) -> List[Tuple[int, str]]:
    factory = _ChemicalFeaturesFactory.get_instance()
    feats = factory.GetFeaturesForMol(mol)
    hydrogen_bonding = []
    for f in feats:
        hydrogen_bonding.append((f.GetAtomIds()[0], f.GetFamily()))
    return hydrogen_bonding


def get_atom_hydrogen_bonding_one_hot(
    atom, hydrogen_bonding: List[Tuple[int, str]]
) -> List[float]:
    one_hot = [0.0, 0.0]
    atom_idx = atom.GetIdx()
    for hb_tuple in hydrogen_bonding:
        if hb_tuple[0] == atom_idx:
            if hb_tuple[1] == "Donor":
                one_hot[0] = 1.0
            elif hb_tuple[1] == "Acceptor":
                one_hot[1] = 1.0
    return one_hot


def get_atom_is_in_aromatic_one_hot(atom) -> List[float]:
    return [float(atom.GetIsAromatic())]


def get_atom_hybridization_one_hot(
    atom,
    allowable_set: List[str] = DEFAULT_HYBRIDIZATION_SET,
    include_unknown_set: bool = False,
) -> List[float]:
    return one_hot_encode(
        str(atom.GetHybridization()), allowable_set, include_unknown_set
    )


def get_atom_total_num_Hs_one_hot(
    atom,
    allowable_set: List[int] = DEFAULT_TOTAL_NUM_Hs_SET,
    include_unknown_set: bool = True,
) -> List[float]:
    return one_hot_encode(
        atom.GetTotalNumHs(), allowable_set, include_unknown_set
    )


def get_atom_formal_charge(atom) -> List[float]:
    return [float(atom.GetFormalCharge())]


def get_atom_partial_charge(atom) -> List[float]:
    gasteiger_charge = atom.GetProp("_GasteigerCharge")
    if gasteiger_charge in ["-nan", "nan", "-inf", "inf"]:
        gasteiger_charge = 0.0
    return [float(gasteiger_charge)]


def get_atom_total_degree_one_hot(
    atom,
    allowable_set: List[int] = DEFAULT_TOTAL_DEGREE_SET,
    include_unknown_set: bool = True,
) -> List[float]:
    return one_hot_encode(
        atom.GetTotalDegree(), allowable_set, include_unknown_set
    )


# ══════════════════════════════════════════════════════════════════════
#  bond (edge) featurization
# ══════════════════════════════════════════════════════════════════════

def get_bond_type_one_hot(
    bond,
    allowable_set: List[str] = DEFAULT_BOND_TYPE_SET,
    include_unknown_set: bool = False,
) -> List[float]:
    return one_hot_encode(
        str(bond.GetBondType()), allowable_set, include_unknown_set
    )


def get_bond_is_in_same_ring_one_hot(bond) -> List[float]:
    return [int(bond.IsInRing())]


def get_bond_is_conjugated_one_hot(bond) -> List[float]:
    return [int(bond.GetIsConjugated())]


def get_bond_stereo_one_hot(
    bond,
    allowable_set: List[str] = DEFAULT_BOND_STEREO_SET,
    include_unknown_set: bool = True,
) -> List[float]:
    return one_hot_encode(
        str(bond.GetStereo()), allowable_set, include_unknown_set
    )


# ══════════════════════════════════════════════════════════════════════
#  feature constructors
# ══════════════════════════════════════════════════════════════════════

def _construct_atom_feature(
    atom,
    h_bond_infos: List[Tuple[int, str]],
    use_chirality: bool,
    use_partial_charge: bool,
) -> np.ndarray:
    atom_type = get_atom_type_one_hot(atom)
    formal_charge = get_atom_formal_charge(atom)
    hybridization = get_atom_hybridization_one_hot(atom)
    acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    aromatic = get_atom_is_in_aromatic_one_hot(atom)
    degree = get_atom_total_degree_one_hot(atom)
    total_num_Hs = get_atom_total_num_Hs_one_hot(atom)
    atom_feat = np.concatenate(
        [
            atom_type,
            formal_charge,
            hybridization,
            acceptor_donor,
            aromatic,
            degree,
            total_num_Hs,
        ]
    )

    if use_partial_charge:
        partial_charge = get_atom_partial_charge(atom)
        atom_feat = np.concatenate([atom_feat, partial_charge])
    return atom_feat


def _construct_bond_feature(bond) -> np.ndarray:
    bond_type = get_bond_type_one_hot(bond)
    same_ring = get_bond_is_in_same_ring_one_hot(bond)
    conjugated = get_bond_is_conjugated_one_hot(bond)
    stereo = get_bond_stereo_one_hot(bond)
    return np.concatenate([bond_type, same_ring, conjugated, stereo])


# ══════════════════════════════════════════════════════════════════════
#  MolGraphConvFeaturizer  (standalone, no deepchem parent)
# ══════════════════════════════════════════════════════════════════════

class MolGraphConvFeaturizer:
    """Featurizer that converts an RDKit Mol into an atom-level graph.

    Node features (atom):  type | formal_charge | hybridization |
                           H-bond donor/acceptor | aromatic | degree | num_Hs
    Edge features (bond):  type | same_ring | conjugated | stereo

    Parameters
    ----------
    use_edges : bool
        Whether to include edge features.
    use_chirality : bool
        Whether to include chirality (requires _CIPCode property on atoms).
    use_partial_charge : bool
        Whether to compute Gasteiger partial charges.
    """

    def __init__(
        self,
        use_edges: bool = False,
        use_chirality: bool = False,
        use_partial_charge: bool = False,
    ):
        self.use_edges = use_edges
        self.use_partial_charge = use_partial_charge
        self.use_chirality = use_chirality

    def _featurize(self, mol) -> GraphData:
        if self.use_partial_charge:
            try:
                mol.GetAtomWithIdx(0).GetProp("_GasteigerCharge")
            except KeyError:
                from rdkit.Chem import AllChem
                AllChem.ComputeGasteigerCharges(mol)

        # atom (node) features
        h_bond_infos = construct_hydrogen_bonding_info(mol)
        atom_features = np.asarray(
            [
                _construct_atom_feature(
                    atom, h_bond_infos, self.use_chirality, self.use_partial_charge
                )
                for atom in mol.GetAtoms()
            ],
            dtype=float,
        )

        # edge (bond) index   — directed, so each bond is added twice
        src, dest = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        # edge (bond) features
        bond_features = None
        if self.use_edges:
            features = []
            for bond in mol.GetBonds():
                features += 2 * [_construct_bond_feature(bond)]
            bond_features = np.asarray(features, dtype=float)

        return GraphData(
            node_features=atom_features,
            edge_index=np.asarray([src, dest], dtype=int),
            edge_features=bond_features,
        )

    def featurize(self, mol):
        """Alias for _featurize."""
        return self._featurize(mol)

    @property
    def num_node_features(self) -> int:
        """Return the dimensionality of a single node feature vector (approx 26)."""
        dummy = [0.0] * 30  # conservative upper bound; actual is ~26
        # compute precisely by running on a minimal molecule
        from rdkit import Chem
        m = Chem.MolFromSequence("G")
        g = self._featurize(m)
        return g.node_features.shape[1]

    @property
    def num_edge_features(self) -> int:
        if not self.use_edges:
            return 0
        from rdkit import Chem
        m = Chem.MolFromSequence("AA")  # has at least one bond
        g = self._featurize(m)
        return g.edge_features.shape[1]

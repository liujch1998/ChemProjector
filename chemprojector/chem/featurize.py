import re
from pathlib import Path

from rdkit import Chem


def atom_features_simple(atom: Chem.rdchem.Atom | None) -> int:
    if atom is None:
        return 0
    return min(atom.GetAtomicNum(), 100)


def bond_features_simple(bond: Chem.rdchem.Bond | None) -> int:
    if bond is None:
        return 0
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        return 1
    elif bt == Chem.rdchem.BondType.DOUBLE:
        return 2
    elif bt == Chem.rdchem.BondType.TRIPLE:
        return 3
    elif bt == Chem.rdchem.BondType.AROMATIC:
        return 4
    return 5


_smiles_vocab_path = Path(__file__).parent / "smiles_vocab.txt"
with open(_smiles_vocab_path) as f:
    _smiles_vocab = f.read().splitlines()
    _smiles_token_to_id = {token: i for i, token in enumerate(_smiles_vocab, start=1)}
    _smiles_token_max = max(_smiles_token_to_id.values())
    _smiles_token_pattern = re.compile("(" + "|".join(map(re.escape, sorted(_smiles_vocab, reverse=True))) + ")")


def tokenize_smiles(s_in: str):
    tok: list[int] = []
    for token in _smiles_token_pattern.findall(s_in):
        tok.append(_smiles_token_to_id.get(token, _smiles_token_max + 1))
    return tok


# ---- Chain-of-Reaction helpers ----

# Special tokens for CoR notation
# Use high token IDs so they do not clash with SMILES tokens
COR_START = 128000
COR_END = 128001
COR_MOL_START = 128002
COR_MOL_END = 128003

# Offset for SMILES tokens within the unified vocabulary
_COR_SMILES_OFFSET = 4

# Offset for reaction tokens within the unified vocabulary
# Reaction tokens occupy 128010 + index (1-based)
_COR_RXN_OFFSET = 128010


def cor_reaction_token(index: int) -> int:
    """Return token id for reaction index."""
    return _COR_RXN_OFFSET + index


def tokenize_cor_smiles(smi: str) -> list[int]:
    """Tokenize a SMILES string for CoR notation."""
    return [COR_MOL_START] + [t + _COR_SMILES_OFFSET for t in tokenize_smiles(smi)] + [COR_MOL_END]

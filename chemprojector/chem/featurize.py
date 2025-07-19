import re
from pathlib import Path
import os
from transformers import AutoTokenizer, logging
logging.set_verbosity_error()

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


# Special tokens for CoR notation
# These are reserved_special_token_xxx in the Llama-3 tokenizer
PAD = 128004 # <|finetune_right_pad_id|>
BOS = 128000 # <|begin_of_text|>
EOS = 128001 # <|end_of_text|>
COR_START = 128002
COR_END = 128003
COR_MOL_START = 128005
COR_MOL_END = 128011
# Reaction tokens occupy 128011 + index (1-based)
_COR_RXN_OFFSET = 128011


def reaction_token(index: int) -> int:
    """Return token id for reaction index."""
    return _COR_RXN_OFFSET + index


tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-3.1-8B',
    use_auth_token=os.environ['HF_TOKEN'],
    trust_remote_code=True,
    add_bos_token=False,
    add_eos_token=False
)


def tokenize_smiles(smi: str) -> list[int]:
    """Tokenize a SMILES string for CoR notation."""
    token_ids: list[int] = [COR_MOL_START]
    for token in _smiles_token_pattern.findall(smi):
        token_ids.extend(tokenizer.encode(token, add_special_tokens=False))
    token_ids.append(COR_MOL_END)
    return token_ids

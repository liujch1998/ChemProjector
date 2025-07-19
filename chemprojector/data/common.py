import enum
from collections.abc import Sequence
from typing import TypedDict

import numpy as np

import torch

from chemprojector.chem.fpindex import FingerprintIndex
from chemprojector.chem.mol import Molecule
from chemprojector.chem.reaction import Reaction
from chemprojector.chem.stack import Stack
from chemprojector.chem.featurize import (
    BOS,
    EOS,
    COR_END,
    COR_MOL_END,
    COR_MOL_START,
    COR_START,
    reaction_token,
    tokenize_smiles,
)
from chemprojector.utils.image import draw_text, make_grid


class TokenType(enum.IntEnum):
    END = 0
    START = 1
    REACTION = 2
    REACTANT = 3


class ProjectionData(TypedDict, total=False):
    # All
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    # Encoder
    atoms: torch.Tensor
    bonds: torch.Tensor
    atom_padding_mask: torch.Tensor
    smiles: torch.Tensor
    # Decoder
    # token_types: torch.Tensor
    # rxn_indices: torch.Tensor
    # reactant_fps: torch.Tensor
    token_ids: torch.Tensor
    token_padding_mask: torch.Tensor
    # Auxilliary
    mol_seq: Sequence[Molecule]
    rxn_seq: Sequence[Reaction | None]


class ProjectionBatch(TypedDict, total=False):
    # Encoder
    atoms: torch.Tensor
    bonds: torch.Tensor
    atom_padding_mask: torch.Tensor
    smiles: torch.Tensor
    # Decoder
    token_types: torch.Tensor
    rxn_indices: torch.Tensor
    reactant_fps: torch.Tensor
    token_ids: torch.Tensor
    token_padding_mask: torch.Tensor
    # Auxilliary
    mol_seq: Sequence[Sequence[Molecule]]
    rxn_seq: Sequence[Sequence[Reaction | None]]


def featurize_stack_actions(
    mol_seq: Sequence[Molecule],
    rxn_idx_seq: Sequence[int | None],
    end_token: bool,
) -> dict[str, torch.Tensor]:
    """Featurize actions using Chain-of-Reaction notation."""
    tokens: list[int] = [COR_START]
    for mol, rxn_idx in zip(mol_seq, rxn_idx_seq):
        if rxn_idx is None:
            tokens.extend(tokenize_smiles(mol.csmiles))
        else:
            tokens.append(reaction_token(rxn_idx))
            # If a reaction exists here, `mol` is the resulting product
            tokens.extend(tokenize_smiles(mol.csmiles))

    if end_token:
        tokens.append(COR_END)

    feats = {
        "token_ids": torch.tensor(tokens, dtype=torch.long),
        "token_padding_mask": torch.zeros(len(tokens), dtype=torch.bool),
    }
    return feats


def featurize_stack(stack: Stack, end_token: bool, fpindex: FingerprintIndex) -> dict[str, torch.Tensor]:
    return featurize_stack_actions(
        mol_idx_seq=stack.get_mol_idx_seq(),
        rxn_idx_seq=stack.get_rxn_idx_seq(),
        end_token=end_token,
        fpindex=fpindex,
    )


def create_data(
    product: Molecule,
    mol_seq: Sequence[Molecule],
    rxn_seq: Sequence[Reaction | None],
    rxn_idx_seq: Sequence[int | None],
    max_input_len: int,
    max_output_len: int,
):
    """Create training data using chain-of-reaction notation."""
    atom_f, bond_f = product.featurize_simple()
    stack_feats = featurize_stack_actions(
        mol_seq=mol_seq,
        rxn_idx_seq=rxn_idx_seq,
        end_token=True,
    )

    input_tokens = [BOS] + product.tokenize_csmiles()
    if len(input_tokens) > max_input_len: # truncate left
        input_tokens = input_tokens[-max_input_len:]
    if len(input_tokens) < max_input_len: # pad left
        input_tokens = [PAD] * (max_input_len - len(input_tokens)) + input_tokens
    input_tokens = torch.tensor(input_tokens, dtype=torch.long)

    output_tokens = stack_feats["token_ids"] + [EOS]
    if len(output_tokens) > max_output_len: # truncate right
        output_tokens = output_tokens[:max_output_len]
    if len(output_tokens) < max_output_len: # pad right
        output_tokens = output_tokens + [PAD] * (max_output_len - len(output_tokens))
    output_tokens = torch.tensor(output_tokens, dtype=torch.long)

    input_ids = torch.cat([input_tokens, output_tokens], dim=0)
    attention_mask = (input_ids != PAD).to(torch.long)
    labels = torch.cat([-100 * torch.ones_like(input_tokens), torch.where(output_tokens == PAD, -100, output_tokens)])

    data: "ProjectionData" = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "mol_seq": mol_seq,
        "rxn_seq": rxn_seq,
        "atoms": atom_f,
        "bonds": bond_f,
        "smiles": product.tokenize_csmiles(),
        "atom_padding_mask": torch.zeros([atom_f.size(0)], dtype=torch.bool),
        "token_ids": stack_feats["token_ids"],
        "token_padding_mask": stack_feats["token_padding_mask"],
    }
    return data


def draw_data(data: ProjectionData):
    im_list = [draw_text("START")]
    for m, r in zip(data["mol_seq"], data["rxn_seq"]):
        if r is not None:
            im_list.append(r.draw())
        else:
            im_list.append(m.draw())
    im_list.append(draw_text("END"))
    return make_grid(im_list)


def draw_batch(batch: ProjectionBatch):
    bsz = len(batch["mol_seq"])
    for b in range(bsz):
        im_list = [draw_text("START")]
        for m, r in zip(batch["mol_seq"][b], batch["rxn_seq"][b]):
            if r is not None:
                im_list.append(r.draw())
            else:
                im_list.append(m.draw())
        im_list.append(draw_text("END"))
        yield make_grid(im_list)

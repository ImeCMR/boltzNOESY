from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from boltz.data import const
from boltz.data.feature.featurizer import BoltzFeaturizer
from boltz.data.feature.pad import pad_to_max
from boltz.data.tokenize.boltz import BoltzTokenizer
from boltz.data.types import (
    MSA,
    Connection,
    Input,
    Manifest,
    Record,
    ResidueConstraints,
    Structure,
)


def load_input(
    record: Record,
    target_dir: Path,
    msa_dir: Path,
    constraints_dir: Optional[Path] = None,
) -> Input:
    """Load the given input data.

    Parameters
    ----------
    record : Record
        The record to load.
    target_dir : Path
        The path to the data directory.
    msa_dir : Path
        The path to msa directory.

    Returns
    -------
    Input
        The loaded input.

    """
    # Load the structure
    structure = np.load(target_dir / f"{record.id}.npz")
    structure = Structure(
        atoms=structure["atoms"],
        bonds=structure["bonds"],
        residues=structure["residues"],
        chains=structure["chains"],
        connections=structure["connections"].astype(Connection),
        interfaces=structure["interfaces"],
        mask=structure["mask"],
    )

    msas = {}
    for chain in record.chains:
        msa_id = chain.msa_id
        # Load the MSA for this chain, if any
        if msa_id != -1:
            msa = np.load(msa_dir / f"{msa_id}.npz")
            msas[chain.chain_id] = MSA(**msa)

    residue_constraints = None
    if constraints_dir is not None:
        residue_constraints = ResidueConstraints.load(
            constraints_dir / f"{record.id}.npz"
        )

    return Input(structure, msas, residue_constraints=residue_constraints)


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


class PredictionDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        constraints_dir: Optional[Path] = None,
        noesy_restraints_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
        target_dir : Path
            The path to the target directory.
        msa_dir : Path
            The path to the msa directory.

        """
        super().__init__()
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.constraints_dir = constraints_dir
        self.noesy_restraints_dir = noesy_restraints_dir
        self.tokenizer = BoltzTokenizer()
        self.featurizer = BoltzFeaturizer()

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Get a sample from the dataset
        record = self.manifest.records[idx]

        # Get the structure
        try:
            input_data = load_input(
                record,
                self.target_dir,
                self.msa_dir,
                self.constraints_dir,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized = self.tokenizer.tokenize(input_data)
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Inference specific options
        options = record.inference_options
        if options is None:
            binders, pocket = None, None
        else:
            binders, pocket = options.binders, options.pocket

        # Compute features
        try:
            features = self.featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=const.max_msa_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=binders,
                inference_pocket=pocket,
                compute_constraint_features=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        features["record"] = record

        # === ADDED NOESY PROCESSING LOGIC ===
        structure = input_data.structure # Get the loaded Structure object

        if self.noesy_restraints_dir and features["record"]: # Ensure record is loaded
            record_id = features["record"].id
            noesy_file_path = self.noesy_restraints_dir / f"{record_id}_noesy.npz"
            
            if noesy_file_path.exists():
                try:
                    # allow_pickle=True because atom_names were saved with dtype='O'
                    noesy_data_npz = np.load(noesy_file_path, allow_pickle=True) 

                    atom_names_1_tuples = noesy_data_npz['atom_names_1']
                    atom_names_2_tuples = noesy_data_npz['atom_names_2']

                    atom_names_1_str_list = ["".join(map(lambda x: chr(x + 32), name_tuple)).strip() for name_tuple in atom_names_1_tuples]
                    atom_names_2_str_list = ["".join(map(lambda x: chr(x + 32), name_tuple)).strip() for name_tuple in atom_names_2_tuples]
                    
                    res_indices_1_np = noesy_data_npz['res_idx_1']
                    res_indices_2_np = noesy_data_npz['res_idx_2']
                    
                    mapped_atom_indices_1 = []
                    mapped_atom_indices_2 = []
                    valid_restraint_indices = [] 

                    for i in range(len(res_indices_1_np)):
                        g_res_idx_1 = res_indices_1_np[i]
                        target_atom_name_1 = atom_names_1_str_list[i]
                        g_res_idx_2 = res_indices_2_np[i]
                        target_atom_name_2 = atom_names_2_str_list[i]
                        
                        atom_idx_1 = -1
                        atom_idx_2 = -1

                        if 0 <= g_res_idx_1 < len(structure.residues):
                            res1_struct = structure.residues[g_res_idx_1]
                            for current_atom_struct_idx_in_res_array in range(res1_struct['atom_idx'], res1_struct['atom_idx'] + res1_struct['atom_num']):
                                atom_struct = structure.atoms[current_atom_struct_idx_in_res_array]
                                current_atom_name_str = "".join(map(lambda x: chr(x + 32), atom_struct['name'])).strip()
                                if current_atom_name_str == target_atom_name_1:
                                    atom_idx_1 = current_atom_struct_idx_in_res_array
                                    break
                        
                        if 0 <= g_res_idx_2 < len(structure.residues):
                            res2_struct = structure.residues[g_res_idx_2]
                            for current_atom_struct_idx_in_res_array in range(res2_struct['atom_idx'], res2_struct['atom_idx'] + res2_struct['atom_num']):
                                atom_struct = structure.atoms[current_atom_struct_idx_in_res_array]
                                current_atom_name_str = "".join(map(lambda x: chr(x + 32), atom_struct['name'])).strip()
                                if current_atom_name_str == target_atom_name_2:
                                    atom_idx_2 = current_atom_struct_idx_in_res_array
                                    break
                        
                        if atom_idx_1 != -1 and atom_idx_2 != -1:
                            mapped_atom_indices_1.append(atom_idx_1)
                            mapped_atom_indices_2.append(atom_idx_2)
                            valid_restraint_indices.append(i)
                    
                    if not valid_restraint_indices:
                        features['noesy_atom_idx_1'] = torch.empty(0, dtype=torch.long)
                        features['noesy_atom_idx_2'] = torch.empty(0, dtype=torch.long)
                        features['noesy_target_distances'] = torch.empty(0, dtype=torch.float32)
                        features['noesy_tolerances'] = torch.empty(0, dtype=torch.float32)
                    else:
                        features['noesy_atom_idx_1'] = torch.tensor(mapped_atom_indices_1, dtype=torch.long)
                        features['noesy_atom_idx_2'] = torch.tensor(mapped_atom_indices_2, dtype=torch.long)
                        features['noesy_target_distances'] = torch.from_numpy(noesy_data_npz['target_distances'][valid_restraint_indices].astype(np.float32))
                        features['noesy_tolerances'] = torch.from_numpy(noesy_data_npz['tolerances'][valid_restraint_indices].astype(np.float32))

                except Exception as e:
                    print(f"Error loading or processing NOESY file {noesy_file_path} for record {record_id}: {e}")
                    features['noesy_atom_idx_1'] = torch.empty(0, dtype=torch.long)
                    features['noesy_atom_idx_2'] = torch.empty(0, dtype=torch.long)
                    features['noesy_target_distances'] = torch.empty(0, dtype=torch.float32)
                    features['noesy_tolerances'] = torch.empty(0, dtype=torch.float32)
            else: 
                features['noesy_atom_idx_1'] = torch.empty(0, dtype=torch.long)
                features['noesy_atom_idx_2'] = torch.empty(0, dtype=torch.long)
                features['noesy_target_distances'] = torch.empty(0, dtype=torch.float32)
                features['noesy_tolerances'] = torch.empty(0, dtype=torch.float32)
        else: 
            features['noesy_atom_idx_1'] = torch.empty(0, dtype=torch.long)
            features['noesy_atom_idx_2'] = torch.empty(0, dtype=torch.long)
            features['noesy_target_distances'] = torch.empty(0, dtype=torch.float32)
            features['noesy_tolerances'] = torch.empty(0, dtype=torch.float32)
        # === END OF ADDED NOESY LOGIC ===

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.manifest.records)


class BoltzInferenceDataModule(pl.LightningDataModule):
    """DataModule for Boltz inference."""

    def __init__(
        self,
        manifest: Manifest,
        target_dir: Path,
        msa_dir: Path,
        num_workers: int,
        constraints_dir: Optional[Path] = None,
        noesy_restraints_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the DataModule.

        Parameters
        ----------
        config : DataConfig
            The data configuration.

        """
        super().__init__()
        self.num_workers = num_workers
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.constraints_dir = constraints_dir
        self.noesy_restraints_dir = noesy_restraints_dir

    def predict_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        dataset = PredictionDataset(
            manifest=self.manifest,
            target_dir=self.target_dir,
            msa_dir=self.msa_dir,
            constraints_dir=self.constraints_dir,
            noesy_restraints_dir=self.noesy_restraints_dir,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate,
        )

    def transfer_batch_to_device(
        self,
        batch: dict,
        device: torch.device,
        dataloader_idx: int,  # noqa: ARG002
    ) -> dict:
        """Transfer a batch to the given device.

        Parameters
        ----------
        batch : Dict
            The batch to transfer.
        device : torch.device
            The device to transfer to.
        dataloader_idx : int
            The dataloader index.

        Returns
        -------
        np.Any
            The transferred batch.

        """
        for key in batch:
            if key not in [
                "all_coords",
                "all_resolved_mask",
                "crop_to_all_atom_map",
                "chain_symmetries",
                "amino_acids_symmetries",
                "ligand_symmetries",
                "record",
            ]:
                batch[key] = batch[key].to(device)
        return batch

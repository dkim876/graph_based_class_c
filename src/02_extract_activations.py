# 02_extract_activations.py

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "THGLab/Llama-3.1-8B-SmileyLlama-1.1"
MODEL_REVISION = "6179dd73b34f055e76ba09d6748baf6a80dcec55"


def regenerate_smiles(dataset_path, smiles_path, reference_path=None):
    with dataset_path.open(newline="", encoding="utf-8") as dataset_file:
        molecules = list(csv.DictReader(dataset_file))
    if not molecules:
        raise ValueError("The dataset contains no molecules")
    molecule_ids = [molecule["molecule_chembl_id"] for molecule in molecules]
    smiles = [molecule["smiles"] for molecule in molecules]
    if any(not molecule_id for molecule_id in molecule_ids) or len(set(molecule_ids)) != len(molecule_ids):
        raise ValueError("Molecule IDs must be present and unique")
    if any(not value or value != value.strip() or "\n" in value or "\r" in value for value in smiles):
        raise ValueError("SMILES must be nonempty single-line strings without surrounding whitespace")
    smiles_bytes = ("\n".join(smiles) + "\n").encode("utf-8")
    if reference_path is not None and smiles_bytes != reference_path.read_bytes():
        raise ValueError("Regenerated SMILES differ byte-for-byte from the reference")
    smiles_path.parent.mkdir(parents=True, exist_ok=True)
    smiles_path.write_bytes(smiles_bytes)
    print(f"Regenerated {len(smiles)} SMILES: {smiles_path}", flush=True)
    return smiles, np.asarray(molecule_ids, dtype=str)


@torch.inference_mode()
def extract_activations(smiles, batch_size, max_length, local_files_only):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run extraction inside a GPU allocation")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, revision=MODEL_REVISION, local_files_only=local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    token_lengths = tokenizer(smiles, padding=False, truncation=False, return_length=True)["length"]
    longest_input = max(token_lengths)
    if longest_input > max_length:
        raise ValueError(
            f"Longest input has {longest_input} tokens; increase --max-length from {max_length}"
        )
    print(f"Longest input: {longest_input} tokens", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=torch.float16,
        local_files_only=local_files_only,
    ).to("cuda").eval()
    activations = {
        f"layer_{layer}": np.empty((len(smiles), model.config.hidden_size), dtype=np.float32)
        for layer in range(model.config.num_hidden_layers + 1)
    }
    for start in range(0, len(smiles), batch_size):
        stop = min(start + batch_size, len(smiles))
        encoded = tokenizer(
            smiles[start:stop], return_tensors="pt", padding=True, truncation=False
        ).to("cuda")
        hidden_states = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states
        last_token_indices = encoded["attention_mask"].sum(dim=1) - 1
        batch_rows = torch.arange(stop - start, device="cuda")
        if len(hidden_states) != len(activations):
            raise ValueError("The model returned an unexpected number of hidden-state layers")
        for layer, hidden_state in enumerate(hidden_states):
            pooled = hidden_state[batch_rows, last_token_indices].float().cpu().numpy()
            if not np.isfinite(pooled).all():
                raise ValueError(f"Nonfinite activations in layer {layer}, rows {start}:{stop}")
            activations[f"layer_{layer}"][start:stop] = pooled
        del hidden_states, hidden_state, encoded
        print(f"Extracted {stop}/{len(smiles)} molecules", flush=True)
    return activations


def main():
    repository_root = Path(__file__).resolve().parents[1]
    processed_directory = repository_root / "data/processed"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=processed_directory / "d2_d3_m50.csv")
    parser.add_argument("--smiles", type=Path, default=processed_directory / "smiles.txt")
    parser.add_argument("--out", type=Path, default=processed_directory / "activations.npz")
    parser.add_argument("--reference-smiles", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        parser.error("--batch-size and --max-length must be positive")
    if args.out.suffix != ".npz":
        parser.error("--out must end in .npz")
    input_paths = {args.dataset.resolve()}
    if args.reference_smiles is not None:
        input_paths.add(args.reference_smiles.resolve())
    if args.smiles.resolve() in input_paths or args.out.resolve() in input_paths:
        parser.error("Output paths must differ from dataset and reference inputs")
    if args.smiles.resolve() == args.out.resolve():
        parser.error("SMILES and activation outputs must have different paths")
    smiles, molecule_ids = regenerate_smiles(args.dataset, args.smiles, args.reference_smiles)
    activations = extract_activations(
        smiles, args.batch_size, args.max_length, args.local_files_only
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.out.with_suffix(".tmp.npz")
    np.savez(temporary_output, molecule_ids=molecule_ids, **activations)
    temporary_output.replace(args.out)
    print(f"Saved {len(activations)} layers to: {args.out}", flush=True)


if __name__ == "__main__":
    main()

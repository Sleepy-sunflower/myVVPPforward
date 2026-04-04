import os

import matplotlib.pyplot as plt
import meshio
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torchaudio
import torchvision.transforms as T
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


VVPP_DATA_DIR = "/Users/bobo/Codes/vv-impact/data/vv++test"


class VVImpactDataset(Dataset):
    def __init__(self, data_dir=VVPP_DATA_DIR, sample_rate=16000, n_mels=64, transform_image=None):
        self.data_dir = self.resolve_data_dir(data_dir)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.preview_transform = transform_image or T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        self.spec_transform = T.ToTensor()
        self.samples = []
        self.mesh_cache = {}
        self.resampler_cache = {}

        specs_dir = os.path.join(self.data_dir, "impact_specs")
        audio_dir = os.path.join(self.data_dir, "impact_audio")
        msh_dir = os.path.join(self.data_dir, "msh")
        if not os.path.isdir(specs_dir) or not os.path.isdir(audio_dir) or not os.path.isdir(msh_dir):
            return

        for group in sorted(os.listdir(specs_dir)):
            group_specs_dir = os.path.join(specs_dir, group)
            group_audio_dir = os.path.join(audio_dir, group)
            group_msh_dir = os.path.join(msh_dir, group)
            if not os.path.isdir(group_specs_dir) or not os.path.isdir(group_audio_dir) or not os.path.isdir(group_msh_dir):
                continue
            for obj_id in sorted(os.listdir(group_specs_dir)):
                obj_specs_dir = os.path.join(group_specs_dir, obj_id)
                obj_audio_dir = os.path.join(group_audio_dir, obj_id)
                msh_path = os.path.join(group_msh_dir, f"{obj_id}.obj_.msh")
                if not os.path.isdir(obj_specs_dir) or not os.path.isdir(obj_audio_dir) or not os.path.exists(msh_path):
                    continue
                impacts = []
                for spec_name in sorted(os.listdir(obj_specs_dir)):
                    if not spec_name.startswith("audio_") or not spec_name.endswith(".png"):
                        continue
                    vertex_id = int(spec_name.split("_")[1].split(".")[0])
                    wav_path = os.path.join(obj_audio_dir, f"audio_{vertex_id}.wav")
                    if not os.path.exists(wav_path):
                        continue
                    impacts.append({
                        "vertex_id": vertex_id,
                        "spec_path": os.path.join(obj_specs_dir, spec_name),
                        "wav_path": wav_path,
                    })
                if impacts:
                    self.samples.append({
                        "group": group,
                        "obj_id": obj_id,
                        "msh_path": msh_path,
                        "samples": impacts,
                    })

    def resolve_data_dir(self, data_dir):
        candidates = [data_dir, os.path.join(data_dir, "vv++test"), VVPP_DATA_DIR]
        for candidate in candidates:
            if candidate and os.path.isdir(os.path.join(candidate, "impact_specs")) and os.path.isdir(os.path.join(candidate, "msh")):
                return candidate
        return VVPP_DATA_DIR

    def __len__(self):
        return len(self.samples)

    def load_mesh(self, msh_path):
        mesh = self.mesh_cache.get(msh_path)
        if mesh is None:
            msh = meshio.read(msh_path)
            tetra = msh.cells_dict.get("tetra")
            if tetra is None:
                tetra = next((cell_block.data for cell_block in msh.cells if cell_block.type == "tetra"), [])
            mesh = {
                "vertices": torch.tensor(msh.points, dtype=torch.float32),
                "tetra": torch.tensor(tetra, dtype=torch.long),
            }
            self.mesh_cache[msh_path] = mesh
        return mesh

    def load_spec(self, spec_path):
        spec_image = Image.open(spec_path).convert("L")
        spec_tensor = self.spec_transform(spec_image).squeeze(0)
        preview_tensor = self.preview_transform(spec_image.convert("RGB"))
        return spec_tensor, preview_tensor

    def load_waveform(self, wav_path):
        sample_rate, waveform_np = wavfile.read(wav_path)
        if np.issubdtype(waveform_np.dtype, np.integer):
            waveform_np = waveform_np.astype(np.float32) / np.iinfo(waveform_np.dtype).max
        elif waveform_np.dtype != np.float32:
            waveform_np = waveform_np.astype(np.float32)
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)
        if sample_rate != self.sample_rate:
            resampler = self.resampler_cache.get(sample_rate)
            if resampler is None:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sample_rate)
                self.resampler_cache[sample_rate] = resampler
            waveform = resampler(waveform.unsqueeze(0)).squeeze(0)
        return waveform

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mesh = self.load_mesh(sample["msh_path"])
        impact_specs = []
        impact_images = []
        waveforms = []
        impact_vertex_index = []
        impact_spec_path = []
        impact_audio_path = []

        for impact in sample["samples"]:
            spec_tensor, preview_tensor = self.load_spec(impact["spec_path"])
            impact_specs.append(spec_tensor)
            impact_images.append(preview_tensor)
            waveforms.append(self.load_waveform(impact["wav_path"]))
            impact_vertex_index.append(impact["vertex_id"])
            impact_spec_path.append(impact["spec_path"])
            impact_audio_path.append(impact["wav_path"])

        impact_vertex_index = torch.tensor(impact_vertex_index, dtype=torch.long)
        impact_point = mesh["vertices"][impact_vertex_index]
        mel_spectrogram = torch.stack(impact_specs)
        impact_image = torch.stack(impact_images)
        waveform = pad_sequence(waveforms, batch_first=True)
        waveform_length = torch.tensor([wave.size(0) for wave in waveforms], dtype=torch.long)

        return {
            "mel_spectrogram": mel_spectrogram,
            "impact_image": impact_image,
            "waveform": waveform,
            "waveform_length": waveform_length,
            "sample_rate": self.sample_rate,
            "mesh_vertices": mesh["vertices"],
            "mesh_tetra": mesh["tetra"],
            "mesh": {"vertices": mesh["vertices"], "tetra": mesh["tetra"]},
            "impact_point": impact_point,
            "impact_vertex_index": impact_vertex_index,
            "num_impacts": torch.tensor(impact_vertex_index.numel(), dtype=torch.long),
            "mesh_path": sample["msh_path"],
            "msh_path": sample["msh_path"],
            "obj_id": sample["obj_id"],
            "group": sample["group"],
            "vertex_id": impact_vertex_index.clone(),
            "impact_spec_path": impact_spec_path,
            "impact_audio_path": impact_audio_path,
        }


def collate_vvimpact_batch(batch):
    return {
        "mel_spectrogram": [item["mel_spectrogram"] for item in batch],
        "impact_image": [item["impact_image"] for item in batch],
        "waveform": [item["waveform"] for item in batch],
        "waveform_length": [item["waveform_length"] for item in batch],
        "sample_rate": batch[0]["sample_rate"],
        "mesh_vertices": [item["mesh_vertices"] for item in batch],
        "mesh_tetra": [item["mesh_tetra"] for item in batch],
        "mesh": [item["mesh"] for item in batch],
        "impact_point": [item["impact_point"] for item in batch],
        "impact_vertex_index": [item["impact_vertex_index"] for item in batch],
        "num_impacts": torch.stack([item["num_impacts"] for item in batch]),
        "mesh_path": [item["mesh_path"] for item in batch],
        "msh_path": [item["msh_path"] for item in batch],
        "obj_id": [item["obj_id"] for item in batch],
        "group": [item["group"] for item in batch],
        "vertex_id": [item["vertex_id"] for item in batch],
        "impact_spec_path": [item["impact_spec_path"] for item in batch],
        "impact_audio_path": [item["impact_audio_path"] for item in batch],
    }


def visualize_sample(batch, save_path="sample_visualization.png"):
    spec = batch["mel_spectrogram"][0][0].numpy()
    impact_image = batch["impact_image"][0][0].permute(1, 2, 0).numpy()
    vertices = batch["mesh_vertices"][0].numpy()
    impact_points = batch["impact_point"][0].numpy()
    obj_id = batch["obj_id"][0]
    vertex_id = batch["vertex_id"][0][0].item()

    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(impact_image)
    ax1.set_title("Impact Spec Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 3, 2)
    cax = ax2.imshow(spec, aspect="auto", origin="lower", cmap="magma")
    ax2.set_title("Impact Spec Tensor")
    fig.colorbar(cax, ax=ax2)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=1, alpha=0.08)
    ax3.scatter(impact_points[:, 0], impact_points[:, 1], impact_points[:, 2], color="red", s=20)
    ax3.set_title(f"Mesh {obj_id} | Vertex {vertex_id}")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    dataset = VVImpactDataset()
    print(f"Total valid meshes found: {len(dataset)}")
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_vvimpact_batch)
        batch = next(iter(dataloader))
        print("Batch keys:", batch.keys())
        print("Meshes per batch:", len(batch["mesh_vertices"]))
        print("Impacts per mesh:", batch["num_impacts"].tolist())
        visualize_sample(batch)

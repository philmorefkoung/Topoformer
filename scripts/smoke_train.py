"""Run tiny Topoformer and TopoGate training/held-out evaluations on CPU.

This is an execution check on synthetic data, not a benchmark or paper reproduction.
No pretrained weights or datasets are downloaded. Temporary data/checkpoints are
removed after the checks complete.
"""

import json
import os
from pathlib import Path
import sys
import tempfile

# This standalone check deliberately runs on CPU, including on GPU hosts.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Topoformer import Config, Topoformer, build_dataloaders, evaluate, set_seed, train
from TopoGate import (
    Config as GateConfig,
    TopoGate3D,
    TopoVolumeDataset,
    evaluate as evaluate_gate,
    train as train_gate,
)


def run_topogate(directory, volumes, feature_matrices, labels):
    """Exercise the unchanged TopoGate trainer, then evaluate held-out samples."""
    gate_dir = directory / "topogate"
    gate_dir.mkdir()
    for split, images in volumes.items():
        np.savez_compressed(gate_dir / f"{split}.npz", **{f"{split}_images": images, f"{split}_labels": labels})
        features = feature_matrices[split]
        np.savetxt(gate_dir / f"M20_{split}.csv", features, delimiter=",")
        # A second synthetic view exercises learned weighting over two widths.
        np.savetxt(gate_dir / f"M40_{split}.csv", np.roll(features, 17, axis=1), delimiter=",")
    cfg = GateConfig(
        train_npz=str(gate_dir / "train.npz"),
        val_npz=str(gate_dir / "val.npz"),
        train_csvs=[str(gate_dir / f"{width}_train.csv") for width in ("M20", "M40")],
        val_csvs=[str(gate_dir / f"{width}_val.csv") for width in ("M20", "M40")],
        topo_dim=150,
        num_classes=2,
        batch_size=2,
        epochs=1,
        num_workers=0,
        amp=False,
        use_pretrained=False,
        ckpt_dir=str(gate_dir / "checkpoints"),
        run_name="smoke",
        seed=42,
    )
    set_seed(cfg.seed)
    initial = TopoGate3D(topo_dim=150, num_classes=2, n_widths=2, use_pretrained=False)
    initial_gate = initial.gate.head[0].weight.detach().clone()
    initial_classifier = initial.classifier.classifier.weight.detach().clone()
    del initial
    train_gate(cfg)
    checkpoint = torch.load(gate_dir / "checkpoints/smoke_best.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert np.isfinite(checkpoint["val_auc"])
    assert not torch.equal(initial_gate, checkpoint["model_state"]["gate.head.0.weight"])
    assert not torch.equal(initial_classifier, checkpoint["model_state"]["classifier.classifier.weight"])
    model = TopoGate3D(topo_dim=150, num_classes=2, n_widths=2, use_pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    test_dataset = TopoVolumeDataset(
        gate_dir / "test.npz",
        "test",
        [gate_dir / f"{width}_test.csv" for width in ("M20", "M40")],
        mu=checkpoint["mu"],
        sd=checkpoint["sd"],
        topo_dim=150,
    )
    assert len(test_dataset) == 4
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=False)
    metrics = evaluate_gate(model, test_loader, num_classes=2, device=torch.device("cpu"))
    assert all(np.isfinite(value) for value in metrics.values()), metrics
    return {
        "train_samples": 4,
        "validation_samples": 4,
        "held_out_test_samples": len(test_dataset),
        "epochs": 1,
        "optimizer_steps": 2,
        "gate_updated": True,
        "classifier_updated": True,
        "checkpoint_loaded_for_test": True,
        "test_metrics": metrics,
    }


def main():
    # Keep the synthetic checks small on CPU.
    torch.set_num_threads(2)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(123)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    with tempfile.TemporaryDirectory(prefix="topoformer-smoke-") as temporary:
        directory = Path(temporary)
        volumes = {}
        feature_matrices = {}
        for split in ("train", "val", "test"):
            images = rng.normal(size=(4, 16, 16, 16)).astype(np.float32)
            images[:, 4:12, 4:12, 4:12] += labels[:, None, None, None]
            features = rng.normal(size=(4, 150)).astype(np.float32)
            features[:, :20] += labels[:, None]
            np.savez_compressed(directory / f"{split}.npz", images=images, labels=labels)
            np.savetxt(directory / f"M20_{split}.csv", features, delimiter=",")
            volumes[split] = images
            feature_matrices[split] = features
        assert not np.array_equal(volumes["train"], volumes["test"])
        assert not np.array_equal(volumes["val"], volumes["test"])

        cfg = Config(
            img_prefix=str(directory),
            topo_prefix=str(directory),
            topo_dim=150,
            topo_block=150,
            batch_size=2,
            epochs=1,
            num_workers=0,
            amp=False,
            use_pretrained=False,
            ckpt_dir=str(directory / "checkpoints"),
            run_name="smoke",
            seed=42,
        )
        set_seed(cfg.seed)
        initial_model = Topoformer(num_classes=2, topo_dim=150, pretrained_weights=None)
        initial_classifier = initial_model.classifier.weight.detach().clone()
        initial_backbone = initial_model.backbone.stem[0].weight.detach().clone()
        del initial_model

        metrics = train(cfg)
        assert all(np.isfinite(value) for value in metrics.values()), metrics
        checkpoint = torch.load(directory / "checkpoints/smoke_best.pt", map_location="cpu", weights_only=True)
        assert checkpoint["epoch"] == 1
        assert not torch.equal(initial_classifier, checkpoint["model_state"]["classifier.weight"])
        assert not torch.equal(initial_backbone, checkpoint["model_state"]["backbone.stem.0.weight"])

        # Verify the saved model reproduces the held-out metrics after reloading.
        reloaded = Topoformer(num_classes=2, topo_dim=150, pretrained_weights=None)
        reloaded.load_state_dict(checkpoint["model_state"])
        _, _, test_loader, num_classes = build_dataloaders(cfg, cfg.in_ch)
        reloaded_metrics = evaluate(reloaded, test_loader, num_classes, torch.device("cpu"))
        for name, value in metrics.items():
            np.testing.assert_allclose(reloaded_metrics[name], value, rtol=1e-6, atol=1e-7)

        topoformer_result = {
            "train_samples": 4,
            "validation_samples": 4,
            "held_out_test_samples": len(test_loader.dataset),
            "epochs": 1,
            "optimizer_steps": 2,
            "backbone_updated": True,
            "classifier_updated": True,
            "checkpoint_reload_matches": True,
            "test_metrics": metrics,
        }
        del reloaded, checkpoint
        topogate_result = run_topogate(directory, volumes, feature_matrices, labels)
        print(json.dumps({
            "status": "passed",
            "device": "cpu",
            "pretrained": False,
            "volume_shape": [16, 16, 16],
            "topoformer": topoformer_result,
            "topogate": topogate_result,
        }, indent=2))


if __name__ == "__main__":
    main()

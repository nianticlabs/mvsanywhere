import torch

from src.geometryhints.modules.confidence import VolumeEntropyConfidence


def build_fake_volume_high_entropy() -> torch.Tensor:
    """Creates a fake cost volume with values from 1 and 64"""
    scores = torch.linspace(1, 64, 64).reshape(1, 64, 1, 1).repeat(1, 1, 12, 12)
    return scores


def build_fake_volume_low_entropy() -> torch.Tensor:
    """Creates a fake cost volume with values from 1 and 64"""
    scores = torch.ones((1, 64, 12, 12), dtype=torch.float32)
    scores[:, 1, :, :] = 100.0
    return scores


def test_volume_entropy_confidence_high_entropy():
    volume = build_fake_volume_high_entropy()
    conf_metric = VolumeEntropyConfidence()
    confidence = conf_metric.forward(volume=volume)
    assert confidence.shape == (1, 1, 12, 12)
    assert torch.isclose(confidence[0, 0, 0, 0], torch.tensor(1.0405), atol=1e-4)


def test_volume_entropy_confidence_low_entropy():
    volume = build_fake_volume_low_entropy()
    conf_metric = VolumeEntropyConfidence()
    confidence = conf_metric.forward(volume=volume)
    assert confidence.shape == (1, 1, 12, 12)
    assert torch.isclose(
        confidence[0, 0, 0, 0], torch.tensor(-1.0014e-05, dtype=torch.float32), atol=1e-4
    )

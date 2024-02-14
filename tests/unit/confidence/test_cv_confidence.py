import pytest
import torch

from src.geometryhints.modules.confidence import MM, MSM


def build_fake_volume() -> torch.Tensor:
    """Creates a fake cost volume with values from 1 and 64"""
    scores = torch.linspace(1, 64, 64).reshape(1, 64, 1, 1).repeat(1, 1, 12, 12)
    return scores


@pytest.mark.parametrize("use_sigmoid", (True, False))
def test_MM_cost(use_sigmoid: bool):
    volume = build_fake_volume()
    mm_metric = MM(is_cost=True, use_sigmoid=use_sigmoid)
    confidence = mm_metric.forward(volume=volume)
    assert confidence.shape == (1, 1, 12, 12)

    expected = torch.sigmoid(torch.tensor(1)) if use_sigmoid else 1
    assert confidence[0, 0, 0, 0] == expected


@pytest.mark.parametrize("use_sigmoid", (True, False))
def test_MM_similarity(use_sigmoid: bool):
    volume = build_fake_volume()
    mm_metric = MM(is_cost=False, use_sigmoid=use_sigmoid)
    confidence = mm_metric.forward(volume=volume)
    assert confidence.shape == (1, 1, 12, 12)

    expected = torch.sigmoid(torch.tensor(1)) if use_sigmoid else 1
    assert confidence[0, 0, 0, 0] == expected


@pytest.mark.parametrize("use_sigmoid", (True, False))
def test_MSM_cost(use_sigmoid: bool):
    volume = build_fake_volume()
    msm_metric = MSM(is_cost=True, use_sigmoid=use_sigmoid)

    confidence = msm_metric.forward(volume=volume)
    assert confidence.shape == (1, 1, 12, 12)

    expected = torch.sigmoid(torch.tensor(-1)) if use_sigmoid else -1
    assert confidence[0, 0, 0, 0] == expected


@pytest.mark.parametrize("use_sigmoid", (True, False))
def test_MSM_similarity(use_sigmoid: bool):
    volume = build_fake_volume()
    msm_metric = MSM(is_cost=False, use_sigmoid=use_sigmoid)

    confidence = msm_metric.forward(volume=volume)
    assert confidence.shape == (1, 1, 12, 12)

    expected = torch.sigmoid(torch.tensor(64)) if use_sigmoid else 64
    assert confidence[0, 0, 0, 0] == expected

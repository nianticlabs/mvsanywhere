import torch


class ConfidenceMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()


class VolumeEntropyConfidence(ConfidenceMetric):
    """Variance along each epipolar line"""

    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Compute the entropy along each ray.
        Params:
            volume: tensor (B,D,H,W), where D is the number of depth bins
        Returns:
            a tensor (B,1,H,W) representing the entropy along each ray. The higher the entropy,
            the lower the confidence.
        """
        prob = self.softmax(volume)
        conf = -torch.sum(prob * torch.log(prob + 1e-5), dim=1, keepdim=True)
        return conf

import torch


class ConfidenceMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()


class VolumeEntropyConfidence(ConfidenceMetric):
    """Variance along each epipolar line"""

    def __init__(self, bound: bool = False):
        """
        Params:
            bound: if True, returns a confidence score that is bounded between [0,1], where 1 means
            high confidence and 0 low confidence. Otherwise, returns an unbounded confidence
            where the higher the less confident (higher => high entropy)
        """
        super().__init__()
        self.bound = bound
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Compute the entropy along each ray.
        Params:
            volume: tensor (B,D,H,W), where D is the number of depth bins
        Returns:
            a tensor (B,1,H,W) representing the entropy along each ray
        """
        prob = self.softmax(volume)
        conf = -torch.sum(prob * torch.log(prob + 1e-5), dim=1, keepdim=True)
        if self.bound:
            # when conf is high (bad), we return a low value.
            # when conf is low (good), we return a high value
            return 1.0 / (1.0 + conf)
        return conf


def compute_volume_entropy(volume: torch.Tensor):
    prob = torch.nn.functional.softmax(input=volume, dim=1)
    conf = -torch.sum(prob * torch.log(prob + 1e-5), dim=1, keepdim=True)
    return 1.0 / (1.0 + conf)

import torch


class ConfidenceMetric(torch.nn.Module):
    def __init__(self):
        super().__init__()


class MSM(ConfidenceMetric):
    """Matching Score confidence measure"""

    def __init__(self, is_cost: bool, use_sigmoid: bool = False):
        """
        Params:
            is_cost: if true, the volume is a cost volume. Otherwise, it is a similarity volume
            use_sigmoid: if true, values are squashed between [0,1]
        """
        super().__init__()
        self.is_cost = is_cost
        self.use_sigmoid = use_sigmoid

    def forward(self, volume: torch.Tensor):
        """Compute Matching Score confidence.
        It gives the minimum (in case of a cost volume) or max (in case of similarity volume)
        score for each depth plane.

        Params:
            volume: tensor (B,D,H,W) representing a volume of costs or similarity scores
        Returns:
            a confidence map (B,1,H,W). The higher the value the better the confidence.
        """
        if self.is_cost:
            conf, _ = torch.min(volume, dim=1, keepdim=True)
            conf = -1 * conf
        else:
            conf, _ = torch.max(volume, dim=1, keepdim=True)

        if self.use_sigmoid:
            conf = torch.sigmoid(conf)
        return conf


class MMN(ConfidenceMetric):
    """Maximum Margin Naive confidence measure"""

    def __init__(self, is_cost: bool, use_sigmoid: bool = False):
        """
        Params:
            is_cost: if true, the volume is a cost volume. Otherwise, it is a similarity volume
            use_sigmoid: if true, values are squashed between [0,1]
        """
        super().__init__()
        self.is_cost = is_cost
        self.use_sigmoid = use_sigmoid

    def forward(self, volume: torch.Tensor):
        """Maximum Margin Naive confidence.
        It computes the difference between the top2 min (max) and the top1 min(max) of the
        cost (similarity) for each depth plane.

        Params:
            volume: tensor (B,D,H,W) representing a volume of costs or similarity scores
        Returns:
            a confidence map (B,1,H,W).
        """
        if self.is_cost:
            # the difference between top2 and top1 is positive (top2>=top1), and the
            # larger the difference the better top1 wrt top2 => high confidence!
            tops, _ = torch.topk(volume, dim=1, k=2, sorted=True, largest=False)
            top1 = tops[:, 0:1]
            top2 = tops[:, 1:2]
            conf = top2 - top1
        else:
            # the difference between top1 and top2 is positive (top1>=top2), and the
            # larger the difference the better top1 wrt top2 => high confidence!
            tops, _ = torch.topk(volume, dim=1, k=2, sorted=True, largest=True)
            top1 = tops[:, 0:1]
            top2 = tops[:, 1:2]
            conf = top1 - top2
        if self.use_sigmoid:
            conf = torch.sigmoid(conf)
        return conf


class NeuralRankNDCG:
    def __init__(self, loss_model):
        super().__init__()
        self.padd_idx = loss_model.padd_idx
        self.dcg = loss_model
        self.dcg.requires_grad_(False)

    def __call__(self, preds, targets):
        self.dcg.to(preds.device)
        preds = preds.unsqueeze(-1)
        masks = targets == -1
        targets[masks] = self.padd_idx
        dcg = self.dcg(preds, targets.long(), masks)
        idcg = self.dcg((targets/self.padd_idx).unsqueeze(-1), targets.long(), masks)
        ndcg = (dcg/(idcg+0.000001)).nan_to_num(0)  # idcg may be 0

        return -ndcg.mean()


class NeuralRankRecall:
    def __init__(self, loss_model):
        super().__init__()
        self.padd_idx = loss_model.padd_idx
        self.recall = loss_model
        self.recall.requires_grad_(False)

    def __call__(self, preds, targets):
        self.recall.to(preds.device)
        preds = preds.unsqueeze(-1)
        masks = targets == -1
        targets[masks] = self.padd_idx
        recall = self.recall(preds, targets.long(), masks)

        return -recall.mean()

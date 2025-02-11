from mmengine.evaluator import BaseMetric
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from typing import List, Sequence

from mmengine.evaluator import BaseMetric

from mmseg.registry import METRICS

@METRICS.register_module()
class RetinalMetrics(BaseMetric):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.targets = np.array([], dtype=np.float32)
        self.outputs = np.array([], dtype=np.float32)

    def process(self, data_batch: dict, data_samples: list) -> None:
        for data_sample in data_samples:
            # pred_label = data_sample['pred_sem_seg']['data'].squeeze()
            # label = data_sample['gt_sem_seg']['data'].squeeze().to(
            #     pred_label)
            logits = data_sample['pred_sem_seg']['data'].sigmoid().detach().cpu().numpy().flatten()
            gt_masks = data_sample['gt_sem_seg']['data'].detach().cpu().numpy().flatten()

            self.targets = np.append(self.targets, gt_masks)

            self.outputs = np.append(self.outputs, logits)

    def compute_metrics(self) -> dict:

        auc = roc_auc_score(self.targets, self.outputs)
        auc = auc*100
        # confusion = confusion_matrix(self.targets, preds)
        # print(confusion)

        # acc = np.trace(confusion) / np.sum(confusion)
        # tn, fp, fn, tp = confusion.ravel()
        # specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        # sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        return {
            'AUC': auc,
            # 'Accuracy': acc,
            # 'Specificity': specificity,
            # 'Sensitivity': sensitivity
        }

    def evaluate(self, size: int) -> dict:
        # 'size' parameter is accepted but not used in this example
        metrics = self.compute_metrics()
        self.reset()  # Reset after each evaluation cycle
        return metrics



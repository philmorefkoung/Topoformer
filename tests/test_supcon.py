"""Regression tests for the correspondence between labels and modality views."""

import unittest

import torch
import torch.nn.functional as F

from Topoformer import SupConLoss


def reference_supcon(features, sample_labels, temperature):
    """Compute the objective anchor by anchor from sample/view identities."""
    batch_size = len(sample_labels)
    losses = []
    for anchor in range(len(features)):
        others = [index for index in range(len(features)) if index != anchor]
        logits = features[anchor] @ features[others].T / temperature
        log_probabilities = F.log_softmax(logits, dim=0)
        positives = [
            offset
            for offset, other in enumerate(others)
            if sample_labels[anchor % batch_size] == sample_labels[other % batch_size]
        ]
        losses.append(-log_probabilities[positives].mean())
    return torch.stack(losses).mean()


class SupConCorrespondenceTests(unittest.TestCase):
    def test_matches_independent_reference_and_gradients(self):
        # A repeated class checks every same-class positive, not just paired views.
        labels = torch.tensor([0, 1, 0])
        generator = torch.Generator().manual_seed(19)
        features = F.normalize(torch.randn(6, 5, generator=generator, dtype=torch.float64), dim=1)
        features.requires_grad_()
        actual = SupConLoss(temp=0.4)(features, labels)
        expected = reference_supcon(features, labels.tolist(), temperature=0.4)
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-10)
        actual_gradient = torch.autograd.grad(actual, features, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, features)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-8, atol=1e-10)

    def test_matching_modalities_have_lower_loss_than_swapped_modalities(self):
        labels = torch.tensor([0, 1])
        image_views = torch.eye(2)
        matched = torch.cat([image_views, image_views])
        mismatched = torch.cat([image_views, image_views.flip(0)])
        objective = SupConLoss(temp=0.2)
        self.assertLess(objective(matched, labels).item(), objective(mismatched, labels).item())

    def test_sample_permutation_preserves_loss(self):
        labels = torch.tensor([0, 1, 1, 0])
        generator = torch.Generator().manual_seed(23)
        features = F.normalize(torch.randn(8, 5, generator=generator), dim=1)
        permutation = torch.tensor([2, 0, 3, 1])
        permuted = torch.cat([features[:4][permutation], features[4:][permutation]])
        objective = SupConLoss(temp=0.4)
        torch.testing.assert_close(objective(features, labels), objective(permuted, labels[permutation]))


if __name__ == "__main__":
    unittest.main()

"""Tests for model architectures and routing logic."""

import pytest
import torch
import torch.nn as nn
import numpy as np

from adaptive_inference_router_with_cascade_serving.models.model import (
    MultiHeadAttention,
    FeatureEncoder,
    RoutePredictor,
    ModelCascade,
    AdaptiveInferenceRouter
)


class TestMultiHeadAttention:
    """Test cases for MultiHeadAttention."""

    @pytest.mark.unit
    def test_attention_initialization(self):
        """Test attention module initialization."""
        attention = MultiHeadAttention(d_model=128, num_heads=8, dropout=0.1)

        assert attention.d_model == 128
        assert attention.num_heads == 8
        assert attention.d_k == 16  # 128 / 8

    @pytest.mark.unit
    def test_attention_forward(self, custom_assertions):
        """Test attention forward pass."""
        batch_size, seq_len, d_model = 4, 10, 128
        attention = MultiHeadAttention(d_model=d_model, num_heads=8)

        x = torch.randn(batch_size, seq_len, d_model)
        output = attention(x)

        custom_assertions.assert_tensor_shape(output, (batch_size, seq_len, d_model))

    @pytest.mark.unit
    def test_attention_with_mask(self):
        """Test attention with mask."""
        batch_size, seq_len, d_model = 2, 5, 64
        attention = MultiHeadAttention(d_model=d_model, num_heads=4)

        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, 4, seq_len, seq_len)  # num_heads in dim 1
        mask[:, :, 2:, 2:] = 0  # Mask last 3 positions

        output = attention(x, mask)

        assert output.shape == x.shape

    @pytest.mark.unit
    def test_attention_invalid_dimensions(self):
        """Test attention with invalid dimensions."""
        with pytest.raises(AssertionError):
            # d_model not divisible by num_heads
            MultiHeadAttention(d_model=127, num_heads=8)


class TestFeatureEncoder:
    """Test cases for FeatureEncoder."""

    @pytest.mark.unit
    def test_encoder_initialization(self):
        """Test encoder initialization."""
        encoder = FeatureEncoder(
            input_dim=20,
            hidden_dim=128,
            num_layers=3,
            dropout=0.1,
            use_attention=True,
            attention_heads=8
        )

        assert encoder.use_attention is True
        assert hasattr(encoder, 'attention')

    @pytest.mark.unit
    def test_encoder_forward_without_attention(self, custom_assertions):
        """Test encoder forward pass without attention."""
        batch_size, input_dim = 8, 20
        hidden_dim = 128

        encoder = FeatureEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            use_attention=False
        )

        x = torch.randn(batch_size, input_dim)
        output = encoder(x)

        custom_assertions.assert_tensor_shape(output, (batch_size, hidden_dim))

    @pytest.mark.unit
    def test_encoder_forward_with_attention(self, custom_assertions):
        """Test encoder forward pass with attention."""
        batch_size, input_dim = 8, 20
        hidden_dim = 128

        encoder = FeatureEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            use_attention=True,
            attention_heads=8
        )

        x = torch.randn(batch_size, input_dim)
        output = encoder(x)

        custom_assertions.assert_tensor_shape(output, (batch_size, hidden_dim))

    @pytest.mark.unit
    def test_encoder_residual_connections(self):
        """Test that encoder preserves information through residual connections."""
        encoder = FeatureEncoder(
            input_dim=20,
            hidden_dim=128,
            num_layers=1,
            dropout=0.0  # No dropout for this test
        )

        x = torch.randn(8, 20)

        # Test that output is not zero (residual connections preserve information)
        with torch.no_grad():
            output = encoder(x)
            assert not torch.allclose(output, torch.zeros_like(output))


class TestRoutePredictor:
    """Test cases for RoutePredictor."""

    @pytest.mark.unit
    def test_predictor_initialization(self):
        """Test predictor initialization."""
        predictor = RoutePredictor(
            feature_dim=128,
            num_routes=4,
            hidden_dim=256,
            num_objectives=4
        )

        assert predictor.num_routes == 4
        assert predictor.num_objectives == 4
        assert len(predictor.performance_heads) == 4

    @pytest.mark.unit
    def test_predictor_forward(self, custom_assertions):
        """Test predictor forward pass."""
        batch_size, feature_dim = 8, 128
        num_routes, num_objectives = 4, 4

        predictor = RoutePredictor(
            feature_dim=feature_dim,
            num_routes=num_routes,
            num_objectives=num_objectives
        )

        features = torch.randn(batch_size, feature_dim)
        predictions = predictor(features)

        # Check output structure
        assert isinstance(predictions, dict)
        assert "route_logits" in predictions
        assert "route_probs" in predictions
        assert "performance_preds" in predictions
        assert "values" in predictions

        # Check shapes
        custom_assertions.assert_tensor_shape(
            predictions["route_logits"], (batch_size, num_routes)
        )
        custom_assertions.assert_tensor_shape(
            predictions["route_probs"], (batch_size, num_routes)
        )
        custom_assertions.assert_tensor_shape(
            predictions["performance_preds"], (batch_size, num_routes, num_objectives)
        )
        custom_assertions.assert_tensor_shape(
            predictions["values"], (batch_size,)
        )

        # Check that probabilities are valid
        custom_assertions.assert_probability_distribution(predictions["route_probs"])

    @pytest.mark.unit
    def test_sample_action(self, setup_seeds, custom_assertions):
        """Test action sampling."""
        predictor = RoutePredictor(feature_dim=128, num_routes=4)

        route_logits = torch.randn(8, 4)
        actions, log_probs = predictor.sample_action(route_logits)

        custom_assertions.assert_tensor_shape(actions, (8,))
        custom_assertions.assert_tensor_shape(log_probs, (8,))

        # Check that actions are valid indices
        assert torch.all(actions >= 0)
        assert torch.all(actions < 4)


class TestModelCascade:
    """Test cases for ModelCascade."""

    @pytest.mark.unit
    def test_cascade_initialization(self, test_config):
        """Test cascade initialization."""
        cascade = ModelCascade(test_config)

        assert cascade.num_variants == 4
        assert hasattr(cascade, 'performance_matrix')
        assert cascade.performance_matrix.shape == (4, 3)

    @pytest.mark.unit
    def test_predict_performance(self, test_config, custom_assertions):
        """Test performance prediction."""
        cascade = ModelCascade(test_config)

        batch_size = 8
        route_indices = torch.randint(0, 4, (batch_size,))
        base_metrics = torch.randn(batch_size, 3)

        predicted_metrics = cascade.predict_performance(route_indices, base_metrics)

        custom_assertions.assert_tensor_shape(predicted_metrics, (batch_size, 3))

    @pytest.mark.unit
    def test_get_variant_names(self, test_config):
        """Test variant name extraction."""
        cascade = ModelCascade(test_config)
        variant_names = cascade.get_variant_names()

        expected_names = [
            "quantized_int8", "pruned_50", "distilled", "full_precision"
        ]
        assert variant_names == expected_names

    @pytest.mark.unit
    def test_get_performance_bounds(self, test_config):
        """Test performance bounds calculation."""
        cascade = ModelCascade(test_config)
        bounds = cascade.get_performance_bounds()

        assert "latency" in bounds
        assert "accuracy" in bounds
        assert "memory" in bounds

        for metric, (min_val, max_val) in bounds.items():
            assert min_val <= max_val


class TestAdaptiveInferenceRouter:
    """Test cases for AdaptiveInferenceRouter."""

    @pytest.mark.unit
    def test_router_initialization(self, test_config):
        """Test router initialization."""
        router = AdaptiveInferenceRouter(test_config)

        assert router.input_dim > 0
        assert router.hidden_dim == test_config.model.router["hidden_dim"]
        assert router.num_routes == 4
        assert hasattr(router, 'cascade')
        assert hasattr(router, 'feature_encoder')
        assert hasattr(router, 'route_predictor')

    @pytest.mark.unit
    def test_router_forward(self, test_config, sample_features, custom_assertions):
        """Test router forward pass."""
        router = AdaptiveInferenceRouter(test_config)

        # Add system features to match expected input dimension
        batch_size = sample_features.shape[0]
        system_features = torch.randn(batch_size, 5)
        combined_features = torch.cat([sample_features, system_features], dim=-1)

        predictions = router.forward(combined_features)

        # Check output structure
        assert isinstance(predictions, dict)
        assert "route_logits" in predictions
        assert "route_probs" in predictions
        assert "performance_preds" in predictions
        assert "values" in predictions
        assert "multi_objective_scores" in predictions
        assert "objective_weights" in predictions

        # Check shapes
        num_routes = router.num_routes
        custom_assertions.assert_tensor_shape(
            predictions["route_logits"], (batch_size, num_routes)
        )
        custom_assertions.assert_tensor_shape(
            predictions["multi_objective_scores"], (batch_size, num_routes)
        )

        # Check probabilities
        custom_assertions.assert_probability_distribution(predictions["route_probs"])
        custom_assertions.assert_probability_distribution(predictions["objective_weights"])

    @pytest.mark.unit
    def test_router_forward_with_internals(self, test_config, sample_features):
        """Test router forward pass with internal representations."""
        router = AdaptiveInferenceRouter(test_config)

        # Add system features
        batch_size = sample_features.shape[0]
        system_features = torch.randn(batch_size, 5)
        combined_features = torch.cat([sample_features, system_features], dim=-1)

        predictions = router.forward(combined_features, return_internals=True)

        assert "encoded_features" in predictions

    @pytest.mark.unit
    def test_select_route_deterministic(self, test_config, sample_features, setup_seeds, custom_assertions):
        """Test deterministic route selection."""
        router = AdaptiveInferenceRouter(test_config)

        # Add system features
        batch_size = sample_features.shape[0]
        system_features = torch.randn(batch_size, 5)
        combined_features = torch.cat([sample_features, system_features], dim=-1)

        routes, pred_info = router.select_route(combined_features, deterministic=True)

        custom_assertions.assert_tensor_shape(routes, (batch_size,))

        # Check that routes are valid
        assert torch.all(routes >= 0)
        assert torch.all(routes < router.num_routes)

        # Check prediction info
        assert isinstance(pred_info, dict)
        assert "route_probs" in pred_info
        assert "log_probs" in pred_info

    @pytest.mark.unit
    def test_select_route_stochastic(self, test_config, sample_features, setup_seeds, custom_assertions):
        """Test stochastic route selection."""
        router = AdaptiveInferenceRouter(test_config)

        # Add system features
        batch_size = sample_features.shape[0]
        system_features = torch.randn(batch_size, 5)
        combined_features = torch.cat([sample_features, system_features], dim=-1)

        routes, pred_info = router.select_route(combined_features, deterministic=False)

        custom_assertions.assert_tensor_shape(routes, (batch_size,))

        # Check that routes are valid
        assert torch.all(routes >= 0)
        assert torch.all(routes < router.num_routes)

    @pytest.mark.unit
    def test_compute_route_rewards(self, test_config, sample_features, sample_sla_constraints, custom_assertions):
        """Test route reward computation."""
        router = AdaptiveInferenceRouter(test_config)

        batch_size = sample_features.shape[0]
        num_routes = router.num_routes

        # Mock data
        performance_preds = torch.randn(batch_size, num_routes, 4)
        actual_performance = torch.rand(batch_size, 4) * torch.tensor([200, 1, 100, 1])  # realistic ranges
        route_indices = torch.randint(0, num_routes, (batch_size,))

        rewards = router.compute_route_rewards(
            performance_preds, actual_performance, sample_sla_constraints, route_indices
        )

        custom_assertions.assert_tensor_shape(rewards, (batch_size,))

    @pytest.mark.unit
    def test_compute_sla_violations(self, test_config, sample_sla_constraints, custom_assertions):
        """Test SLA violation computation."""
        router = AdaptiveInferenceRouter(test_config)

        batch_size = sample_sla_constraints.shape[0]

        # Create actual performance with some violations
        actual_performance = torch.stack([
            sample_sla_constraints[:, 0] + torch.randn(batch_size) * 20,  # Latency around SLA
            sample_sla_constraints[:, 1] + torch.randn(batch_size) * 0.05,  # Accuracy around SLA
            torch.ones(batch_size) * 50,  # Throughput
            torch.zeros(batch_size),  # Placeholder
        ], dim=1)

        violations = router._compute_sla_violations(actual_performance, sample_sla_constraints)

        custom_assertions.assert_tensor_shape(violations, (batch_size,))

        # Check that violations are binary
        assert torch.all((violations == 0) | (violations == 1))

    @pytest.mark.unit
    def test_get_model_summary(self, test_config):
        """Test model summary generation."""
        router = AdaptiveInferenceRouter(test_config)
        summary = router.get_model_summary()

        assert isinstance(summary, dict)
        assert "input_dim" in summary
        assert "hidden_dim" in summary
        assert "num_routes" in summary
        assert "total_parameters" in summary
        assert "trainable_parameters" in summary
        assert "cascade_variants" in summary
        assert "objective_weights" in summary

        # Check parameter counts are reasonable
        assert summary["total_parameters"] > 0
        assert summary["trainable_parameters"] > 0
        assert summary["trainable_parameters"] <= summary["total_parameters"]

    @pytest.mark.integration
    def test_router_end_to_end(self, test_config, sample_batch, setup_seeds):
        """Test end-to-end router functionality."""
        router = AdaptiveInferenceRouter(test_config)

        batch_size = sample_batch["features"].shape[0]

        # Add system features
        system_features = torch.randn(batch_size, 5)
        combined_features = torch.cat([
            sample_batch["features"],
            sample_batch["sla_constraints"],
            system_features
        ], dim=-1)

        # Forward pass
        predictions = router(combined_features)

        # Route selection
        routes, pred_info = router.select_route(combined_features)

        # Mock actual performance
        actual_performance = torch.rand(batch_size, 4) * torch.tensor([200, 1, 100, 1])

        # Compute rewards
        rewards = router.compute_route_rewards(
            predictions["performance_preds"],
            actual_performance,
            sample_batch["sla_constraints"],
            routes
        )

        # Check that everything worked
        assert routes.shape == (batch_size,)
        assert rewards.shape == (batch_size,)

    @pytest.mark.unit
    def test_router_gradient_flow(self, test_config, sample_features, setup_seeds):
        """Test that gradients flow properly through the router."""
        router = AdaptiveInferenceRouter(test_config)

        # Add system features
        batch_size = sample_features.shape[0]
        system_features = torch.randn(batch_size, 5)
        combined_features = torch.cat([sample_features, system_features], dim=-1)

        # Forward pass
        predictions = router(combined_features)

        # Compute a simple loss
        loss = predictions["route_logits"].sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist
        for name, param in router.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter {name}"

    @pytest.mark.unit
    def test_router_different_batch_sizes(self, test_config, setup_seeds):
        """Test router with different batch sizes."""
        router = AdaptiveInferenceRouter(test_config)

        for batch_size in [1, 4, 16, 32]:
            features = torch.randn(batch_size, router.input_dim)

            # Should work without errors
            predictions = router(features)
            routes, _ = router.select_route(features)

            assert routes.shape == (batch_size,)
            assert predictions["route_probs"].shape == (batch_size, router.num_routes)
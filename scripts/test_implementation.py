#!/usr/bin/env python
"""
Simple test to verify the MS predictor implementation works.
"""

import torch
import sys
sys.path.insert(0, '/root/ms/src')

from ms_predictor import MSPredictor, AminoAcidTokenizer, Config


def test_model_creation():
    """Test that the model can be created."""
    print("Testing model creation...")
    
    config = Config()
    model = MSPredictor(
        vocab_size=config.model.vocab_size,
        hidden_dim=config.model.hidden_dim,
        num_encoder_layers=config.model.num_encoder_layers,
        num_decoder_layers=config.model.num_decoder_layers,
        num_heads=config.model.num_heads,
        dim_feedforward=config.model.dim_feedforward,
        num_predictions=config.model.num_predictions,
        max_length=config.model.max_length,
        max_charge=config.model.max_charge,
        dropout=config.model.dropout
    )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created successfully with {num_params:,} parameters")
    
    return model


def test_forward_pass(model):
    """Test a forward pass through the model."""
    print("\nTesting forward pass...")
    
    batch_size = 2
    seq_len = 20
    
    # Create dummy input
    tokenizer = AminoAcidTokenizer()
    sequences = ['PEPTIDE', 'SEQUENCE']
    
    # Tokenize
    tokens_list = [tokenizer.encode(seq, max_length=seq_len, padding=True) for seq in sequences]
    tokens = torch.LongTensor(tokens_list)
    mask = tokens != tokenizer.pad_idx
    
    precursor_mz = torch.tensor([500.0, 600.0]) / 2000.0
    charge = torch.tensor([2, 3])
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        pred_mz, pred_intensity, pred_confidence = model(tokens, mask, precursor_mz, charge)
    
    print(f"✓ Forward pass successful")
    print(f"  Output shapes:")
    print(f"    pred_mz: {pred_mz.shape}")
    print(f"    pred_intensity: {pred_intensity.shape}")
    print(f"    pred_confidence: {pred_confidence.shape}")
    
    # Check output ranges
    assert pred_mz.min() >= 0 and pred_mz.max() <= 1, "m/z should be in [0, 1]"
    assert pred_intensity.min() >= 0 and pred_intensity.max() <= 1, "intensity should be in [0, 1]"
    assert pred_confidence.min() >= 0 and pred_confidence.max() <= 1, "confidence should be in [0, 1]"
    
    print(f"✓ Output ranges verified")
    
    return pred_mz, pred_intensity, pred_confidence


def test_loss_computation():
    """Test loss computation."""
    print("\nTesting loss computation...")
    
    from ms_predictor.loss import SetPredictionLoss, CosineSimilarityLoss
    
    batch_size = 2
    num_predictions = 100
    
    # Dummy predictions
    pred_mz = torch.rand(batch_size, num_predictions)
    pred_intensity = torch.rand(batch_size, num_predictions)
    pred_confidence = torch.rand(batch_size, num_predictions)
    
    # Dummy targets
    target_mz = torch.rand(batch_size, num_predictions)
    target_intensity = torch.rand(batch_size, num_predictions)
    target_mask = torch.ones(batch_size, num_predictions)
    target_mask[:, 50:] = 0  # Only first 50 are real targets
    
    # Test set loss
    set_loss = SetPredictionLoss()
    loss_dict = set_loss(pred_mz, pred_intensity, pred_confidence,
                         target_mz, target_intensity, target_mask)
    
    print(f"✓ Set prediction loss computed: {loss_dict['loss'].item():.4f}")
    
    # Test cosine loss
    cosine_loss = CosineSimilarityLoss(weight=0.5)
    cos_loss_val = cosine_loss(pred_mz, pred_intensity, pred_confidence,
                                target_mz, target_intensity, target_mask)
    
    print(f"✓ Cosine similarity loss computed: {cos_loss_val.item():.4f}")


def test_dummy_dataset():
    """Test dummy dataset."""
    print("\nTesting dummy dataset...")
    
    from ms_predictor.data import DummyMSDataset
    from torch.utils.data import DataLoader
    
    dataset = DummyMSDataset(num_samples=10, max_length=30, num_predictions=100, top_k=200, max_mz=2000.0)
    loader = DataLoader(dataset, batch_size=2)
    
    batch = next(iter(loader))
    
    print(f"✓ Dummy dataset created")
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Batch shapes:")
    for key, value in batch.items():
        print(f"    {key}: {value.shape}")


def test_config_system():
    """Test configuration system."""
    print("\nTesting configuration system...")
    
    from ms_predictor.training import Config
    
    # Load default config
    config = Config.from_yaml('/root/ms/configs/default_config.yaml')
    
    print(f"✓ Configuration loaded")
    print(f"  Experiment name: {config.experiment_name}")
    print(f"  Hidden dim: {config.model.hidden_dim}")
    print(f"  Num predictions: {config.model.num_predictions}")
    print(f"  Use cosine loss: {config.loss.use_cosine_loss}")
    print(f"  Cosine loss weight: {config.loss.cosine_loss_weight}")


def main():
    print("=" * 60)
    print("MS Predictor Implementation Test")
    print("=" * 60)
    
    try:
        # Test 1: Model creation
        model = test_model_creation()
        
        # Test 2: Forward pass
        test_forward_pass(model)
        
        # Test 3: Loss computation
        test_loss_computation()
        
        # Test 4: Dummy dataset
        test_dummy_dataset()
        
        # Test 5: Configuration
        test_config_system()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nThe implementation is ready to use.")
        print("Next steps:")
        print("1. Add real data when available")
        print("2. Run training: python scripts/train.py")
        print("3. Run inference: python scripts/inference.py --help")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())


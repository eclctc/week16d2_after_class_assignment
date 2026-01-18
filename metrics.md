# Experiment Metrics

## What I Tried
I trained a simple MLP model augmented with a custom `LearnedAffine` layer on the FashionMNIST dataset.
- **Model**: `MLPWithAffine` (Flatten -> Linear -> ReLU -> LearnedAffine -> Dropout -> Linear)
- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: StepLR (step_size=1, gamma=0.7)
- **Training Config**: Batch size 128, 3 epochs.
- **Seed**: 42

## What Worked
The training process was stable and converged quickly over the 3 epochs.
- **Epoch 1**: Loss 0.5280 | Accuracy 83.89%
- **Epoch 2**: Loss 0.3895 | Accuracy 85.72%
- **Epoch 3**: Loss 0.3525 | Accuracy 86.31%

The custom `LearnedAffine` layer successfully integrated into the pipeline without causing exploding gradients or shape mismatches, allowing the model to reach >86% accuracy in just 3 epochs.

## What I'd Change Next
1. **Extend Training**: 3 epochs is quite short. Extending to 10-20 epochs would likely yield higher accuracy.
2. **Ablation Study**: Compare `LearnedAffine` against a standard `nn.BatchNorm1d` or no normalization key to quantify its specific benefit.
3. **Hyperparameters**: Tune the dropout rate (currently 0.2) and the learning rate to see if we can squeeze out more performance.

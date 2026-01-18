# Reproduction Steps

To reproduce the results, ensure you have the dependencies installed and run the following command from the project root:

```bash
python main.py
```

*Note: I successfully ran this using the project's virtual environment: `.venv/bin/python main.py`*

## Configuration
The script uses a fixed seed for reproducibility:
```python
torch.manual_seed(42)
```
This ensures the random initialization of weights and data shuffling remains consistent across runs.

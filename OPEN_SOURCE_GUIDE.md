# BioAcoustica: Open Source & Contribution Guide

BioAcoustica is built by and for the bioacoustic research community. We welcome contributions, from bug fixes to new model architectures.

## 🍴 How to Fork and Adapt

If you want to use BioAcoustica for your own research project:

1. **Fork the Repository**: Use the GitHub 'Fork' button to create your own copy.
2. **Setup your environment**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Organize your data**: Follow the `/data/raw` structure. Use [Sonic Visualiser](https://www.sonicvisualiser.org/) or [Raven](https://ravensoundsoftware.com/) for annotations.
4. **Create a Config**: Store your species parameters in `/configs`.
5. **Cite**: If you use this framework in your research, please link back to this repository.

## 🤝 How to Contribute

We follow a typical open-source workflow:

### 1. Reporting Issues
Use GitHub Issues to report bugs or suggest features. Please include:
- A clear description of the issue.
- Sample data or steps to reproduce if applicable.

### 2. Pull Requests
1. **Branching**: Create a feature branch (`git checkout -b feature/cool-new-transformer`).
2. **Quality**: Ensure your code follows PEP8 standards.
3. **Docstrings**: All new functions should have clear docstrings.
4. **Tests**: If possible, include a small test script for new modules.

## 🏗️ Adding New Models

BioAcoustica uses a **Model Factory** pattern (`get_model` in `networks.py`):

1. Open `src/bioacoustica/training/networks.py`.
2. Implement your model in a new `build_<name>` function.
3. Register it in `get_model()` — add a new `elif name == "<your_model>":` branch.
4. Add the new name to the `--arch` choices in `cli/train.py`.
5. Document the model's best use-case in its docstring.

## 📜 Code of Conduct

Please be respectful and constructive in all interactions. We aim to foster a helpful environment for researchers worldwide.

# R3GAN-TensorFlow

An unofficial TensorFlow implementation of the **R3GAN** (["The GAN is dead; long live the GAN! A Modern GAN Baseline"](https://arxiv.org/abs/2501.05441)) model—a modern, simplified GAN baseline introduced in a NeurIPS 2024 paper. The official code, implemented in PyTorch by the authors of the R3GAN paper, is also publicly available in this [repo](https://github.com/brownvc/R3GAN). 

##  Project Overview

This repository recasts R3GAN in TensorFlow, providing:

- A streamlined, maintainable codebase structured for ease-of-use.
- Core modules for configuration, modeling, training, and utilities.
- Support for both notebook-based exploratory work (`train.ipynb`) and script-driven training (`train.py`).
- Reduce the dependency on external custom library (only require tensorflow).

##  Repository Structure

- `configs/`: Configuration files (e.g., hyperparameters).
- `models/`: TensorFlow model definitions for generator/discriminator.
- `training/`: Training-related code—generator/discriminator training function, loss function, and penalty.
- `utils/`: Helper functions (e.g., logging).
- `train.ipynb`: Interactive notebook for experiment tinkering.
- `train.py`: Python script for running the training sessions.
- `tb.py`: TensorBoard integration for real-time monitoring.

##  Getting Started

1. **Install dependencies**

```bash
   pip install tensorflow
```

2. **Configure Your Experiment**

 * Browse `configs/` to review and adjust settings.
 * Ensure your dataset is prepared and paths are correctly referenced.

3. **Run Training**

 * Interactive mode: open and run `train.ipynb`.
 * CLI mode:

   ```bash
   python train.py
   ```
 * Monitor with TensorBoard using:

   ```bash
   python tb.py
   ```

4. **Inference / Results**

 * Output models, samples, and logs will be saved based on configured directories.
 * Visualize training and sample outputs via TensorBoard.

## Highlights & Features

* Modular code architecture for clarity and flexibility.
* Dual interface: interactive notebook and script-based workflows.
* TensorBoard support for live monitoring.

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes and push them.
4. Open a Pull Request for review.

## License & Acknowledgments

This code is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Authors

* [**Sorawit Chokphantavee**](https://github.com/SorawitChok)
* [**Sirawit Chokphantavee**](https://github.com/SirawitC)


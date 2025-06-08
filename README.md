
<p align="center">
  <img src="assets/guidex-png.png" width="250">
</p>

# Guided Synthetic Data Generation for Zero-Shot Information Extraction *(ACL Findings 2025)*

This repository contains the official implementation of GuideX, a novel method for synthetic data generation that automatically defines domain-specific schemas, infers guidelines, and generates synthetically labeled instances for Information Extraction (IE) tasks.

## 📚 Paper

Our paper is available at [guidex.com](https://neilus03.github.io/guidex.com/). Please cite our work if you use our code or models:

```bibtex
@misc{delafuente2025guidexguidedsyntheticdata,
      title={GuideX: Guided Synthetic Data Generation for Zero-Shot Information Extraction}, 
      author={Neil De La Fuente and Oscar Sainz and Iker García-Ferrero and Eneko Agirre},
      year={2025},
      eprint={2506.00649},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.00649}, 
}
```

## 🚀 Coming Soon

We are in the process of open-sourcing the following components:
- [x] **Models**: GuideX fine-tuned models
- [x] **Dataset**: GuideX training and evaluation datasets ([link](https://huggingface.co/datasets/HiTZ/GuideX_pre-training_data))
- [x] **Code**: Complete implementation of the GuideX framework
- [ ] **Reproduction Guide**: Step-by-step instructions to reproduce our experiments

## 🛠️ Setup

The project uses Python 3.7+ and is managed through `pyproject.toml`. Key dependencies will be listed here once the reproduction guide is complete.

## Generating a GuideX dataset

To generate a GuideX dataset, you can use the `GUIDEX_pipeline.py` script. This script will generate a GuideX dataset from a given input file. We provide a small example dataset in `data/GUIDEX_gen/fineweb-edu-1k.json`, that you can use to test the pipeline.

To generate a GuideX dataset, you need to have a Hugging Face token. You can get one by logging in to your Hugging Face account and going to [this page](https://huggingface.co/settings/tokens).

1. Clone the repository
```bash
git clone https://github.com/HiTZ/GUIDEX.git
cd GUIDEX
```

2. Create a virtual environment and install the dependencies
```bash
python3 -m venv guidex_env 
source guidex_env/bin/activate.csh
pip install -r requirements.txt
```

3. Set the Hugging Face token
```bash
cd data/GUIDEX_gen
setenv HF_TOKEN "<your_huggingface_token>"
```

4. Run the pipeline
```bash
python GUIDEX_pipeline.py --input fineweb-edu-1k.json --output guidex_data.jsonl --batch-size 32
```

The Llama3.1-70B-Instruct model with which we annotate the GuideX dataset ios big, probably won't fit in your GPU. You can use the `run_GUIDEX_pipeline_1.slurm` script to run the pipeline on a cluster, our experiments were run on a cluster with 2x A100 GPUs.

```bash
sbatch run_GUIDEX_pipeline_cluster.slurm
```

5. Check the output's first 10 lines
```bash
cat guidex_out.jsonl | head -n 10
```



## Doing NER with GuideX




## 📝 License

This project is licensed under the terms of the license included in the repository.

## 🤝 Contributing

We welcome contributions! Please stay tuned for our contribution guidelines.

## 📧 Contact

For questions and feedback, please contact Neil De La Fuente at neil.de@tum.de.

---

*Note: This repository is under active development. More details about setup, usage, and reproduction will be added soon.*


> **Note:** if the notebook filename in your repo differs from `Fake_News_Paper_Code.ipynb` (e.g. after recent updates), update this path accordingly so it matches what you actually push.

## ⚙️ Setup

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
```

The notebook was developed for a Google Colab GPU runtime. It works locally as well, provided you have a CUDA-capable GPU (recommended) or are prepared to run on CPU (slower).

## 🚀 Usage

1. Open `notebooks/Fake_News_Paper_Code.ipynb` in Jupyter or Google Colab.
2. Run the cells sequentially — the notebook is organized into clearly labeled steps (environment setup, data download, preprocessing, model training, fusion, consistency scoring, and evaluation).
3. A Kaggle account/API key is required for `kagglehub` to download the Fakeddit dataset.
4. Outputs (model checkpoints, result CSVs, and figures) are saved automatically to the `outputs/` and `checkpoints/` directories.

## 📝 Notes

- Random seeds are fixed (`SEED = 42`) throughout for reproducibility.
- Class imbalance is handled via balanced class weighting in the loss function.
- The fusion classifier's category prediction (ĉ) and the weighted consistency score (S_EMCR) are two distinct, complementary outputs of the pipeline — the former is a learned six-way classifier, the latter a fixed-weight, interpretable agreement score used for the CONSISTENT/INCONSISTENT decision. See the paper for details.
- This repository accompanies a research paper on multimodal consistency reasoning for fake news detection; figure-generation cells are included for publication-quality plots.
- All reported results are obtained on a held-out validation split used during training; no independently held-out test partition is currently reserved (see the paper's Limitations section).

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

- [Fakeddit dataset](https://github.com/entitize/Fakeddit) authors
- Hugging Face `transformers` and `sentence-transformers`
- OpenAI CLIP
- EasyOCR

# Does Sentiment Improve Neural SDE Option Pricing Models?

**Stevens Institute of Technology - Senior Design Final Project**  
*Steve Linehan-Reckford, Samuel Shlyam, Ethan Itkis, Matthew Lascoe, Om Patel*

---

## Research Question

> *"What is the impact of incorporating financial news sentiment into the training process of a Neural Stochastic Differential Equation options pricing model?"*

**Objective:** Build and compare two neural SDE models - a standard baseline model and a model that receives daily sentiment data as an additional input.

**Hypothesis:** The sentiment-augmented models will achieve measurable reductions in pricing error, with the most pronounced improvements occurring during periods of high market volatility.

---

## Methodology Overview

### The Problem with Direct Training
Our initial approach trained the CNN directly on cleaned real SPX options market data. The neural network generated stochastic model parameters, which were passed through a Monte Carlo simulation to produce option prices. These were compared against real market prices and the error was backpropagated. In practice, this approach struggled because real-world data was too noisy for the model to learn from directly.

### Solution: Offline \(\rightarrow\) Online (Two-Stage) Training
Inspired by *Deep Learning Volatility* (Horvath, Muguruza, Tomas), we switched to a two-stage training process:

**Offline Training**
- Generate 50,000 synthetic parameter combinations for each of four base stochastic volatility models: **Heston**, **Bates**, **Bergomi**, and **Rough Bergomi**.
- For each parameter set, run a Monte Carlo simulation and price options across an **8 x 11 grid** of strikes and maturities.
- Convert option prices to implied volatility (IV) surfaces and train a CNN (PyTorch) to learn the mapping from IV surface $\rightarrow$ model parameters, minimizing MSE against known ground-truth parameters.
- For the sentiment-augmented variant, 50,000 synthetic sentiment series are generated using a correlated Ornstein-Uhlenbeck process, with coefficients derived from a regression on real-world sentiment data.

**Online Fine-Tuning**
- Load the offline-trained CNN weights and transition to real SPX market data.
- The model takes in real IV surfaces (and real daily sentiment scores for the augmented variant) and outputs predicted stochastic parameters.
- These parameters are passed through the pricer to generate predicted option prices, which are compared against observed market prices.
- The pricing error is backpropagated through the CNN to refine the weights.
- This is done separately for both the baseline and sentiment-augmented versions, across four market regimes: **2010-2012, 2013-2015, 2016-2019, 2020-2022**.

---

## About This Repository

The `fine_tuning/` folder is the only part of this repository that is completely ready to run out of the box. It contains all the pre-built synthetic IV surface datasets, pre-trained offline CNN weights (one `.pth` file per model), and cleaned SPX options data needed to reproduce the fine-tuning experiments. Both `Non_Sentiment_Train/` and `Sentiment_Train/` subfolders share the same code structure - the only difference is that the sentiment variant passes the daily sentiment score as an additional input to the CNN.

The `SPX_Data_Gathering/` folder contains the code we used to build the sentiment data and merge it into the SPX options files. This pipeline requires credentials to the LSEG news database and was run on an H100 GPU - it is included for reference only, as the output data is already bundled in `fine_tuning/Fine_Tuning_Files/`.

Similarly, `full_project/` contains the full offline and online training pipeline we ran to produce the results in this README. It is reference code showing how the pre-trained `.pth` weights and synthetic datasets were originally generated.

---

## Running the Fine-Tuning

### Requirements
```bash
pip install torch pandas numpy scipy scikit-learn tqdm matplotlib
```

### Baseline (No Sentiment)
```bash
cd fine_tuning/Non_Sentiment_Train
python launch.py
```

### Sentiment-Augmented
```bash
cd fine_tuning/Sentiment_Train
python launch.py
```

`launch.py` auto-detects available GPUs and distributes models across them (1 GPU runs everything sequentially; 4 GPUs assign one model per GPU). Training runs all four models (Heston, Bates, Bergomi, rBergomi) across all four date regimes (2010-2012, 2013-2015, 2016-2019, 2020-2022) for 750 epochs each.

Output models and plots are saved to `fine_tuning_output/`.

---

## Data

### SPX Options Data
~20 years of SPX options data sourced from the **Hanlon Financial Systems Lab** at Stevens. The raw data is cleaned, filtered, and organized by trade date. For each day, option prices are mapped into an **8 × 11 grid** across log-moneyness strikes and maturities, giving a consistent daily market surface.

### Sentiment Data
Daily sentiment is constructed from **LSEG machine-readable financial news** (2010-2024):
1. Articles mentioning the S&P 500, SPX, or SPY are queried from the database.
2. Each article body is passed through **FinBERT** (`yiyanghkust/finbert-tone`) to obtain positive/negative/neutral probabilities. Long articles are split into 400-character chunks and averaged.
3. The final daily sentiment score = `P(positive) - P(negative)`, giving a value in $[-1, +1]$.
4. This daily score is merged onto the SPX options data by trade date.

The sentiment construction pipeline lives in `SPX_Data_Gathering/` - this code requires credentials to the LSEG database and was run on an H100 GPU. The resulting merged CSVs are already included in `fine_tuning/Fine_Tuning_Files/SPX_Data/`.

---

## Sample Results

### Training Convergence (Heston, 2013-2015)

![Training History](full_project_output/base_output/plots/Heston/2013-2015/training_history.png)

*Huber loss and relative error decrease steadily across 750 epochs of online fine-tuning.*

---

### Predicted vs. Actual Prices (Heston, 2013-2015)

![Parity and Metrics](full_project_output/base_output/plots/Heston/2013-2015/parity_and_metrics.png)

*Left: Actual vs. Predicted price scatter (density). Points cluster tightly along the perfect-fit diagonal. Right: Distribution of relative errors and summary metrics.*

---

### Volatility Smile & Term Structure Fit (Heston, 2013-2015 - Best Day)

![Smile Term Best](full_project_output/base_output/plots/Heston/2013-2015/smile_term_best.png)

*Left: Smile slice at $\tau = 1.20$ - the model closely tracks the actual skew across log-moneyness. Right: ATM term structure fit.*

---

### IV Surface Heatmaps (Heston, 2013-2015 - Best Day)

![Surface Heatmap Best](full_project_output/base_output/plots/Heston/2013-2015/surface_heatmap_best.png)

*From left to right: Input IV surface, Actual prices, Predicted prices, Prediction error. The error surface is near-zero across most of the grid.*

---

### All Models vs. Actual (2013–2015)

![All Models Smile](full_project_output/base_output/plots/smiles/2013-2015/all_models_smile.png)

*Volatility smile and ATM term structure for all eight model variants (four base models x sentiment/non-sentiment) compared against the actual market surface on a representative day.*

---

## Results Summary

We evaluated each model across four market regimes using **Huber loss**, **MAE**, **RMSE**, and **Relative error**.

### 2010-2012
| Model | Huber (NS) | RMSE (NS) | Huber (S) | RMSE (S) |
|-------|-----------|----------|----------|---------|
| Heston | 0.000671 | 0.042124 | 0.000721 | 0.043985 |
| Bates | 0.000724 | 0.042987 | 0.000907 | 0.049452 |
| Bergomi | 0.000778 | 0.050707 | 0.000794 | 0.051469 |
| rBergomi | 0.000805 | 0.051575 | 0.000866 | 0.053542 |

Sentiment did not improve performance in this early period. Non-sentiment Heston had the lowest Huber loss and RMSE overall.

### 2013-2015
| Model | Huber (NS) | RMSE (NS) | Huber (S) | RMSE (S) |
|-------|-----------|----------|----------|---------|
| Heston | 0.000762 | 0.044849 | **0.000661** | **0.041126** |
| Bates | 0.001052 | 0.054346 | **0.000767** | **0.044166** |
| Bergomi | 0.000962 | 0.057826 | 0.000973 | 0.058134 |
| rBergomi | 0.000910 | 0.056074 | 0.000906 | 0.056892 |

**Sentiment's biggest impact** - Heston and Bates improved across all metrics. This quieter market period appears to be where sentiment adds the most signal.

### 2016-2019
| Model | Huber (NS) | RMSE (NS) | Huber (S) | RMSE (S) |
|-------|-----------|----------|----------|---------|
| Heston | 0.000522 | 0.035796 | **0.000501** | **0.034969** |
| Bates | **0.000495** | **0.034243** | 0.000571 | 0.036983 |
| Bergomi | 0.000662 | 0.046811 | **0.000631** | **0.045600** |
| rBergomi | **0.000638** | **0.045969** | 0.000717 | 0.048499 |

Mixed results - sentiment helped Heston and Bergomi but hurt Bates and rBergomi.

### 2020-2022
| Model | Huber (NS) | RMSE (NS) | Huber (S) | RMSE (S) |
|-------|-----------|----------|----------|---------|
| Heston | 0.000721 | 0.043985 | **0.000552** | **0.040454** |
| Bates | **0.000559** | 0.041807 | 0.000616 | 0.043676 |
| Bergomi | **0.001224** | **0.077161** | 0.001258 | 0.078121 |

Non-sentiment models generally stronger in the volatile COVID-era period. Bergomi struggled the most, likely because the sudden volatility regime was hard to capture with a smooth variance structure.

---

## Conclusions

- **Sentiment had a small, inconsistent impact** on pricing accuracy across regimes.
- **The largest improvement appeared in 2013-2015**, the calmest of the four regimes - suggesting sentiment is more informative when market noise is lower. When markets are chaotic, sentiment adds noise; when calmer, the signal stands out.
- **Heston and Bates** benefited most consistently from sentiment augmentation.
- **Next steps:** Longer training runs, more market regimes, and exploration of other model families that may respond more consistently to sentiment.

---

## Citation

Architecture inspired by:
> Blanka Horvath, Aitor Muguruza, Mehdi Tomas. *Deep Learning Volatility.* Quantitative Finance, 2021.

Sentiment model:
> `yiyanghkust/finbert-tone` - FinBERT fine-tuned on financial tone classification.

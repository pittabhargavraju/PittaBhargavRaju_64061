# Assignment 2 — IMDB Sentiment Neural Network Hyperparameter Study

## 1. Assignment Overview

The goal of this assignment was to explore and extend our first neural network model using the IMDB movie review dataset.  
We were asked to:

1. Change the **number of hidden layers** (use 1 or 3 instead of 2) and compare how it affects validation and test accuracy.  
2. Change the **number of hidden units** (e.g., 32, 64) to see the effect on model performance.  
3. Try different **loss functions** — `binary_crossentropy` vs `mse`.  
4. Try different **activation functions** — `relu` vs `tanh`.  
5. Use **regularization techniques** like dropout or L2 to improve validation accuracy.

We also had to summarize the results clearly (tables or graphs) and explain how different approaches affected the model’s performance.

---

## 2. What I Did (Step by Step)

### a. Data Preparation
- Loaded the **IMDB dataset** using Keras (top 10,000 most frequent words).  
- Converted reviews to **multi-hot encoded vectors** for a simple baseline representation.  
- Split data into **80% training / 20% validation**, using a fixed random seed for reproducibility.

### b. Model Building From Scratch
- Wrote a **`make_mlp` function** to dynamically build models with:
  - Variable number of hidden layers (1, 2, 3)  
  - Variable number of units (32 or 64)  
  - Choice of activation (`relu` or `tanh`)  
  - Optional dropout and/or L2 regularization  
  - Variable loss function (`binary_crossentropy` or `mse`)  

### c. Training & Evaluation
- Used a **grid search** over all required hyperparameter combinations:
  - Layers × Units × Activations × Loss × Regularization.  
- Trained up to 20 epochs with **EarlyStopping** (patience=3) on validation accuracy.  
- Saved best models with **ModelCheckpoint**.  
- Evaluated on both validation and test sets with restored best weights.  
- Logged all results (val/test accuracy, epochs, model names) into a table.

### d. Regularization
- Tried **dropout values of 0.3 and 0.5** to reduce overfitting.  
- Tried **L2 regularization (1e-4)** to improve stability and generalization.  

---

## 3. Analysis & Results

I exported results to CSV and plotted the top-15 models by validation and test accuracy.  
I also computed averages grouped by each factor (depth, width, activation, loss, regularization) to analyze trends.

### Key Findings
- **Depth:** Going from 1 → 2 layers improved validation accuracy. Adding a 3rd layer helped only with regularization.  
- **Width:** Increasing units from 32 to 64 gave consistent small gains.  
- **Activation:** ReLU performed slightly better than tanh overall.  
- **Loss:** Binary crossentropy generally outperformed MSE, though MSE had some good runs due to seed variance.  
- **Regularization:** Dropout and L2 helped control overfitting and improved validation accuracy on deeper models.

Best model achieved roughly:
- **Validation Accuracy:** ~0.88  
- **Test Accuracy:** ~0.86

---

## 4. Why My Approach Worked

- I built a **flexible experiment pipeline** from scratch to test all required variations systematically.  
- Used **early stopping** and **model checkpoints** for fair comparisons.  
- Applied **dropout and L2 regularization** to improve generalization.  
- Summarized results with tables and plots, clearly showing how each parameter affected performance.

---

## 5. How to Reproduce

### Environment
- Python 3.10+
- TensorFlow 2.x
- NumPy, Pandas, Matplotlib, scikit-learn

### Run Steps
1. Install dependencies:
   ```bash
   pip install tensorflow numpy pandas scikit-learn matplotlib
Run the notebook cells top to bottom.

All outputs will be saved in an artifacts_<timestamp>/ folder:

results_assignment2.csv

leaderboard.json

best_config.json

Top-15 validation/test accuracy plots

Saved model checkpoints


6. Conclusion
By systematically experimenting with network architecture, loss functions, activations, and regularization, I was able to understand how each change affects model performance.
My approach was successful because it followed a structured process, reproduced results reliably, and achieved high validation and test accuracy while meeting all assignment requirements.

Thank You :)

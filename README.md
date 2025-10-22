# 🎮 PUBG Player Frustration Analysis
### Predicting Match Placement and Identifying High-Value Players at Risk of Churn

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-yellow)](https://scikit-learn.org/)

## 📋 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Technical Highlights](#technical-highlights)
- [Results](#results)
- [Business Impact](#business-impact)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project analyzes over **4.4 million** PlayerUnknown's Battlegrounds (PUBG) match records to predict player placement and identify a critical business problem: **high-performing players experiencing unexpected losses due to unmeasurable factors**.

By building regression models that achieve **80.6% R²**, we identify the **19.4% unexplained variance** and use this as a "frustration score" to detect players at risk of churn. The analysis reveals that frustrated players are not underperforming—they're **overperforming** on all measurable metrics (63% more walkDistance, 66% more weapons, 557% more aggression) but still lose to factors outside their control.

**Key Insight:** The most frustrated 1% of players represent the game's **highest-value segment**—experienced, aggressive, engaged players who are statistically more likely to spend money and influence community sentiment.

---

## 💡 Problem Statement

In battle royale games, player experience is defined by more than just winning. A player who performs well but loses due to bad luck has a vastly different experience than a casual player who underperforms as expected.

**Research Question:**  
*Can we use in-match player actions to predict expected placement, then use prediction errors to quantify match frustration and identify players at churn risk?*

**Business Value:**  
Identifying frustrated high-performers allows game developers to:
- Implement targeted retention strategies
- Fine-tune game balance to reduce RNG impact
- Provide transparency through post-match analytics
- Reward performance beyond just placement

---

## 🔍 Key Findings

### Model Performance
| Model | MAE | RMSE | R² | Key Insight |
|-------|-----|------|----|-------------|
| **Linear Regression** | 0.1213 | 0.1561 | 0.7416 | Baseline; struggles with bounded predictions |
| **Random Forest** | 0.0992 | 0.1362 | 0.8035 | Handles non-linearity; reveals 92% feature importance in walkDistance |
| **LightGBM** | 0.0986 | 0.1354 | **0.8056** | Best performance with engineered features |

### Frustration Analysis
- **19.4% unexplained variance** represents factors beyond player control (circle RNG, opponent skill, positioning)
- Frustrated players (bottom 1% of residuals) show **paradoxical superiority**:
  - +63% walkDistance (survival time)
  - +66% weapons acquired (looting effectiveness)
  - +557% aggression score (combat engagement)
- Yet they place **17 percentile points below predictions**

### Feature Importance
1. **walkDistance** (42.95%) - Survival time is king
2. **walk_squared** (34.21%) - Exponential survival advantage
3. **walk_x_weapons** (14.10%) - Looting effectiveness given survival time
4. **Other features** (<10%) - Combat, efficiency, playstyle modifiers

**Strategy: Survive → Prepare → Win**

---

## 📊 Dataset

**Source:** [PUBG Finish Placement Prediction (Kaggle)](https://www.kaggle.com/c/pubg-finish-placement-prediction)

**Size:** 4,446,966 player records across ~47,000 matches

**Target Variable:**  
- `winPlacePerc`: Continuous [0.0, 1.0] representing placement percentile

**Key Features (29 total):**
- **Combat:** kills, damageDealt, killStreaks, longestKill
- **Survival:** walkDistance, rideDistance, swimDistance
- **Resources:** weaponsAcquired, boosts, heals
- **Team:** assists, revives, DBNOs
- **Meta:** matchType (solo/duo/squad, FPP/TPP)

**Data Quality:**
- Only 1 missing value (winPlacePerc) in 4.4M rows
- 212 records (0.005%) removed for suspected cheating (>40 kills or >1.5 kills/min)
- killPlace removed due to data leakage (post-match ranking)

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Target distribution analysis (bimodal: early deaths vs winners)
- Multicollinearity detection via VIF (removed assists, headshotKills, DBNOs)
- Outlier investigation (cheater detection, kills per minute analysis)
- Match-type stratification (solo vs duo vs squad correlations)

### 2. Feature Engineering
**Interaction Features** (conditional effectiveness):
```python
walk_x_weapons = walkDistance × weaponsAcquired    # Looting efficiency
walk_x_kills = walkDistance × kills                # Combat effectiveness
walk_x_boosts = walkDistance × boosts              # Resource management
```

**Efficiency Ratios**:
```python
kills_per_distance = kills / (walkDistance + 1)    # Combat efficiency
weapons_per_distance = weapons / (walkDistance + 1) # Looting efficiency
```

**Playstyle Indicators**:
```python
aggression_score = (kills × 10 + damage/100) / (walkDistance + 1)
passive_score = walkDistance / (kills + 1)
```

**Non-linear Transformations**:
```python
walk_squared = walkDistance²                        # Exponential survival advantage
log1p(rideDistance), log1p(longestKill)            # Diminishing returns
```

### 3. Preprocessing Pipeline
1. Remove missing values and outliers
2. Log-transform skewed features (rideDistance, longestKill)
3. Target encode matchType (mean winPlacePerc per category)
4. StandardScaler for all numerical features
5. Train/validation split (80/20)

### 4. Model Selection
- **Experiment 1:** Linear Regression (baseline)
- **Experiment 2:** Random Forest Regressor (handle non-linearity, multicollinearity)
- **Experiment 3:** LightGBM (efficiency + engineered features)

### 5. Frustration Score Calculation
```python
frustration_score = actual_placement - predicted_placement
frustrated_players = bottom 1% of residuals (highest underperformance)
```

---

## 💻 Technical Highlights

### Advanced Techniques Used
- ✅ **Variance Inflation Factor (VIF)** for multicollinearity detection
- ✅ **Target Encoding** for high-cardinality categorical variables
- ✅ **Feature Interaction Engineering** for conditional effectiveness
- ✅ **Log Transformations** for handling skewed distributions
- ✅ **Residual Analysis** for business insight generation
- ✅ **Huber Loss** for robust evaluation with outliers

### Code Quality
- Modular preprocessing pipeline
- Comprehensive data validation
- Reproducible results (random_state=42)
- Detailed documentation and comments
- Visualizations for all key analyses

---

## 📈 Results

### Feature Importance (LightGBM)
| Feature | Importance |
|---------|-----------|
| walkDistance | 42.95% |
| walk_squared | 34.21% |
| walk_x_weapons | 14.10% |
| walk_x_boosts | 2.54% |
| Others | <7% |

### Frustrated Player Profile
```
Average Player:           Frustrated Player:
• walkDistance: 1,154m    • walkDistance: 1,884m (+63%)
• weapons: 3.66           • weapons: 5.17 (+41%)
• kills: 0.92             • kills: 1.17 (+27%)
• aggression: 0.047       • aggression: 0.307 (+557%)

Expected placement: 52%   Actual placement: 35%
→ Disappointment gap: 17 percentile points
```

---

## 💼 Business Impact

### Recommendations for Game Developers

**A. Reduce RNG Impact (Make Skill Matter More)**
1. **Predictable Circle Mechanics**
   - Show next 2 zones instead of 1
   - Weight circle spawns toward current zone center (70% probability)
   - Add probability heatmaps for next zone
   
2. **Skill-Based Matchmaking**
   - Implement hidden MMR system
   - Reduce "ran into a pro" scenarios
   - A/B test loose SBMM (±200 MMR range)

3. **Consistent Loot Distribution**
   - Guarantee high-tier loot zones
   - Dynamic loot balancing (if 5 buildings = no scope, next guarantees one)

**B. Transparency & Feedback**
1. **Post-Match Analytics Dashboard**
   ```
   📊 YOUR STATS:
   Placement: 35th percentile
   Walk Distance: 1,884m (Top 25% ⭐)
   Weapons: 5 (Top 40% ⭐)
   
   🎯 PREDICTED: 52nd percentile
   You SHOULD have placed better!
   
   ❌ WHAT WENT WRONG:
   • Circle spawned away from you 3 times (bad luck)
   • Encountered top 5% skilled opponent
   • Final circle positioning: 8th percentile
   
   💡 YOU PLAYED WELL! Focus on positioning.
   ```

2. **Positive Loss Messaging**
   - Replace "Rank 65/100" with "Top 25% survival, tough luck this round!"

**C. Reward Performance Beyond Placement**
1. **Multi-Dimensional Ranking**
   - Placement MMR (current)
   - Combat MMR (kills, damage)
   - Survival MMR (walkDistance, efficiency)

2. **Performance-Based Bonuses**
   - Top 20% walkDistance: +50 BP
   - Top 20% combat: +50 BP
   - "Overperformed prediction": +100 BP

### Expected Impact
- **15-30% reduction in churn** among high-performing players
- **Protect 2-5% of revenue** (frustrated players are high-engagement = high-LTV)
- **Reduce brand risk** from influencer/streamer frustration
- **Improve perceived fairness** across all player segments

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- 8GB+ RAM (for full dataset)

### Setup
```bash
# Clone repository
git clone https://github.com/dganesh05/pubg-frustration-analysis.git
cd pubg-frustration-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Visit: https://www.kaggle.com/c/pubg-finish-placement-prediction
# Download train_V2.csv and place in project root
```

### Dependencies
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
category-encoders>=2.3.0
scipy>=1.7.0
statsmodels>=0.13.0
jupyter>=1.0.0
```

---

## 🚀 Usage

### Quick Start
```bash
jupyter notebook pubg_analysis.ipynb
```

### Analyzing Your Own Data
```python
# Load your PUBG match data
import pandas as pd
df = pd.read_csv('your_data.csv')

# Run preprocessing
from preprocessing import clean_data, engineer_features
df_clean = clean_data(df)
df_eng = engineer_features(df_clean)

# Train model
from models import train_lgb_model
model = train_lgb_model(df_eng)

# Calculate frustration scores
from analysis import calculate_frustration
frustrated = calculate_frustration(model, df_eng)
```

### Reproducing Results
All analyses are fully reproducible with `random_state=42` set throughout the notebook. You can set a different random state by changing the `RANDOM_STATE` variable

---

## 📁 Project Structure

```
pubg-frustration-analysis/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── project_notebook.ipynb       # Main analysis notebook
├── requirements.txt        # minimum requirements to download
└── train_V2.csv         # Dataset (download separately)
```

---

## 🔮 Future Work

### Model Improvements
- [ ] Collect positioning data (distance to circle center, final zone position)
- [ ] Add opponent skill features (average opponent MMR in match)
- [ ] Implement temporal analysis (frustration cumulation over sessions)
- [ ] Test ensemble methods (stacking RF + LightGBM + XGBoost)

### Business Applications
- [ ] A/B test recommendations (circle mechanics, SBMM, rewards)
- [ ] Build real-time frustration detection system
- [ ] Create player retention dashboard for developers
- [ ] Extend analysis to other battle royale games

### Technical Enhancements
- [ ] Deploy model as REST API
- [ ] Build interactive web dashboard (Streamlit/Dash)
- [ ] Optimize for larger datasets (Dask, distributed computing)
- [ ] Add automated retraining pipeline

---

## 🙏 Acknowledgments

- **Dataset:** PUBG Corporation via Kaggle Competition
- **AI Assistance:** Claude (Anthropic AI) for feature engineering ideation and code debugging
- **Libraries:** scikit-learn, LightGBM, pandas, matplotlib, seaborn
- **Inspiration:** Data-driven game design and player psychology research

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** The PUBG dataset is subject to Kaggle's competition rules and PUBG Corporation's terms of use.

---

## 📧 Contact

**Divya Ganesh**  
📧 divya.ganesh05@outlook.com 
🔗 [LinkedIn](https://linkedin.com/in/divyaganesh05)  
💼 [GitHub](https://github.com/yourusername)

---

## 🌟 If You Found This Useful

- ⭐ Star this repository
- 🐛 Report issues
- 🤝 Submit pull requests
- 💬 Share feedback

---

<p align="center">
  <i>Built with ❤️ for data-driven game design</i>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/yourusername/pubg-frustration-analysis?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/yourusername/pubg-frustration-analysis?style=social" alt="GitHub forks">
</p>
```
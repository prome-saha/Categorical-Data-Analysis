import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols 
from scipy.stats import chi2_contingency, fisher_exact, f_oneway, shapiro, levene, nbinom
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, recall_score
from statsmodels.discrete.count_model import ZeroInflatedPoisson
from statsmodels.base.model import GenericLikelihoodModel

# ==============================================================
# PROBLEM 1: LUNG DISEASE DATASET ANALYSIS
# ==============================================================

df = pd.read_csv('lung_disease.csv')
print(df.head())

# EDA Questions
print(df['Age'].describe())
print('Skewness =', df['Age'].skew())
print(df['Smoking'].value_counts(normalize=True) * 100)
print(df['Income'].value_counts())
print((df['Pollution'] == 'High').mean() * 100)
print((df['Lung Disease'] == 'Yes').mean() * 100)

# Association Tests
crosstab_smoking = pd.table(df['Smoking'], df['Lung Disease'])
chi2, p, dof, expected = chi2_contingency(crosstab_smoking)
print('Chi-square =', chi2, 'p-value =', p)

oddsratio, p_fisher = fisher_exact([[155, 52], [109, 184]])
print('Odds Ratio =', oddsratio)

# Logistic Regression
df2 = df.copy()
df2['Smoking'] = df2['Smoking'].map({'Yes': 1, 'No': 0})
df2['Pollution'] = df2['Pollution'].map({'High': 1, 'Low': 0})
df2['Lung Disease'] = df2['Lung Disease'].map({'Yes': 1, 'No': 0})

model1 = smf.logit('Q("Lung Disease") ~ Smoking + Age + Pollution + C(Income)', data=df2).fit()
print(model1.summary())

# Performance Metrics
pred_prob = model1.predict(df2)
print('AUC =', roc_auc_score(df2['Lung Disease'], pred_prob))
pred_class = (pred_prob > 0.5).astype(int)
print(confusion_matrix(df2['Lung Disease'], pred_class))
print('Accuracy =', accuracy_score(df2['Lung Disease'], pred_class))

# ==============================================================
# PROBLEM 2: ONE-WAY ANOVA (EXERCISE PROGRAMS)
# ==============================================================

np.random.seed(42)
prog_A = np.random.normal(5, 1.5, 30)
prog_B = np.random.normal(7, 1.5, 30)
prog_C = np.random.normal(5.5, 1.5, 30)

df_anova = pd.DataFrame({
    'WeightLoss': np.concatenate([prog_A, prog_B, prog_C]),
    'Program': ['A']*30 + ['B']*30 + ['C']*30
})

model_anova = ols('WeightLoss ~ Program', data=df_anova).fit()
print(sm.stats.anova_lm(model_anova, typ=2))

# Assumptions
print('Shapiro p-value:', shapiro(model_anova.resid)[1])
print('Levene p-value:', levene(prog_A, prog_B, prog_C)[1])

# ==============================================================
# PROBLEM 3: CHI-SQUARE ANALYSIS (CHILD ANEMIA)
# ==============================================================

df_anemia = pd.read_csv('children anemia.csv')
clean_df = df_anemia.dropna(subset=['Wealth index combined', 'Anemia level'])
ct = pd.crosstab(clean_df['Wealth index combined'], clean_df['Anemia level'])
chi2_an, p_an, dof_an, exp_an = chi2_contingency(ct)
print(f"Chi-Square: {chi2_an}, P-value: {p_an}")

# ==============================================================
# PROBLEM 4: FISHER'S EXACT TEST (DRUG TREATMENTS)
# ==============================================================

table = np.array([[40, 10], [10, 40], [25, 25]])
_, p_glob, _, _ = chi2_contingency(table)
print(f"Global P-value: {p_glob}")

p_vals = [fisher_exact(table[[0,1]])[1], fisher_exact(table[[0,2]])[1], fisher_exact(table[[1,2]])[1]]
reject, p_adj, _, _ = multipletests(p_vals, method='bonferroni')
print("Adjusted P-values:", p_adj)

# ==============================================================
# PROBLEM 5: LOGISTIC REGRESSION (ADMISSION)
# ==============================================================

df_admit = pd.read_csv('binary.csv')
print(smf.logit('admit ~ gre + gpa + C(rank)', data=df_admit).fit().summary())

# ==============================================================
# PROBLEM 6: POISSON REGRESSION (AWARDS)
# ==============================================================

df_awards = pd.read_csv('poisson_sim.csv')
model_p = smf.glm('num_awards ~ C(prog) + math', data=df_awards, family=sm.families.Poisson()).fit()
print(model_p.summary())

# ==============================================================
# PROBLEM 7: NEGATIVE BINOMIAL REGRESSION (ABSENCES)
# ==============================================================

df_abs = pd.read_stata('nb_data.dta')
model_nb = smf.glm('daysabs ~ C(prog) + math', data=df_abs, family=sm.families.NegativeBinomial()).fit()
print(model_nb.summary())

# ==============================================================
# PROBLEM 8: ZERO-INFLATED POISSON (FISH DATA)
# ==============================================================

df_fish = pd.read_csv('fish.csv')
X = sm.add_constant(df_fish[['camper', 'persons', 'child']])
X_infl = sm.add_constant(df_fish[['child', 'persons']])
model_zip = ZeroInflatedPoisson(df_fish['count'], X, exog_infl=X_infl).fit()
print(model_zip.summary())

# ==============================================================
# PROBLEM 9: ZIP ANALYSIS WITH IRR (FISH DATA)
# ==============================================================

fish_url = "https://stats.idre.ucla.edu/stat/data/fish.csv"
df_f = pd.read_csv(fish_url)
zip_res = sm.ZeroInflatedPoisson.from_formula("count ~ persons + child + camper", df_f, 
                                             exog_infl=df_f[['persons', 'child', 'camper']], 
                                             inflation='logit').fit(method='bfgs')
print(zip_res.summary())
print("Incident Rate Ratios:\n", np.exp(zip_res.params))

# ==============================================================
# PROBLEM 10: ZERO-TRUNCATED POISSON (HOSPITAL STAY)
# ==============================================================

df_ztp = pd.read_csv('ztp_temp.csv')
model_ztp = smf.glm('stay ~ age + hmo + died', data=df_ztp, family=sm.families.NegativeBinomial()).fit()
print(model_ztp.summary())

# ==============================================================
# PROBLEM 11: ZERO-TRUNCATED NEGATIVE BINOMIAL
# ==============================================================

class ZTNB(GenericLikelihoodModel):
    def loglike(self, params):
        beta = params[:-1]
        alpha = params[-1]
        if alpha <= 0: return -np.inf
        mu = np.exp(np.dot(self.exog, beta))
        size = 1 / alpha
        prob = size / (size + mu)
        ll_nb = nbinom.logpmf(self.endog, size, prob)
        p0 = (1 + alpha * mu) ** (-1/alpha)
        return np.sum(ll_nb - np.log(1 - p0))

X_ztnb = sm.add_constant(df_ztp[['age', 'hmo', 'died']])
model_ztnb = ZTNB(df_ztp['stay'], X_ztnb).fit()
print(model_ztnb.summary())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV, Lasso


# קריאת הנתונים ומחיקת ערכים חסרים
df = pd.read_csv("UTF-8NBA_Data.csv") #"UTF-8NBA_Data.csv"
df = df.dropna()

# משתנים תלויים ועצמאיים
y = df['salary']
X = df.drop(['salary'], axis=1)

# גרף התפלגות שכר רגיל
plt.figure(figsize=(8,6))
sns.histplot(df['salary'], kde=True, bins=30)
plt.title('Distribution of Player Salaries')
plt.xlabel('Salary')
plt.ylabel('Number of Players')
plt.grid(True)
plt.show()

# טרנספורמציית לוג על salary
df['log_salary'] = np.log(df['salary'])

# גרף התפלגות שכר אחרי לוג
plt.figure(figsize=(8,6))
sns.histplot(df['log_salary'], kde=True, bins=30)
plt.title('Distribution of Log-Transformed Player Salaries')
plt.xlabel('Log Salary')
plt.ylabel('Number of Players')
plt.grid(True)
plt.show()

# ---------------  גרף קורלציה מול log_salary בלבד ---------------

# חישוב מטריצת קורלציה רק על משתנים מספריים
corr_matrix = df.select_dtypes(include=[np.number]).corr()

# קורלציה של כל משתנה מול log_salary
corr_with_log_salary = corr_matrix['log_salary'].drop('log_salary').sort_values(ascending=False)

# הצגת הקורלציה מול log_salary כגרף (בר גרף)
plt.figure(figsize=(10,6))
corr_with_log_salary.plot(kind='barh')
plt.title('Correlation of Variables with log_salary')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Variables')
plt.grid(True)
plt.gca().invert_yaxis()  # להפוך את הסדר כדי שהגבוהים יופיעו למעלה
plt.show()

# --------------- גרף קורלציה בין כל המשתנים עצמם ---------------

# קורלציה בין משתנים עצמם (רק משתנים מספריים)
features_only = df.drop(['salary', 'log_salary'], axis=1)
features_only = features_only.select_dtypes(include=[np.number])

corr_features = features_only.corr()

# הצגת מטריצת קורלציה כ-Heatmap
plt.figure(figsize=(15,12))
sns.heatmap(corr_features, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Between Features')
plt.show()



# בחירת משתנים לאחר סינון
selected_columns = [
    'log_salary',
    'MIN', 'FGM', '3PM' ,'FTM', 'FT%' ,'REB' ,'AST', 'STL', 'BLK', 'TO'
]

# יצירת דאטה רק עם העמודות הרלוונטיות
df_model = df[selected_columns]

# הגדרת משתנה תלוי ובלתי תלויים לרגרסיה
X_model = df_model.drop('log_salary', axis=1).astype('float64')
y_model = df_model['log_salary']


# יצירת גרפים לבדיקת קשר לינארי
num_features = X_model.shape[1]
cols = 4  # מספר עמודות בתצוגה
rows = (num_features // cols) + int(num_features % cols > 0)

fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
axes = axes.flatten()

for i, col in enumerate(X_model.columns):
    sns.scatterplot(x=X_model[col], y=y_model, ax=axes[i])
    axes[i].set_title(f'{col} vs Log Salary')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Log Salary')
    axes[i].grid(True)

# הסתרת גרפים ריקים
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# חלוקה ל־Train ו־Test
X_train, X_test, y_train, y_test = train_test_split(X_model, y_model, test_size=0.2, random_state=42)


# יצירת מתקן הסטנדרטיזציה והתאמתו לנתוני סט האימון
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# המרת התוצאות בחזרה ל-DataFrame עם שמות העמודות המקוריות
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


#-------------------ols------------------------

# פונקציה שמתאמנת על תת-קבוצה של משתנים ומחשבת את MSE על סט הבדיקה
def processSubset(feature_set, X_train, y_train, X_test, y_test):
    # הוספת קבוע (Intercept) למודל
    X_train_subset = sm.add_constant(X_train[feature_set])
    X_test_subset = sm.add_constant(X_test[feature_set])
    # אימון מודל OLS על סט האימון עם המשתנים הנתונים
    model = sm.OLS(y_train, X_train_subset).fit()
    # חיזוי על סט הבדיקה וחישוב MSE
    predictions = model.predict(X_test_subset)
    mse = np.mean((predictions - y_test) ** 2)
    return model, mse

# פונקציה שמבצעת צעד אחד של Forward Selection: בודקת הוספת כל משתנה שנותר ובוחרת את המשתנה שמשפר את ה-MSE יותר מכל
def forward(current_predictors, X_train, y_train, X_test, y_test):
    remaining_predictors = [p for p in X_train.columns if p not in current_predictors]
    best_mse = float("inf")
    best_model = None
    best_feature = None
    # נסה להוסיף כל משתנה שטרם במודל
    for feature in remaining_predictors:
        trial_features = current_predictors + [feature]
        model, mse = processSubset(trial_features, X_train, y_train, X_test, y_test)
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_feature = feature
    return best_feature, best_model, best_mse

# ביצוע הבחירה הקדימה
selected_predictors = []    # רשימת משתנים שנבחרו עד כה
results = {'num_features': [], 'features': [], 'test_MSE': [], 'model': []}

for k in range(1, len(X_train_scaled.columns) + 1):
    # הוספת המשתנה הטוב ביותר בשלב הבא
    feat, mod, mse = forward(selected_predictors, X_train_scaled, y_train, X_test_scaled, y_test)
    selected_predictors.append(feat)
    # שמירת התוצאות של שלב זה
    results['num_features'].append(k)
    results['features'].append(list(selected_predictors))
    results['test_MSE'].append(mse)
    results['model'].append(mod)

# המרת התוצאות ל-DataFrame לנוחות
forward_results = pd.DataFrame(results)
print("משתנים בכל שלב של הבחירה הקדימה:")
print(forward_results[['num_features', 'features', 'test_MSE']])
# זיהוי מספר המשתנים האופטימלי לפי MSE בסט הבדיקה
optimal_idx = forward_results['test_MSE'].idxmin()
optimal_num_features = forward_results.loc[optimal_idx, 'num_features']
optimal_features = forward_results.loc[optimal_idx, 'features']
print(f"\nה-MSE הנמוך ביותר על סט הבדיקה הושג עם {optimal_num_features} משתנים: {optimal_features}")



# הוספת חישובי AIC, BIC ו-R^2 מתוקן לכל מודל (מבוסס על סט האימון) בתוצאות הבחירה הקדימה
forward_results['AIC'] = forward_results['model'].apply(lambda m: m.aic)
forward_results['BIC'] = forward_results['model'].apply(lambda m: m.bic)
forward_results['Adj_R2'] = forward_results['model'].apply(lambda m: m.rsquared_adj)

# הדפסת מדדי AIC, BIC ו-R^2 מתוקן לכל מודל לפי שלב (מספר המשתנים)
print("\nמדדים על סט האימון בכל שלב:")
print(forward_results[['num_features', 'AIC', 'BIC', 'Adj_R2']].round(3))

# שליפת המדדים כרשימות עבור השרטוט
num_features = forward_results['num_features']
test_mse = forward_results['test_MSE']
aic = forward_results['AIC']
bic = forward_results['BIC']
adj_r2 = forward_results['Adj_R2']

# שרטוט הגרפים
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.plot(num_features, test_mse, marker='o')
plt.xlabel('Number of Predictors')
plt.ylabel('Test MSE')
plt.title('Test MSE vs Number of Predictors')

plt.subplot(2, 2, 2)
plt.plot(num_features, adj_r2, marker='o')
plt.xlabel('Number of Predictors')
plt.ylabel('Adjusted R² (Train)')
plt.title('Adjusted R² vs Number of Predictors')

plt.subplot(2, 2, 3)
plt.plot(num_features, aic, marker='o')
plt.xlabel('Number of Predictors')
plt.ylabel('AIC (Train)')
plt.title('AIC vs Number of Predictors')

plt.subplot(2, 2, 4)
plt.plot(num_features, bic, marker='o')
plt.xlabel('Number of Predictors')
plt.ylabel('BIC (Train)')
plt.title('BIC vs Number of Predictors')

plt.tight_layout()
plt.show()

# בחירת המודל הסופי: המודל עם MSE הנמוך ביותר על סט הבדיקה (כפי שחישבנו קודם)
final_features = optimal_features  # מהשלב הקודם
final_model = forward_results.loc[optimal_idx, 'model']
# חיזוי עם המודל הסופי ו חישוב ביצועים על סט הבדיקה
final_preds = final_model.predict(sm.add_constant(X_test_scaled[final_features]))
final_mse = np.mean((final_preds - y_test) ** 2)
final_r2 = 1 - (np.sum((final_preds - y_test)**2) / np.sum((y_test - y_test.mean())**2))
print(f"\nביצועי מודל OLS הסופי (עם {optimal_num_features} משתנים) על סט הבדיקה: MSE = {final_mse:.4f},  R² = {final_r2:.4f}")

final_model = forward_results.loc[optimal_idx, 'model']
print(final_model.summary())



#-----------------pcr------------------


# בחירת מספר רכיבי PCA באמצעות Cross-Validation (5-fold)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
mse_cv = []

# נבחן מספר רכיבים מ-1 עד 10
for n in range(1, X_train_scaled.shape[1] + 1):
    # Pipeline: הפעלת PCA עם n רכיבים ולאחריו רגרסיה לינארית
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train_scaled)
    # מודל רגרסיה על הרכיבים
    model = LinearRegression()
    # חישוב MSE ב-CV עבור מודל זה
    scores = cross_val_score(model, X_train_pca, y_train, cv=cv, scoring='neg_mean_squared_error')
    mse_cv.append(-scores.mean())

# המרת רשימת ה-MSE הממוצעים לסדרת pandas (לנוחות)
mse_cv_series = pd.Series(mse_cv, index=range(1, X_train_scaled.shape[1] + 1))
best_components = mse_cv_series.idxmin()
print("MSE ממוצע ב-CV עבור כל מספר רכיבים PCA אפשרי:")
print(mse_cv_series.round(4))
print(f"\nה-MSE הנמוך ביותר ב-CV מתקבל עם {best_components} רכיבים ראשיים.")

# אימון PCA סופי עם מספר הרכיבים הנבחר
pca_final = PCA(n_components=best_components)
X_train_pca_final = pca_final.fit_transform(X_train_scaled)
X_test_pca_final = pca_final.transform(X_test_scaled)

# אימון מודל לינארי על הרכיבים הנבחרים
pcr_model = LinearRegression()
pcr_model.fit(X_train_pca_final, y_train)

# הערכת המודל על סט הבדיקה
pcr_preds = pcr_model.predict(X_test_pca_final)
pcr_mse = mean_squared_error(y_test, pcr_preds)
pcr_r2 = r2_score(y_test, pcr_preds)
print(f"PCR MSE על סט הבדיקה: {pcr_mse:.4f}")
print(f"PCR R² על סט הבדיקה: {pcr_r2:.4f}")


#-----------------pls------------------
# בחירת מספר רכיבי PLS באמצעות Cross-Validation (5-fold)
mse_cv_pls = []
for n in range(1, X_train_scaled.shape[1] + 1):
    pls = PLSRegression(n_components=n)
    scores = cross_val_score(pls, X_train_scaled, y_train, cv=cv, scoring='neg_mean_squared_error')
    mse_cv_pls.append(-scores.mean())

mse_cv_pls_series = pd.Series(mse_cv_pls, index=range(1, X_train_scaled.shape[1] + 1))
best_pls_components = mse_cv_pls_series.idxmin()
print("MSE ממוצע ב-CV עבור כל מספר רכיבי PLS אפשרי:")
print(mse_cv_pls_series.round(4))
print(f"\nה-MSE הנמוך ביותר ב-CV מתקבל עם {best_pls_components} רכיבי PLS.")

# אימון מודל PLS סופי עם מספר הרכיבים הנבחר
pls_model = PLSRegression(n_components=best_pls_components)
pls_model.fit(X_train_scaled, y_train)

# הערכת המודל על סט הבדיקה
pls_preds = pls_model.predict(X_test_scaled).ravel()  # נקבל כווקטור
pls_mse = mean_squared_error(y_test, pls_preds)
pls_r2 = r2_score(y_test, pls_preds)
print(f"PLS MSE על סט הבדיקה: {pls_mse:.4f}")
print(f"PLS R² על סט הבדיקה: {pls_r2:.4f}")




































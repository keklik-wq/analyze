import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
from sklearn.model_selection import train_test_split
from statsmodels.genmod.families.links import Log, Identity, Sqrt  # Важно: Log() вместо log()
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance

df = pd.read_csv("statistics_ml/task2/data_flare/flare.csv", header=0)

targets = ['c_flares', 'm_flares', 'x_flares']
y = df[targets]
X = df.drop(columns=targets)

cat_cols = ['class', 'size', 'dist', 'activity', 'evolution', 
            'prev_act', 'hist_complex', 'new_hist', 'area', 'area_largest_spot']
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
X = X.astype(float)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

num_cols = [col for col in X_train.columns if X_train[col].nunique() > 2 and 
           X_train[col].dtype in ['int64', 'float64']]

if len(num_cols) > 0:
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

best_deviance = float('inf')
general_link = Log() # identity, and sqrt
for target in targets:

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    final_model = sm.GLM(
        y_train_val[target], 
        sm.add_constant(X_train_val), 
            family=Poisson(link=general_link)
    ).fit()

    # 6. Оценка на тестовой выборке
    y_pred = final_model.predict(sm.add_constant(X_test))
    test_mae = mean_absolute_error(y_test[target], y_pred)
    test_deviance = mean_poisson_deviance(y_test[target], y_pred)

    print(f"Пуассоновская регрессия для {target}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Test Deviance: {test_deviance:.3f}")
    print(final_model.summary())
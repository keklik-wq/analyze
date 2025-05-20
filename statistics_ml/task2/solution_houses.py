import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import typing as t
import statsmodels.regression.linear_model as lm
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from scipy.stats import norm 
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import label_binarize

class ConstantPredictor:
    def predict(self, X):
         return np.zeros(X.shape[0])
    def predict_proba(self, X):
         return np.column_stack([np.ones(X.shape[0]), np.zeros(X.shape[0])])
    def decision_function(self, X):
        # Логит для P(y=1) = 0: log(0 / (1 - 0)) = -∞
        return np.full(X.shape[0], -np.inf)
    
class GeneralizedLinearModelsResolver:
    def __init__(
        self,
        df: pd.DataFrame,
        target_col: t.List[str],
        continious_cols: t.List[str],
        categorical_cols: t.List[str],
    ):
        self.df = df
        self.target_col = target_col
        self.continious_cols = continious_cols
        self.categorical_cols = categorical_cols
        self.X = None
        self.y = None
        self.qual_levels = None

    def _balance_data(self, X, y, sampling_strategy = 'auto'):
        class_counts = Counter(y)
        print(f"Исходное распределение: {class_counts}")
        
        if len(class_counts) < 2:
            return X, y
        
        if np.bincount(y)[1] < 6:
            return X, y
            
        minority_class = min(class_counts, key=class_counts.get)
        n_minority = class_counts[minority_class]
        
        if sampling_strategy == 'auto':
            strategy = 'oversample' if n_minority < 100 else 'undersample'
        else:
            strategy = sampling_strategy
        
        if strategy == 'oversample':
            print("Применяем SMOTE")
            return SMOTE(random_state=42).fit_resample(X, y)
        elif strategy == 'undersample':
            print("Применяем RandomUnderSampler")
            return RandomUnderSampler(random_state=42).fit_resample(X, y)
        else:
            return X, y

    def _prepare_data(self, df) -> None:
        df = df[self.continious_cols + self.categorical_cols + self.target_col]

        num_imputer = SimpleImputer(strategy="median")
        df[self.continious_cols] = num_imputer.fit_transform(df[self.continious_cols])

        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[self.categorical_cols] = cat_imputer.fit_transform(df[self.categorical_cols])

        for col in self.categorical_cols:
            df[col] = df[col].astype("category")

        X = df.drop(self.target_col, axis=1)
        X = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.X = pd.DataFrame(X_scaled, columns=X.columns)

        self.y = df[self.target_col[0]]
        self.qual_levels = sorted(self.y.unique())

        return None

    def _fit_and_compare_gamma_gaussian_5(self, data, target_col, test_size=0.2, random_state=42):
        
        # Проверка данных
        data = data[data[target_col] > 0]
        if (data[target_col] <= 0).any():
            raise ValueError("Целевая переменная должна содержать только положительные значения")
        
        # Подготовка данных
        X = data.drop(columns=[target_col])
        y = data[target_col]
        X = sm.add_constant(X)  # Добавляем константу для intercept
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        gamma_model = sm.GLM(
            y_train,
            X_train,
            family=sm.families.Gamma(link=sm.families.links.Log())  # Логарифмическая связь
        ).fit()
        
        gaussian_model = sm.OLS(y_train, X_train).fit()
        
        gamma_pred = gamma_model.predict(X_test)
        gaussian_pred = gaussian_model.predict(X_test)
       
        acc_gamma_report = mean_squared_error(y_test, gamma_pred)
        acc_gaussian_report = mean_squared_error(y_test, gaussian_pred)
        
        f1_gamma_report = mean_absolute_error(y_test, gamma_pred)
        f1_gaussian_report = mean_absolute_error(y_test, gaussian_pred)
        
        print("\n\n\nGamma and Gaussian Regressions")
        print(f"- Отчёт по гамме: mse - {acc_gamma_report} mae - {f1_gamma_report}")
        print(f"- Отчёт по гауссиану: mse - {acc_gaussian_report} mae - {f1_gaussian_report}")
        
        return None

    def _resolve_logistic_regression_1(self, X_train, y_train, X_test, y_test) -> t.List[LogisticRegression]:
        print(f"1. Бинарная логистическая регрессия для каждого уровня {self.target_col}:")
        
        models = []
        
        for level in self.qual_levels:
            # try:
                y_binary_train = (y_train == level).astype(int)
                y_binary_test = (y_test == level).astype(int)
                
                #X_train_res, y_binary_train_res = self._balance_data(X_train, y_binary_train)
                X_train_res, y_binary_train_res = X_train, y_binary_train
                
                if len(np.unique(y_binary_train)) < 2:
                    print(f"\nУровень {self.target_col} = {level}:")
                    print(
                        "- Недостаточно данных (только один класс в y_binary), пропускаем обучение."
                    )
                    continue

                model = LogisticRegression(max_iter=1000, penalty="l2", solver="lbfgs", class_weight='balanced')

                # ros = RandomOverSampler(random_state=42)
                # X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
                # model.fit(X_resampled, y_resampled)

                model.fit(X_train_res, y_binary_train_res)

                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_binary_test, y_pred)
                #f1_score_value = f1_score(y_test, y_pred)

                # if f1_score_value < 0.6:
                #     models.append(ConstantPredictor())
                # else:
                print(f"\nУровень {self.target_col} = {level}:")
                print(f"- Точность (Accuracy): {accuracy:.4f}")
                print(f"- Классы для уровня {level}: {np.bincount(y_binary_train)}")
                print(f"- Отчёт: {classification_report(y_binary_test, y_pred, zero_division=0)}")
                print(f"- Коэффициенты модели (первые 5): {model.coef_[0][:5]}...")
                print(f"- Свободный член (Intercept): {model.intercept_[0]}")
                a = np.bincount(y_binary_train) 
                if np.bincount(y_binary_train)[1] > 30:
                    models.append(model)
                else:
                    models.append(ConstantPredictor())
            # except:
            #      models.append(ConstantPredictor())

        return models

    def _resolve_probit_regression_2(self, X_train, y_train, X_test, y_test) -> t.List[lm.RegressionResultsWrapper]:
        print(f"2. Пробит-регрессия для каждого уровня {self.target_col}:")
        
        models = []
        
        for level in self.qual_levels:
            y_binary_train = (y_train == level).astype(int)
            y_binary_test = (y_test == level).astype(int)
            
            if len(np.unique(y_binary_train)) < 2:
                print(f"\nУровень {self.target_col} = {level}:")
                print(
                    "- Недостаточно данных (только один класс в y_binary_train), пропускаем обучение."
                )
                continue

            if X_train.isna().any().any() or X_test.isna().any().any():
                raise ValueError("В данных есть NaN после обработки!")

            if y_binary_train.isna().any().any() or y_binary_test.isna().any().any():
                raise ValueError("В данных есть NaN после обработки!")

            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_binary_train = y_binary_train.reset_index(drop=True)
            y_binary_test = y_binary_test.reset_index(drop=True)

            X_train_np = X_train.values
            X_test_np = X_test.values

            X_train_const = np.hstack([np.ones((X_train_np.shape[0], 1)), X_train_np])
            X_test_const = np.hstack([np.ones((X_test_np.shape[0], 1)), X_test_np])

            probit_model = sm.Probit(y_binary_train, X_train_const)

            probit_model = probit_model.fit_regularized(
                method="l1", alpha=10, disp=0, max_iter=10000
            )

            y_pred_prob = probit_model.predict(X_test_const)
            y_pred = (y_pred_prob >= 0.5).astype(int)

            accuracy = accuracy_score(y_binary_test, y_pred)

            print(f"\nУровень {self.target_col} = {level}:")
            print(f"- Классы для уровня {level}: {np.bincount(y_binary_train)}")
            print(f"- Точность (Accuracy): {accuracy:.4f}")
            print(f"- Отчёт: {classification_report(y_binary_test, y_pred, zero_division=0)}")
            print(
                f"- Коэффициенты модели (первые 5): {probit_model.params.iloc[1:6].values}..."
            )
            print(f"- Свободный член (Intercept): {probit_model.params.iloc[0]}")
        
            models.append(probit_model)
        
        return models
    
    def _fit_multinomial_logistic_regression_4(self, test_size=0.2, random_state=42):
        """
        Fits multinomial logistic regression for a categorical variable.
        If the category has only 2 levels, partitions a continuous variable into 5 quantile groups.
        
        Parameters:
        - df: pandas DataFrame
        - categorical_col: str, name of categorical column to predict
        - continuous_col: str, name of continuous column to use if categorical has 2 levels (optional)
        - test_size: float, proportion for test split
        - random_state: int, random seed
        
        Returns:
        - Trained model
        - Classification report
        """
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(self.y)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )

        # Fit multinomial logistic regression
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        model = LogisticRegression(
            multi_class='multinomial',
            class_weight=dict(zip(np.unique(y_train), class_weights)),
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test, y_pred, zero_division=0
        )
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print("Multinomial Logistic Regression")
        print(f"- Точность (Accuracy): {accuracy:.4f}")
        print(f"- Отчёт: {report}")
        print(f"- Коэффициенты модели (первые 5): {model.coef_[0][:5]}...")
        print(f"- Свободный член (Intercept): {model.intercept_[0]}")
        
        return model, report


    def _evaluate_multiclass_likelihood_3(self, logistic_models, probit_models, X_test, y_test):
        classes = np.array(sorted(self.y.unique()))
        n_classes = len(classes)
        
        # Определяем тип целевой переменной
        is_numeric = np.issubdtype(y_test.dtype, np.number)
        
        # Создаем placeholder для неизвестных значений в соответствии с типом
        unknown_label = -1 if is_numeric else "UNK"
        
        y_proba_logistic = np.zeros((len(X_test), n_classes))
        y_proba_probit = np.zeros((len(X_test), n_classes))

        print("Формирование предсказаний по бинарным моделям...")

        for idx, cls in enumerate(classes):
            # Логистическая регрессия
            logits = np.hstack([model.decision_function(X_test).reshape(-1, 1) for model in logistic_models])
            y_proba_logistic = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

            # Пробит-регрессия
            if probit_models[idx] is not None:
                try:
                    X_test_const = sm.add_constant(X_test, has_constant='add')
                    linear_pred = probit_models[idx].predict(X_test_const)
                    probs = norm.cdf(linear_pred)
                    y_proba_probit[:, idx] = probs
                except Exception as e:
                    print(f"[Probit] Ошибка для класса {cls}: {e}")
                    y_proba_probit[:, idx] = 0.0
            else:
                y_proba_probit[:, idx] = 0.0

        # Защита от inf и nan
        y_proba_logistic = np.clip(y_proba_logistic, 1e-12, 1 - 1e-12)
        y_proba_probit = np.clip(y_proba_probit, 1e-12, 1 - 1e-12)

        # Нормализация вероятностей
        y_proba_logistic /= y_proba_logistic.sum(axis=1, keepdims=True)
        y_proba_probit /= y_proba_probit.sum(axis=1, keepdims=True)

        # Порог классификации
        threshold = 0.1
        
        # Предсказания с учетом типа данных
        if is_numeric:
            y_pred_logistic = np.where(
                np.max(y_proba_logistic, axis=1) > threshold,
                classes[np.argmax(y_proba_logistic, axis=1)],
                unknown_label
            )
            y_pred_probit = np.where(
                np.max(y_proba_probit, axis=1) > threshold,
                classes[np.argmax(y_proba_probit, axis=1)],
                unknown_label
            )
        else:
            y_pred_logistic = np.where(
                np.max(y_proba_logistic, axis=1) > threshold,
                classes[np.argmax(y_proba_logistic, axis=1)],
                unknown_label
            )
            y_pred_probit = np.where(
                np.max(y_proba_probit, axis=1) > threshold,
                classes[np.argmax(y_proba_probit, axis=1)],
                unknown_label
            )

        # Метрики
        acc_logistic = accuracy_score(y_test, y_pred_logistic)
        acc_probit = accuracy_score(y_test, y_pred_probit)

        # Для F1-score нужно убедиться, что все классы присутствуют
        unique_test = np.unique(y_test)
        unique_pred_logistic = np.unique(y_pred_logistic)
        unique_pred_probit = np.unique(y_pred_probit)
        
        # Все уникальные классы, которые встречаются в данных
        all_classes = np.union1d(unique_test, np.union1d(unique_pred_logistic, unique_pred_probit))
        
        f1_logistic = f1_score(y_test, y_pred_logistic, average='macro', labels=all_classes)
        f1_probit = f1_score(y_test, y_pred_probit, average='macro', labels=all_classes)

        # One-hot кодирование для log-loss
        y_true_onehot = label_binarize(y_test, classes=classes)
        if y_true_onehot.shape[1] == 1:  # Для бинарного случая
            y_true_onehot = np.hstack([1 - y_true_onehot, y_true_onehot])

        # Log-Likelihood
        ll_logistic = -log_loss(y_true_onehot, y_proba_logistic, normalize=True)
        ll_probit = -log_loss(y_true_onehot, y_proba_probit, normalize=True)

        print("\nРезультаты оценки:")
        print(f"Логистическая регрессия — Accuracy: {acc_logistic:.4f}, Log-Likelihood: {ll_logistic:.4f}, F1: {f1_logistic:.4f}")
        print(f"Пробит-регрессия — Accuracy: {acc_probit:.4f}, Log-Likelihood: {ll_probit:.4f}, F1: {f1_probit:.4f}")

        winner = "Логистическая регрессия" if ll_logistic > ll_probit else "Пробит-регрессия"
        print(f"Лучшая модель по правдоподобию: {winner}")

        return {
            "accuracy_logistic": acc_logistic,
            "log_likelihood_logistic": ll_logistic,
            "accuracy_probit": acc_probit,
            "log_likelihood_probit": ll_probit
        }


    def resolve(self):
        self._prepare_data(self.df)
        X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42, stratify = self.y
            )
        logistic_models = self._resolve_logistic_regression_1(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        print("\n\n\n")
        probit_models = self._resolve_probit_regression_2(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        self._evaluate_multiclass_likelihood_3(logistic_models=logistic_models,
                                            probit_models=probit_models,
                                            X_test=X_test,
                                            y_test=y_test)
        
        self._fit_multinomial_logistic_regression_4()
        self._fit_and_compare_gamma_gaussian_5(self.X, 'SalePrice')
        
    
if __name__ == "__main__":
    df = pd.read_csv("statistics_ml/task2/data_houses/house_prices.csv", header=0)

    house_prices_resolver = GeneralizedLinearModelsResolver(
        df=df,
        target_col=["ExterQual"],
        continious_cols=[
            "LotFrontage",
            "LotArea",
            "YearBuilt",
            "YearRemodAdd",
            "1stFlrSF",
            "2ndFlrSF",
            "GrLivArea",
            "TotalBsmtSF",
            "GarageArea",
            "SalePrice"
        ],
        categorical_cols=[
            "MSZoning",
            "Street",
            "Alley",
            "LotShape",
            "LandContour",
            "OverallQual",
            "Utilities",
            "LotConfig",
            "LandSlope",
            "Neighborhood",
        ],
    )
    house_prices_resolver.resolve()

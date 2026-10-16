#%%
import sys
import matplotlib
import numpy as np
import pandas as pd
import sklearn

# Печатаем версии среды и основных библиотек одним блоком.
print("Python:", sys.version)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("scikit-learn:", sklearn.__version__)

#%%
# Скрываем только предупреждения о будущих изменениях API;
# остальные предупреждения оставляем видимыми.
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Базовые библиотеки для вычислений, таблиц и графиков.
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Компоненты scikit-learn сгруппированы по назначению.
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Все случайные операции используют одно значение для воспроизводимости.
RANDOM_STATE = 1004
DATA_ID = 16
TEST_SIZE = 0.20
SCORING = "f1_macro"

#%%
from IPython.display import display
from pathlib import Path

# Сначала пытаемся использовать приложенную локальную копию.
example_path = Path("ML_Lab1_Datasets") / "variant_04.csv"

if example_path.exists():
    frame = pd.read_csv(example_path)
    print("Источник: локальная копия", example_path)
else:
    # Резервный вариант нужен, если архив с данными не был распакован.
    data = fetch_openml(data_id=16, as_frame=True)
    frame = data.data.copy()
    frame["class"] = data.target
    print("Источник: OpenML, data_id 16")

X = frame.drop(columns=["class"]).copy()
y = pd.to_numeric(frame["class"]).astype("int64")

# Проверяем размерность, классы и первые строки до дальнейшей обработки.
print("Размер X:", X.shape)
print("Классы:", sorted(y.unique()))
display(X.head())

# Карточка признаков объединяет тип, число значений и статистику пропусков.
data_card = pd.DataFrame({
    "dtype": X.dtypes.astype(str),
    "n_unique": X.nunique(dropna=True),
    "n_missing": X.isna().sum(),
    "missing_share": X.isna().mean().round(3),
})

# Отдельно оцениваем абсолютный и относительный размер каждого класса.
class_table = pd.DataFrame({
    "count": y.value_counts().sort_index(),
    "share": y.value_counts(normalize=True).sort_index().round(3),
})

# display удобнее print для просмотра таблиц внутри Jupyter Notebook.
display(data_card)
display(class_table)

# На одном рисунке показываем баланс классов и распределение возраста.
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Столбчатая диаграмма позволяет быстро увидеть дисбаланс классов.
y.value_counts().sort_index().plot.bar(ax=axes[0], color="0.45")
axes[0].set_title("Распределение классов")
axes[0].set_xlabel("class")
axes[0].set_ylabel("Число объектов")

plot_frame = pd.DataFrame({"att1": X["att1"], "class": y})
# Boxplot показывает медиану, разброс и возможные выбросы по каждому классу.
plot_frame.boxplot(column="att1", by="class", ax=axes[1])
axes[1].set_title("класс по атрибуту att1")
axes[1].set_xlabel("class")
axes[1].set_ylabel("att1")

# Pandas добавляет общий заголовок автоматически — убираем его перед выводом.
plt.suptitle("")
plt.tight_layout()
plt.show()

#%%
#Далее идёт уже ML
## 5. Разбиение данных
#%%
# Тестовая выборка отделяется один раз и остаётся закрытой до финальной оценки.
# stratify сохраняет приблизительные доли классов в обеих частях.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

# Кросс-валидация применяется только внутри обучающей части.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

print("CV:", cv)

# Размерности помогают проверить, что разбиение выполнено ожидаемым образом.
print("Train:", X_train.shape)
print("Test:", X_test.shape)



#%%
## 6. Базовая модель
# most_frequent всегда предсказывает самый частый класс и не использует признаки.
baseline = DummyClassifier(strategy="most_frequent")

# Используем ту же CV-схему, что и для основных моделей.
baseline_cv = cross_validate(
    baseline,
    X_train,
    y_train,
    cv=cv,
    scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
)

print(baseline_cv)

# Усредняем метрики по пяти фолдам для компактного сравнения.
baseline_summary = pd.Series({
    "accuracy_cv_mean": baseline_cv["test_accuracy"].mean(),
    "f1_macro_cv_mean": baseline_cv["test_f1_macro"].mean(),
})
display(baseline_summary.round(4))
#%%
## 7. Подготовка данных для ручной реализации SGD

manual_features = X.axes[1][:-1]

X_demo_train, X_demo_val, y_demo_train, y_demo_val = train_test_split(
    X_train[manual_features],
    y_train,
    test_size=0.20,
    stratify=y_train,
    random_state=RANDOM_STATE,
)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_demo_train = imputer.fit_transform(X_demo_train)
X_demo_val = imputer.transform(X_demo_val)

X_demo_train = scaler.fit_transform(X_demo_train)
X_demo_val = scaler.transform(X_demo_val)

y_demo_train_pm = np.where(y_demo_train.to_numpy() == 1, 1.0, -1.0)
y_demo_val_pm = np.where(y_demo_val.to_numpy() == 1, 1.0, -1.0)
#%%
## 8. Функции потерь и обучения

def sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def mean_logistic_loss(X, y_pm, w, b):
    margins = y_pm * (X @ w + b)
    return np.mean(np.logaddexp(0.0, -margins))


def train_logreg_sgd(
    X_train, y_train_pm, X_val, y_val_pm, *,
    eta=0.01, tau=1e-4, epochs=50, random_state=RANDOM_STATE,
):
    rng = np.random.default_rng(random_state)
    w = np.zeros(X_train.shape[1], dtype=float)
    b = 0.0
    history = {"train_loss": [], "val_loss": [], "weight_norm": []}

    for _ in range(epochs):
        for i in rng.permutation(len(X_train)):
            x_i = X_train[i]
            y_i = y_train_pm[i]
            margin = y_i * (x_i @ w + b)

            factor = -y_i * sigmoid(-margin)
            grad_w = factor * x_i + tau * w
            grad_b = factor

            w -= eta * grad_w
            b -= eta * grad_b

        history["train_loss"].append(mean_logistic_loss(X_train, y_train_pm, w, b))
        history["val_loss"].append(mean_logistic_loss(X_val, y_val_pm, w, b))
        history["weight_norm"].append(np.linalg.norm(w))

    return w, b, pd.DataFrame(history)
#%%
## 9. Исследование темпа обучения
eta_values = [0.0001, 0.001, 0.01, 0.1]
eta_runs = {}

for eta in eta_values:
    _, _, history = train_logreg_sgd(
        X_demo_train, y_demo_train_pm,
        X_demo_val, y_demo_val_pm,
        eta=eta, tau=1e-4, epochs=50,
        random_state=RANDOM_STATE,
    )
    eta_runs[eta] = history
    plt.plot(history["val_loss"], label=f"eta={eta}")

plt.xlabel("Эпоха")
plt.ylabel("Validation log-loss")
plt.title("Влияние темпа обучения")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

eta_result = pd.DataFrame({
    eta: {
        "final_train_loss": h["train_loss"].iloc[-1],
        "final_val_loss": h["val_loss"].iloc[-1],
        "weight_norm": h["weight_norm"].iloc[-1],
    }
    for eta, h in eta_runs.items()
}).T
display(eta_result.round(4))
#%%
## 10. Исследование регуляризации
best_eta = eta_result["final_val_loss"].idxmin()
tau_values = [0.0, 1e-4, 1e-2]
tau_result = []

for tau in tau_values:
    w_sgd, b_sgd, history = train_logreg_sgd(
        X_demo_train, y_demo_train_pm,
        X_demo_val, y_demo_val_pm,
        eta=float(best_eta), tau=tau, epochs=50,
        random_state=RANDOM_STATE,
    )
    tau_result.append({
        "tau": tau,
        "val_loss": history["val_loss"].iloc[-1],
        "weight_norm": np.linalg.norm(w_sgd),
    })

    plt.plot(history["val_loss"], label=f"tau={tau}")

plt.xlabel("Эпоха")
plt.ylabel("Validation log-loss")
plt.title("Влияние темпа обучения")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

tau_result = pd.DataFrame(tau_result)
display(tau_result.round(4))
#%%
## 11. Полный preprocessing pipeline
numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = [
    column for column in X_train.columns
    if column not in numeric_features
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

sgd_pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", SGDClassifier(
        loss="log_loss",
        penalty="l2",
        learning_rate="constant",
        max_iter=2000,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )),
])
#%%
## 12. Подбор гиперпараметров SGD
sgd_param_grid = {
    "model__eta0": [0.001, 0.01, 0.1],
    "model__alpha": [1e-6, 1e-4, 1e-2],
}

sgd_search = GridSearchCV(
    estimator=sgd_pipeline,
    param_grid=sgd_param_grid,
    scoring=SCORING,
    cv=cv,
    n_jobs=-1,
    return_train_score=True,
)
sgd_search.fit(X_train, y_train)

print("Лучшие параметры SGD:", sgd_search.best_params_)
print("Лучший CV macro-F1:", round(sgd_search.best_score_, 4))

sgd_cv_table = pd.DataFrame(sgd_search.cv_results_)[[
    "param_model__eta0",
    "param_model__alpha",
    "mean_train_score",
    "mean_test_score",
    "std_test_score",
    "rank_test_score",
]].sort_values("rank_test_score")
display(sgd_cv_table.round(4))
#%%
## 13. Тепловая карта CV
pivot = sgd_cv_table.pivot_table(
    index="param_model__alpha",
    columns="param_model__eta0",
    values="mean_test_score",
)

fig, ax = plt.subplots(figsize=(6, 4))
image = ax.imshow(pivot.to_numpy(), cmap="Greys", aspect="auto")
ax.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
ax.set_yticks(range(len(pivot.index)), labels=pivot.index)
ax.set_xlabel("eta0")
ax.set_ylabel("alpha")
ax.set_title("CV macro-F1 для SGDClassifier")
fig.colorbar(image, ax=ax, label="macro-F1")
plt.tight_layout()
plt.show()
#%%
## 14. Подбор C для LogisticRegression
logreg_pipeline = Pipeline([
    ("prep", clone(preprocessor)),
    ("model", LogisticRegression(
        max_iter=3000,
        random_state=RANDOM_STATE,
    )),
])

logreg_search = GridSearchCV(
    estimator=logreg_pipeline,
    param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
    scoring=SCORING,
    cv=cv,
    n_jobs=-1,
)
logreg_search.fit(X_train, y_train)

cv_comparison = pd.DataFrame([
    {
        "model": "DummyClassifier",
        "best_params": "strategy=most_frequent",
        "cv_macro_f1": baseline_summary["f1_macro_cv_mean"],
    },
    {
        "model": "SGDClassifier",
        "best_params": str(sgd_search.best_params_),
        "cv_macro_f1": sgd_search.best_score_,
    },
    {
        "model": "LogisticRegression",
        "best_params": str(logreg_search.best_params_),
        "cv_macro_f1": logreg_search.best_score_,
    },
]).sort_values("cv_macro_f1", ascending=False)

display(cv_comparison.round(4))

candidate_comparison = cv_comparison[
    cv_comparison["model"] != "DummyClassifier"
].reset_index(drop=True)
#%%
## 15. Финальная оценка на test
def evaluate_classifier(name, estimator, X_eval, y_eval):
    prediction = estimator.predict(X_eval)
    probabilities = estimator.predict_proba(X_eval)
    classes = estimator.classes_

    if len(classes) == 2:
        positive_class = classes[1]
        y_binary = (np.asarray(y_eval) == positive_class).astype(int)
        auc = roc_auc_score(y_binary, probabilities[:, 1])
    else:
        auc = roc_auc_score(
            y_eval, probabilities, labels=classes,
            multi_class="ovr", average="macro",
        )

    return {
        "model": name,
        "accuracy": accuracy_score(y_eval, prediction),
        "precision_macro": precision_score(y_eval, prediction, average="macro", zero_division=0),
        "recall_macro": recall_score(y_eval, prediction, average="macro", zero_division=0),
        "f1_macro": f1_score(y_eval, prediction, average="macro", zero_division=0),
        "roc_auc": auc,
    }
#%%
baseline.fit(X_train, y_train)

final_models = {
    "DummyClassifier": baseline,
    "SGDClassifier": sgd_search.best_estimator_,
    "LogisticRegression": logreg_search.best_estimator_,
}

test_results = pd.DataFrame([
    evaluate_classifier(name, model, X_test, y_test)
    for name, model in final_models.items()
])
display(test_results.round(4))

selected_name = candidate_comparison.iloc[0]["model"]
selected_model = final_models[selected_name]
selected_prediction = selected_model.predict(X_test)

print(classification_report(y_test, selected_prediction, zero_division=0))

ConfusionMatrixDisplay.from_estimator(
    selected_model, X_test, y_test,
    normalize="true", cmap="Greys",
)
plt.title(f"Нормированная confusion matrix: {selected_name}")
plt.show()
#%%
## 16. Анализ ошибок
error_table = X_test.copy()
error_table["true"] = np.asarray(y_test)
error_table["predicted"] = selected_prediction

selected_probabilities = selected_model.predict_proba(X_test)
error_table["confidence"] = selected_probabilities.max(axis=1)

errors = error_table[
    error_table["true"] != error_table["predicted"]
].sort_values("confidence", ascending=False)

display(errors.head(10))
#%%
## 17. Анализ коэффициентов
feature_names = selected_model.named_steps["prep"].get_feature_names_out()
model = selected_model.named_steps["model"]

class_index = 0
coefficients = model.coef_[class_index]

coefficient_table = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "abs_coefficient": np.abs(coefficients),
}).sort_values("abs_coefficient", ascending=False)

display(coefficient_table.head(15))

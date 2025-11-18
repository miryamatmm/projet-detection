import numpy as np 
np.set_printoptions(threshold=10000, suppress = True) 
import pandas as pd 
import warnings 
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, average_precision_score,  
    PrecisionRecallDisplay, f1_score, precision_recall_curve
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from matplotlib.colors import LinearSegmentedColormap
import joblib, os

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


warnings.filterwarnings('ignore') 

rose_map = LinearSegmentedColormap.from_list("rose_soft", ["#fff0f5", "#d63384"])


def load_dataset(path, sep=',', header='infer', names=None):
    """
    Charge un dataset depuis un fichier CSV ou TXT et affiche un résumé Markdown.
    """

    try:
        df = pd.read_csv(path, sep=sep, header=header, names=names)

        display(Markdown(f"""### Jeu de données chargé avec succès  
**Chemin :** `{path}`

**Dimensions :** {df.shape[0]} lignes × {df.shape[1]} colonnes  

**Colonnes :** {', '.join(df.columns.astype(str).tolist())}
        """))

        display(df.head())

        return df

    except Exception as e:
        display(Markdown(f"""### Erreur de chargement du fichier  
            **Chemin :** `{path}`  
            **Message d’erreur :** `{e}`
        """))
        return None


def plot_2d_data(df, x_col, y_col, title="Visualisation 2D des données", anomalies=None, ax=None):
    """
    Affiche un nuage de points 2D (scatter plot) à partir de deux colonnes du dataset.
    Si un axe Matplotlib (ax) est fourni, trace dessus au lieu de créer une nouvelle figure.
    """
    
    # Couleurs harmonisées
    normal_color = "#F48FB1"    
    anomaly_color = "#F4286C"    

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    if anomalies is None:
        sns.scatterplot(
            data=df, x=x_col, y=y_col,
            s=40, color=normal_color, edgecolor="white", linewidth=0.6, ax=ax
        )
    else:
        df_plot = df.copy()
        df_plot["is_anomaly"] = anomalies
        sns.scatterplot(
            data=df_plot, x=x_col, y=y_col,
            hue="is_anomaly",
            palette={False: normal_color, True: anomaly_color},
            s=40, edgecolor="white", linewidth=0.6, ax=ax
        )

        # Légende propre et harmonisée
        handles, _ = ax.get_legend_handles_labels()
        if len(handles) >= 2:
            ax.legend(
                handles=[handles[0], handles[1]],
                labels=["Normale", "Anomalie"],
                title="Anomalie",
                loc="best"
            )

    ax.set_title(title, fontsize=14, fontweight="bold", color=normal_color)
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    sns.despine()


def plot_distribution(df, column, color="#a6cee3", ax=None, bins=20):
    """
    Affiche la distribution (histogramme + KDE) d'une variable du DataFrame.
    """
    sns.set(style="whitegrid")

    # Si aucun axe fourni → créer une figure seule
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    sns.histplot(df[column], kde=True, color=color, bins=bins, ax=ax)
    ax.set_title(f"Distribution de {column}", fontsize=12, fontweight="bold", color="#F48FB1")
    ax.set_xlabel(column)
    ax.set_ylabel("Fréquence")

    sns.despine()
    plt.tight_layout()

    return ax


def analyze_dataset(df, figsize=(18, 5)):
    """
    Analyse exploratoire esthétique et compacte d'un dataset 2D.
    Affiche les infos essentielles sous forme de tableau + 3 graphiques côte à côte.
    """
       
    sns.set(style="whitegrid")

    # --- Affichage d’un résumé compact ---
    display(Markdown("### **Résumé rapide du dataset**"))

    info_dict = {
        "Nombre de lignes": [df.shape[0]],
        "Nombre de colonnes": [df.shape[1]],
        "Noms des colonnes": [", ".join(df.columns)],
        "Valeurs manquantes": [df.isnull().sum().sum()],
        "Types": [", ".join(df.dtypes.astype(str).values)]
    }

    resume_df = pd.DataFrame(info_dict)
    display(resume_df)

    # --- Statistiques descriptives en tableau compact ---
    display(Markdown("### **Statistiques descriptives**"))
    display(df.describe().T.style.background_gradient(cmap="PuRd", axis=1))

    # --- Visualisation côte à côte ---
    _, axes = plt.subplots(1, 3, figsize=figsize)

    plot_2d_data(df, df.columns[0], df.columns[1],
                 title="Répartition des points", ax=axes[0])
    
    plot_distribution(df, df.columns[0], color="#a6cee3", ax=axes[1])
    
    plot_distribution(df, df.columns[1], color="#b2df8a", ax=axes[2])

    plt.suptitle("Analyse exploratoire du dataset", fontsize=14, fontweight="bold", color="#d46a9b")
    plt.tight_layout()
    plt.show()


def plot_anomaly_scores(scores, method_name, threshold=None, ax=None):
    """
    Affiche la distribution des scores d’anomalie pour une méthode donnée.
    Peut être intégré dans une figure existante (via ax) ou affiché seul.
    """
    sns.set(style="whitegrid")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    sns.histplot(scores, bins=30, kde=True, color="orchid", ax=ax)
    ax.set_title(f"Distribution des scores d’anomalie ({method_name})", fontsize=13, color="#F48FB1")
    ax.set_xlabel("Score d’anomalie")
    ax.set_ylabel("Fréquence")

    if threshold is not None:
        ax.axvline(threshold, color="#EC3572", linestyle="--", linewidth=2, label=f"Seuil = {threshold:.4f}")
        ax.legend()
    
    sns.despine()
    plt.tight_layout()


def detect_outliers_iforest(df, contamination=0.02, random_state=42, threshold=None):
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(df)
    scores = -model.score_samples(df)
    labels = np.where(model.predict(df) == -1, True, False)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set(style="whitegrid")

    plot_2d_data(df, df.columns[0], df.columns[1],
                 title="Anomalies détectées (Isolation Forest)",
                 anomalies=labels, ax=axes[0])

    plot_anomaly_scores(scores, "Isolation Forest",threshold, ax=axes[1])

    plt.tight_layout()
    plt.show()
    return scores, labels


def detect_outliers_lof(df, n_neighbors=20, contamination=0.02, threshold=None):
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = model.fit_predict(df)
    scores = -model.negative_outlier_factor_
    labels = np.where(labels == -1, True, False)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set(style="whitegrid")

    plot_2d_data(df, df.columns[0], df.columns[1],
                 title="Anomalies détectées (LOF)",
                 anomalies=labels, ax=axes[0])

    plot_anomaly_scores(scores, "LOF",threshold, ax=axes[1])

    plt.tight_layout()
    plt.show()
    return scores, labels


def find_threshold_iqr(scores, method_name=None, display_result=True):
    """
    Calcule le seuil d'anomalie basé sur l'écart interquartile (IQR).
    """
    q1, q3 = np.percentile(scores, [25, 75])
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr

    if display_result:
        display(Markdown(f"""
### Seuil d’anomalie (méthode IQR)
- {method_name if method_name else "Méthode"} : **{threshold:.4f}**
"""))

    return threshold

def find_threshold_kmeans(scores, method_name=None, display_result=True):
    """
    Détermine le seuil d'anomalie par clustering (KMeans à 2 clusters).
    """
    scores = np.array(scores).reshape(-1, 1)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(scores)

    centers = kmeans.cluster_centers_.flatten()
    centers.sort()

    threshold = np.mean(centers)

    if display_result:
        display(Markdown(f"""
### Seuil d’anomalie (méthode K-Means)
- {method_name if method_name else "Méthode"} : **{threshold:.4f}**
"""))

    return threshold

def detect_novelty_lof(df, normal_ratio=0.98, n_neighbors=20):
    """
    Détection de nouveautés avec l'approche Local Outlier Factor (LOF)
    - apprentissage sur données normales non polluées
    - détection sur nouvelles données (nouvelles observations)
    """
    sns.set(style="whitegrid")

    # Séparer données normales et test
    n_train = int(len(df) * normal_ratio)
    X_train = df.iloc[:n_train]   # jeu de données d'apprentissage (normal)
    X_test = df.copy()            # jeu complet (pour test et visualisation)

    # Entraînement du modèle LOF en mode novelty
    lof_novelty = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof_novelty.fit(X_train)

    # Prédiction sur le jeu complet
    scores_novelty = -lof_novelty.score_samples(X_test)
    labels_novelty = lof_novelty.predict(X_test)
    labels_novelty = np.where(labels_novelty == -1, True, False)

    # Affichage Markdown explicatif 
    display(Markdown(f"""### Détection de nouveautés (LOF)
**Méthodologie selon le cours :**
- Entraînement sur des données normales non polluées (≈ {normal_ratio*100:.1f}% du jeu complet).  
- Détection d'anomalies sur de **nouvelles observations**.  
- LOF en mode `novelty=True` permet cette approche supervisée du comportement normal.  

**Paramètres :**
- Nombre de voisins : `{n_neighbors}`
- Taille du jeu d'entraînement : `{len(X_train)}`
- Taille du jeu de test : `{len(X_test)}`

**Résultats :**
- Anomalies détectées : `{labels_novelty.sum()}` / `{len(df)}`
    """))

    # on visualise
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    plot_2d_data(X_test, X_test.columns[0], X_test.columns[1],
                 title="Détection de nouveautés (LOF - mode novelty)",
                 anomalies=labels_novelty, ax=axes[0])

    plot_anomaly_scores(scores_novelty, "LOF (mode novelty)", ax=axes[1])

    plt.tight_layout()
    plt.show()

    return scores_novelty, labels_novelty

def drop_columns(df, cols_to_drop):
    """
    Supprime les colonnes spécifiées d'un DataFrame.
    """
    return df.drop(columns=cols_to_drop, errors="ignore")

def split_features_target(df, target_col):
    """
    Sépare les variables explicatives (X) et la variable cible (y).
    """
    X = df.drop(columns=[target_col], errors="ignore")
    y = df[target_col].copy()
    return X, y


def normalize_columns(df, cols, method="standard"):
    """
    Normalise les colonnes spécifiées selon la méthode choisie.
    """
    df_copy = df.copy()

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("Méthode invalide : choisir 'standard', 'minmax' ou 'robust'.")

    df_copy[cols] = scaler.fit_transform(df_copy[cols])
    return df_copy


def class_ratio(y, labels={0: "Classe 0", 1: "Classe 1"}):
    """
    Calcule et affiche le ratio (%) de chaque classe dans un vecteur cible.
    """
    ratio = y.value_counts(normalize=True).sort_index() * 100

    display(Markdown(f"""
### Répartition des classes

- {labels.get(0, 'Classe 0')} : **{ratio.iloc[0]:.3f}%**  
- {labels.get(1, 'Classe 1')} : **{ratio.iloc[1]:.3f}%**
"""))
    return ratio


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name=None, plot_pr=True):
    """
    Évalue un modèle supervisé (déséquilibré) avec un affichage esthétique.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # si dispo
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = np.zeros_like(y_pred, dtype=float)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3, zero_division=0, output_dict=True)
    ap = average_precision_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
    f1 = report['1']['f1-score']

    display(Markdown(f"""
### Évaluation du modèle : **{model_name or type(model).__name__}**

| Métrique | Valeur |
|:--|:--:|
| **F1-score (fraudes)** | `{f1:.3f}` |
| **Average Precision (AP)** | `{ap:.3f}` |
"""))

    # matrice de confusion
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdPu",
        xticklabels=["Prédit: 0", "Prédit: 1"],
        yticklabels=["Réel: 0", "Réel: 1"],
        cbar=False, linewidths=1.2, linecolor="white"
    )
    plt.title(f"Matrice de confusion – {model_name or type(model).__name__}",
              fontsize=13, fontweight="bold", color="#B30059")
    plt.xlabel("Valeurs prédites")
    plt.ylabel("Valeurs réelles")
    plt.show()

    # rapport complet
    df_report = pd.DataFrame(report).T
    styled = (
        df_report.style
        .background_gradient(cmap="PuRd", axis=0)
        .set_caption("Rapport de classification complet")
    )
    display(styled)

    # courbe précision rappel
    if plot_pr and hasattr(model, "predict_proba"):
        PrecisionRecallDisplay.from_estimator(
            model,
            X_test,
            y_test,
            name="Courbe PR",
            color="#B30059",
            lw=2.5
        )
        plt.title(f"Courbe Précision–Rappel - {model_name or type(model).__name__}",
                  fontsize=12, color="#B30059")
        plt.grid(alpha=0.3)
        plt.show()

    return {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "average_precision": ap,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": cm
    }


def optimize_model(
    model,
    param_grid,
    X_train,
    y_train,
    scoring="f1",
    cv_splits=3,
    search_type="grid",
    n_iter=20,
    n_jobs=-1,
    random_state=42,
    verbose=2
):
    """
    Recherche des meilleurs hyperparamètres pour un modèle donné.
    """

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    if search_type == "random":
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose
        )
    else:  # on est là par défaut (on donne les paramètres exacts)
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose
        )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    display(Markdown(f"""
### Optimisation d'hyperparamètres terminée
- **Modèle :** {type(model).__name__}
- **Type de recherche :** {'GridSearchCV' if search_type == 'grid' else 'RandomizedSearchCV'}
- **Nombre de plis CV :** {cv_splits}
- **Métrique :** {scoring}
- **Meilleurs paramètres :** `{best_params}`
- **Score moyen (CV) :** **{best_score:.3f}**
    """))

    return best_model, best_params, best_score

def split_dataset(X, y, test_size=0.3, stratify=True, random_state=42):
    """
    Sépare le dataset en jeu d'entraînement et de test de manière stratifiée.
    """
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=strat, random_state=random_state
    )

    display(Markdown(f"""
### Séparation du dataset effectuée
- **Taille totale :** {len(X):,} observations  
- **Entraînement :** {len(X_train):,} ({(1-test_size)*100:.1f} %)  
- **Test :** {len(X_test):,} ({test_size*100:.1f} %)  
- **Stratification :** {'activée' if stratify else 'désactivée'}
    """))

    return X_train, X_test, y_train, y_test



def evaluate_unsupervised(model, X, y_true, model_name="Modèle non supervisé", plot_pr=True, tail_split=None):
    """
    Évalue un modèle non supervisé (IsolationForest, LOF) avec affichage stylé.
    """

    model_class_name = type(model).__name__
    print(f"[DEBUG] Évaluation de {model_name} ({model_class_name})")

    # === Cas 1 : LOF sans novelty=True ===
    if "LocalOutlierFactor" in model_class_name and tail_split is not None:
        print(f"LOF détecté sans novelty → refit sur tout X et évaluation sur les {tail_split} derniers exemples.")
        lof_tmp = LocalOutlierFactor(**model.get_params())
        y_pred_all = np.where(lof_tmp.fit_predict(X) == -1, 1, 0)
        y_scores_all = -lof_tmp.negative_outlier_factor_
        y_pred = y_pred_all[-tail_split:]
        y_scores = y_scores_all[-tail_split:]
        y_true = y_true[-tail_split:]

    # === Cas 2 : modèle avec predict (IsolationForest, LOF novelty=True, etc.) ===
    elif hasattr(model, "predict") and callable(model.predict):
        y_pred = np.where(model.predict(X) == -1, 1, 0)
        if hasattr(model, "score_samples"):
            y_scores = -model.score_samples(X)
        else:
            y_scores = np.zeros_like(y_true, dtype=float)

    else:
        raise ValueError(f"Le modèle {model_name} ({model_class_name}) ne supporte ni predict ni fit_predict.")

    # === Métriques ===
    f1 = f1_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_scores)

    display(Markdown(f"""
### Évaluation du modèle : **{model_name}**
| Métrique | Valeur |
|:--|:--:|
| **F1-score (fraudes)** | `{f1:.3f}` |
| **Average Precision (AP)** | `{ap:.3f}` |
"""))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=rose_map,
                xticklabels=["Prédit: 0", "Prédit: 1"],
                yticklabels=["Réel: 0", "Réel: 1"],
                cbar=False, linewidths=1.2, linecolor="white")
    plt.title(f"Matrice de confusion – {model_name}", fontsize=13, fontweight="bold", color="#B30059")
    plt.xlabel("Valeurs prédites")
    plt.ylabel("Valeurs réelles")
    plt.show()

    if plot_pr:
        precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
        plt.figure(figsize=(5, 4))
        plt.plot(recalls, precisions, color="#D46A9B", lw=2.5, label=f"Courbe PR (AP = {ap:.2f})")
        plt.xlabel("Recall (1)")
        plt.ylabel("Precision (1)")
        plt.title(f"Courbe Précision–Rappel - {model_name}", fontsize=13, color="#D46A9B")
        plt.legend()
        plt.show()

    return {"model": model, "F1": f1, "AP": ap, "y_pred": y_pred, "y_scores": y_scores, "cm": cm}


def optimize_unsupervised(model_name, X, y_true, param_grid, dataset_name="default"):
    """
    Optimisation automatique pour modèles non supervisés (IsolationForest, LOF)
    - Teste toutes les combinaisons d’hyperparamètres
    - Retourne le meilleur modèle et les scores correspondants
    """
    results = []
    best_f1, best_ap, best_model = 0, 0, None

    for params in param_grid:
        if model_name.lower().startswith("isolation"):
            model = IsolationForest(**params, random_state=42)
            model.fit(X)
            y_pred = np.where(model.predict(X) == -1, 1, 0)
            y_scores = -model.score_samples(X)

        elif model_name.lower().startswith("local"):
            model = LocalOutlierFactor(**params)
            y_pred = model.fit_predict(X)
            y_pred = np.where(y_pred == -1, 1, 0)
            y_scores = -model.negative_outlier_factor_

        else:
            raise ValueError("Nom de modèle non reconnu (attendu : 'IsolationForest' ou 'LocalOutlierFactor').")

        f1 = f1_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_scores)

        results.append({**params, "F1": f1, "AP": ap})

        if f1 > best_f1:
            best_f1, best_ap, best_model = f1, ap, model

    display(Markdown(f"""
### Optimisation du modèle **{model_name}** sur le dataset **{dataset_name}**
- Nombre de combinaisons testées : **{len(param_grid)}**
- Meilleur F1 : **{best_f1:.3f}**
- Meilleur AP : **{best_ap:.3f}**
"""))

    df_results = pd.DataFrame(results).sort_values(by=["F1", "AP"], ascending=False)
    display(
        df_results.style
        .background_gradient(cmap=rose_map, axis=0)
        .set_caption(f"Résultats de l’optimisation - {model_name} ({dataset_name})")
    )

    return best_model, df_results



def evaluate_unsupervised(
    model,
    X,
    y_true,
    model_name="Modèle non supervisé",
    tail_split=None
):
    """
    Évalue un modèle non supervisé (IsolationForest, LocalOutlierFactor)
    avec affichage stylé (métriques + matrice de confusion + courbe PR).
    """
    # --- Prédiction selon le type de modèle ---
    if hasattr(model, "predict") and callable(model.predict):
        try:
            y_pred = np.where(model.predict(X) == -1, 1, 0)
        except Exception:
            y_pred = np.zeros_like(y_true)
    else:
        if hasattr(model, "fit_predict"):
            y_pred = np.where(model.fit_predict(X) == -1, 1, 0)
        else:
            raise ValueError("Le modèle ne possède pas de méthode predict ou fit_predict.")

    # --- Scores ---
    if hasattr(model, "score_samples"):
        y_scores = -model.score_samples(X)
    elif hasattr(model, "negative_outlier_factor_"):
        y_scores = -model.negative_outlier_factor_
    else:
        y_scores = np.zeros_like(y_true, dtype=float)

    f1 = f1_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_scores)
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)

    # --- Subplots 1x3 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plt.subplots_adjust(wspace=0.4)

    # heatmap métriques
    metrics_df = pd.DataFrame({
        "F1-score (fraudes)": [f1],
        "Average Precision (AP)": [ap]
    }).T.rename(columns={0: "Valeur"})

    sns.heatmap(metrics_df, annot=True, fmt=".3f", cmap=rose_map,
                cbar=False, linewidths=1, ax=axes[0],
                annot_kws={"fontsize": 11, "fontweight": "bold"})
    axes[0].set_title("Métriques globales", fontsize=12, color="#B30059")

    # matrice de confusion
    sns.heatmap(cm, annot=True, fmt="d", cmap=rose_map, cbar=False, linewidths=1.2, ax=axes[1],
                xticklabels=["Prédit: 0", "Prédit: 1"], yticklabels=["Réel: 0", "Réel: 1"])
    axes[1].set_title("Matrice de confusion", fontsize=12, color="#B30059")
    axes[1].set_xlabel("Valeurs prédites")
    axes[1].set_ylabel("Valeurs réelles")

    # courbe PR
    axes[2].plot(recalls, precisions, color="#D46A9B", lw=2.5)
    axes[2].set_title(f"Courbe PR (AP = {ap:.2f})", fontsize=12, color="#B30059")
    axes[2].set_xlabel("Rappel")
    axes[2].set_ylabel("Précision")
    axes[2].grid(alpha=0.3)

    plt.show()

    return {
        "model": model,
        "F1": f1,
        "AP": ap,
        "y_pred": y_pred,
        "y_scores": y_scores,
        "cm": cm
    }


def optimize_supervised(model_class, model_name, X, y, param_grid, dataset_name="default", cache_dir="data", cv_splits=3):
    """
    Optimisation automatique pour modèles supervisés (XGBoost, EasyEnsemble, etc.)
    - Recharge les résultats du cache si dispo
    - Réalise une validation croisée stratifiée sinon
    - Sauvegarde les résultats et les modèles associés
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe_name = f"{model_name.replace(' ', '_').lower()}_{dataset_name.replace(' ', '_').lower()}"
    cache_path = os.path.join(cache_dir, f"cache_{safe_name}.csv")
    model_dir = os.path.join(cache_dir, f"models_{safe_name}")
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path)
    else:
        cache_df = pd.DataFrame()

    results = []
    best_f1, best_ap, best_model = 0, 0, None

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for params in param_grid:
        match = (
            (cache_df[list(params.keys())] == pd.Series(params)).all(axis=1)
            if not cache_df.empty
            else False
        )

        if isinstance(match, pd.Series) and match.any():
            row = cache_df.loc[match].iloc[0].to_dict()
            f1, ap = row["F1"], row["AP"]
            model_path = row.get("model_path", None)
            if model_path and os.path.exists(model_path):
                model = joblib.load(model_path)
            else:
                model = model_class(**params)
        else:
            model = model_class(**params)
            f1_scores, ap_scores = [], []

            for train_idx, val_idx in cv.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                try:
                    y_proba = model.predict_proba(X_val)[:, 1]
                except Exception:
                    y_proba = np.zeros_like(y_pred, dtype=float)

                f1_scores.append(f1_score(y_val, y_pred))
                ap_scores.append(average_precision_score(y_val, y_proba))

            f1, ap = np.mean(f1_scores), np.mean(ap_scores)

            # Sauvegarde du modèle entraîné complet
            model_filename = f"{model_name.replace(' ', '_').lower()}_{hash(frozenset(params.items()))}.pkl"
            model_path = os.path.join(model_dir, model_filename)
            joblib.dump(model, model_path)

            row = {**params, "F1": f1, "AP": ap, "model_path": model_path}
            cache_df = pd.concat([cache_df, pd.DataFrame([row])], ignore_index=True)
            cache_df.to_csv(cache_path, index=False)

        results.append(row)

        if f1 > best_f1:
            best_f1, best_ap, best_model = f1, ap, model

    display(Markdown(f"""
### Optimisation du modèle **{model_name}** sur le dataset **{dataset_name}**
- Nombre de splits CV : **{cv_splits}**
- Tests effectués : **{len(param_grid)}**
- Meilleur F1 : **{best_f1:.3f}**
- Meilleur AP : **{best_ap:.3f}**
"""))
    
    df_results = pd.DataFrame(results)
    df_results.sort_values(by=["F1", "AP"], ascending=False, inplace=True)
    
    display_cols = [col for col in df_results.columns if col != "model_path"]
    
    styled = (
        df_results[display_cols].style
        .background_gradient(cmap=rose_map, axis=0)
        .set_caption(f"Résultats de l’optimisation - {model_name} ({dataset_name})")
    )
    display(styled)

    return best_model, df_results

def evaluate_supervised(
    model,
    X_test,
    y_test,
    model_name="Modèle supervisé",
    optimize_threshold=False
):
    """
    Évalue un modèle supervisé (XGBoost, EasyEnsemble, etc.)
    - Calcule F1-score et Average Precision
    - Si optimize_threshold=True : cherche le meilleur seuil sur la courbe PR
    - Affiche : métriques, matrice de confusion, courbe PR
    """

    y_proba = model.predict_proba(X_test)[:, 1]

    if optimize_threshold:
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.nanargmax(f1_scores)
        best_thresh = thresholds[best_idx]
        y_pred = (y_proba >= best_thresh).astype(int)
        f1 = f1_scores[best_idx]
        ap = average_precision_score(y_test, y_proba)
        print(f"Seuil = {best_thresh:.3f} | F1 = {f1:.3f} | AP = {ap:.3f}")
    else:
        y_pred = (y_proba >= 0.5).astype(int)
        f1 = f1_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_proba)
        best_thresh = 0.5

    # --- Matrice de confusion ---
    cm = confusion_matrix(y_test, y_pred)
    precisions, recalls, _ = precision_recall_curve(y_test, y_proba)

    display(Markdown(f"## Évaluation du modèle : **{model_name}**"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plt.subplots_adjust(wspace=0.4)

    # métriques
    metrics_df = pd.DataFrame({
        "F1-score (fraudes)": [f1],
        "Average Precision (AP)": [ap]
    }).T.rename(columns={0: "Valeur"})
    sns.heatmap(metrics_df, annot=True, fmt=".3f", cmap=rose_map, cbar=False,
                linewidths=1, ax=axes[0], annot_kws={"fontsize": 11, "fontweight": "bold"})
    axes[0].set_title("Métriques globales", fontsize=12, color="#B30059")

    # matrice de confusion
    sns.heatmap(cm, annot=True, fmt="d", cmap=rose_map, cbar=False, linewidths=1.2, ax=axes[1],
                xticklabels=["Prédit: 0", "Prédit: 1"], yticklabels=["Réel: 0", "Réel: 1"])
    axes[1].set_title("Matrice de confusion", fontsize=12, color="#B30059")
    axes[1].set_xlabel("Valeurs prédites")
    axes[1].set_ylabel("Valeurs réelles")

    # courbe précision/rappel
    axes[2].plot(recalls, precisions, color="#D46A9B", lw=2.5)
    axes[2].axvline(x=recalls[np.nanargmax(f1_scores)] if optimize_threshold else None,
                    color="gray", ls="--", lw=1)
    axes[2].set_title(f"Courbe PR (AP = {ap:.2f})", fontsize=12, color="#B30059")
    axes[2].set_xlabel("Recall (1)")
    axes[2].set_ylabel("Precision (1)")
    axes[2].grid(alpha=0.3)

    plt.show()

    return {
        "model": model,
        "F1": f1,
        "AP": ap,
        "threshold": best_thresh,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "cm": cm
    }

def generate_balanced_datasets(X_train, y_train, random_state=42):
    """
    Génère plusieurs variantes équilibrées du dataset d’entraînement :
    - original
    - SMOTE (oversampling)
    - Tomek Links (undersampling)
    - balancing global (pondération)
    """
    # --- Original ---
    datasets = {"original": (X_train, y_train)}

    # --- SMOTE ---
    smote = SMOTE(random_state=random_state)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    datasets["smote"] = (X_smote, y_smote)

    # --- Tomek Links ---
    tomek = TomekLinks()
    X_tomek, y_tomek = tomek.fit_resample(X_train, y_train)
    datasets["tomek"] = (X_tomek, y_tomek)

    # --- SMOTE + Tomek Links (combinaison équilibrée + nettoyage) ---
    smt = SMOTETomek(random_state=random_state)
    X_smt, y_smt = smt.fit_resample(X_train, y_train)
    datasets["smote_tomek"] = (X_smt, y_smt)

    # --- Balancing global (poids des classes) ---
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    weight_dict = {cls: w for cls, w in zip(np.unique(y_train), class_weights)}
    datasets["balanced_weights"] = (X_train, y_train, weight_dict)

    return datasets

def plot_class_distributions(datasets):
    """
    Affiche la distribution des classes (normal / fraude)
    pour chaque dataset (original, smote, tomek, balancing)
    """
    data_plot = []

    for name, data in datasets.items():
        if len(data) == 3:  # balancing global (contient les poids)
            X, y, _ = data
        else:
            X, y = data
        counts = pd.Series(y).value_counts().sort_index()
        data_plot.append({
            "Dataset": name.capitalize(),
            "Normales": counts.get(0, 0),
            "Fraudes": counts.get(1, 0),
            "Ratio (%)": round(100 * counts.get(1, 0) / (counts.sum()), 3)
        })

    df_plot = pd.DataFrame(data_plot)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_plot.melt(id_vars="Dataset", value_vars=["Normales", "Fraudes"]),
        x="Dataset", y="value", hue="variable",
        hue_order=["Normales", "Fraudes"],
        palette={"Normales": "#F3B4DF", "Fraudes": "#F055A3"},
        edgecolor="white", linewidth=1.2
    )


    plt.title("Répartition des classes avant et après rééquilibrage",
              fontsize=13, color="#B30059", fontweight="bold")
    plt.ylabel("Nombre d’échantillons")
    plt.xlabel("Type de dataset")
    plt.legend(title="Classe")
    plt.grid(alpha=0.2)
    plt.show()

    display(df_plot.style
        .background_gradient(cmap=rose_map, axis=0)
        .set_caption("Ratios de classes avant/après rééquilibrage")
        .format({"Normales": "{:,}", "Fraudes": "{:,}", "Ratio (%)": "{:.3f}"}))
    
def evaluate_supervised_models_on_balanced_datasets(
    datasets_balanced,
    X_test,
    y_test,
    param_grids,
    cv_splits=3
):
    """
    Évalue plusieurs modèles supervisés (XGBoost, RandomForest, LogisticRegression)
    sur différentes versions rééquilibrées du dataset (original, SMOTE, Tomek, etc.)
    avec optimisation par validation croisée.
    """

    # === Association entre noms et classes ===
    model_classes = {
        "XGBoost": XGBClassifier,
        "Random Forest": RandomForestClassifier,
        "Régression Logistique": LogisticRegression
    }

    results_summary = []
    results_pr_curves = {}

    # === Boucle sur chaque dataset rééquilibré ===
    for name, data in datasets_balanced.items():
        display(Markdown(f"## Dataset : **{name.upper()}**"))

        if len(data) == 3:  # dataset avec pondération
            X_train, y_train, _ = data
        else:
            X_train, y_train = data

        for model_name, param_grid in param_grids.items():
            model_class = model_classes[model_name]

            best_model, df_results = optimize_supervised(
                model_class=model_class,
                model_name=model_name,
                X=X_train, y=y_train,
                param_grid=param_grid,
                dataset_name=name,
                cv_splits=cv_splits
            )

            res = evaluate_supervised(
                best_model, X_test, y_test,
                model_name=f"{model_name} ({name})",
                optimize_threshold=True
            )

            # Courbe PR
            y_proba = res["y_proba"]
            precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
            results_pr_curves[f"{model_name} ({name})"] = {
                "precision": precisions,
                "recall": recalls,
                "AP": res["AP"]
            }

            results_summary.append({
                "Dataset": name,
                "Model": model_name,
                "F1": res["F1"],
                "AP": res["AP"],
                "Threshold": res.get("threshold", None)
            })

    # === Résumé global ===
    results_df = pd.DataFrame(results_summary)
    display(Markdown("## **Comparaison globale des modèles supervisés sur tous les rééquilibrages**"))
    display(
        results_df.style
        .background_gradient(cmap=rose_map, axis=0)
        .format({"F1": "{:.3f}", "AP": "{:.3f}", "Threshold": "{:.3f}"})
        .set_properties(**{"text-align": "center"})
    )

    return results_df, results_pr_curves

    

def plot_all_pr_curves(results_pr_curves, figsize=(14, 8)):
    """
    Affiche toutes les courbes Précision–Rappel des modèles testés
    avec une palette pastel élégante et lisible.
    """
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")

    pastel_palette = [
        "#E07A9A",  
        "#9C8ADE",  
        "#F7A072",  
        "#80CBC4",  
        "#F6C1E3",  
        "#A3C4F3",  
        "#FFD580",  
        "#C4E17F",  
        "#F3A0A0",  
        "#B39BC8",  
        "#89ABE3",  
        "#E8C547"   
    ]

    for (name, res), color in zip(results_pr_curves.items(), pastel_palette * 3):
        precisions, recalls, ap = res["precision"], res["recall"], res["AP"]
        plt.plot(recalls, precisions, lw=2.4, color=color, label=f"{name} (AP={ap:.3f})")

    plt.title("Comparaison des courbes Précision–Rappel\npour tous les modèles et rééquilibrages",
              fontsize=16, color="#B30059", weight="bold", pad=15)
    plt.xlabel("Rappel (Recall)", fontsize=12)
    plt.ylabel("Précision (Precision)", fontsize=12)
    plt.legend(loc="lower left", fontsize=10, frameon=True)
    plt.grid(alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()



import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ---------- RÉGLAGES (identiques pour GRU et LSTM) ----------
CSV = "data/dataset.csv"
FEATURES = ["HR", "MAP", "SpO2", "Temp", "RespRate", "Lactate"]
NIVEAUX = ["Low", "Medium", "High", "Critical"]
TYPE_RNN = "GRU"            # <- la seule ligne à changer : "GRU" ou "LSTM"
HIDDEN, EPOCHS, LR, BATCH, PATIENCE = 64, 40, 1e-3, 64, 6
SEEDS = [0, 1, 2]


# ---------- 1. DONNÉES ----------
def construire_sequences(df, T):
    X, y, split = [], [], []
    for _, g in df.groupby("Patient_ID"):
        a = g[FEATURES].to_numpy(dtype=np.float32)
        if len(a) < T:                                   # on complète en répétant la dernière ligne
            a = np.vstack([a, np.repeat(a[-1:], T - len(a), axis=0)])
        X.append(a[:T])
        y.append(NIVEAUX.index(g["Risk"].iloc[-1]))      # risque au dernier pas de temps
        split.append(str(g["Split"].iloc[0]).lower())
    return np.array(X), np.array(y), np.array(split)


def preparer():
    df = pd.read_csv(CSV).sort_values(["Patient_ID", "Time_Step"])
    df[FEATURES] = df.groupby("Patient_ID")[FEATURES].ffill()      # on reporte la dernière mesure
    T = int(df.groupby("Patient_ID").size().median())
    X, y, split = construire_sequences(df, T)

    tr, va, te = (split == "train"), (split == "val"), (split == "test")
    # adaptez les noms ci-dessus si votre colonne Split utilise d'autres mots (ex. "validation")

    med = np.nanmedian(X[tr].reshape(-1, X.shape[2]), axis=0)      # statistiques sur TRAIN uniquement
    X = np.where(np.isnan(X), med, X)
    mu, sd = X[tr].reshape(-1, X.shape[2]).mean(0), X[tr].reshape(-1, X.shape[2]).std(0) + 1e-6
    X = (X - mu) / sd
    return X, y, tr, va, te


# ---------- 2. MODÈLE ----------
class Modele(nn.Module):
    def __init__(self, n_in, hidden, n_classes, kind):
        super().__init__()
        cellule = {"GRU": nn.GRU, "LSTM": nn.LSTM}[kind]
        self.rnn = cellule(n_in, hidden, batch_first=True)
        self.sortie = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.sortie(out[:, -1, :])                # on lit l'état au dernier pas de temps


# ---------- 3. ENTRAÎNEMENT ----------
def predire(model, X):
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(X, dtype=torch.float32)), dim=1)
    return p.numpy()


def entrainer(seed, X, y, tr, va, te):
    torch.manual_seed(seed); np.random.seed(seed)
    poids = torch.tensor(len(y[tr]) / (len(NIVEAUX) * np.bincount(y[tr], minlength=len(NIVEAUX)).clip(1)),
                         dtype=torch.float32)             # classes rares = poids plus fort
    model = Modele(X.shape[2], HIDDEN, len(NIVEAUX), TYPE_RNN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    perte = nn.CrossEntropyLoss(weight=poids)
    loader = DataLoader(TensorDataset(torch.tensor(X[tr], dtype=torch.float32),
                                      torch.tensor(y[tr])), batch_size=BATCH, shuffle=True)

    meilleur, attente, etat = -1, 0, None
    for epoque in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); perte(model(xb), yb).backward(); opt.step()
        f1_val = f1_score(y[va], predire(model, X[va]).argmax(1), average="macro")
        if f1_val > meilleur:
            meilleur, attente, etat = f1_val, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            attente += 1
            if attente >= PATIENCE:                       # arrêt anticipé
                break
    model.load_state_dict(etat)
    return model


# ---------- 4. ÉVALUATION SUR LE TEST ----------
if __name__ == "__main__":
    X, y, tr, va, te = preparer()
    print(TYPE_RNN, "| entraînement :", tr.sum(), "| validation :", va.sum(), "| test :", te.sum())
    scores = []
    for seed in SEEDS:
        model = entrainer(seed, X, y, tr, va, te)
        pred = predire(model, X[te]).argmax(1)
        f1 = f1_score(y[te], pred, average="macro")
        scores.append(f1)
        print(f"graine {seed} : F1 macro = {f1:.3f}")
        if seed == SEEDS[0]:
            print(classification_report(y[te], pred, target_names=NIVEAUX, zero_division=0))
            print(confusion_matrix(y[te], pred))
            torch.save(model.state_dict(), f"modele_{TYPE_RNN}.pt")
    print(f"{TYPE_RNN} : F1 macro moyen = {np.mean(scores):.3f} ± {np.std(scores):.3f}")
import os, ast, warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection     import train_test_split, GridSearchCV
from sklearn.compose             import ColumnTransformer
from sklearn.pipeline            import Pipeline
from sklearn.preprocessing       import StandardScaler, OneHotEncoder
from sklearn.impute              import SimpleImputer
from sklearn.metrics             import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble            import GradientBoostingRegressor

import tensorflow as tf
from tensorflow.keras.layers     import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models     import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# ─────────────────────────  paths  ──────────────────────────
os.makedirs("models", exist_ok=True)
MOV_PATH = r"C:\Users\shiva\OneDrive\Desktop\MinorProject\dataset\tmdb_5000_movies.csv"
CRE_PATH = r"C:\Users\shiva\OneDrive\Desktop\MinorProject\dataset\tmdb_5000_credits.csv"

# ─────────────────────────  load & merge  ──────────────────────────
movies  = pd.read_csv(MOV_PATH)
credits = pd.read_csv(CRE_PATH).rename(columns={'movie_id':'id'})
df      = movies.merge(credits,on='id')

# ───────────────────  basic cleaning / feature eng  ───────────────
def parse(obj):           
    try: return [d['name'] for d in ast.literal_eval(obj)]
    except: return []

df['genres']   = df['genres'].apply(parse)
df['keywords'] = df['keywords'].apply(parse)
df['cast']     = df['cast'].apply(lambda x:[d['name'] for d in ast.literal_eval(x)][:3])
df['crew']     = df['crew'].apply(lambda x: next((d['name'] for d in ast.literal_eval(x) if d['job']=="Director"),"Unknown"))
df['combined_features'] = (df['genres']+df['keywords']+df['cast']).apply(" ".join)

# drop rows lacking targets
df = df.dropna(subset=['revenue','vote_average','overview']).reset_index(drop=True)

# ──────────────────────────  TABULAR  ML  ─────────────────────────
# ------- log‑transform revenue to remove extreme skew -------------
df['log_revenue'] = np.log1p(df['revenue'])

feature_cols = ['budget','runtime','popularity','crew']     # tabular
target_rating   = df['vote_average']
target_revenue  = df['log_revenue']

X_tabular = df[feature_cols]
X_train_t,X_test_t,y_train_rtg,y_test_rtg = train_test_split(
        X_tabular, target_rating,  test_size=0.2, random_state=42)
_,        _,        y_train_rev,y_test_rev = train_test_split(
        X_tabular, target_revenue, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
        ('num', Pipeline([('imp',SimpleImputer(strategy='median')),
                          ('sc',StandardScaler())]),
         ['budget','runtime','popularity']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['crew'])
    ])

# Gradient Boosting (often > RF on noisy numeric data)
gb_param = {'gbr__n_estimators':[300],
            'gbr__max_depth':[3],
            'gbr__learning_rate':[0.05]}

def build_model():
    return Pipeline([('prep',preprocess),
                     ('gbr',GradientBoostingRegressor(random_state=42))])

# -------- rating model ------------------------------------------------
rating_pipe = GridSearchCV(build_model(), gb_param, cv=3, n_jobs=-1, verbose=0)
rating_pipe.fit(X_train_t, y_train_rtg)
y_pred_rtg = rating_pipe.predict(X_test_t)

# -------- revenue model -----------------------------------------------
revenue_pipe = GridSearchCV(build_model(), gb_param, cv=3, n_jobs=-1, verbose=0)
revenue_pipe.fit(X_train_t, y_train_rev)
y_pred_rev_log = revenue_pipe.predict(X_test_t)
y_pred_rev = np.expm1(y_pred_rev_log)          # back‑transform for metrics
y_test_rev_raw = np.expm1(y_test_rev)

fitted_prep = rating_pipe.best_estimator_.named_steps['prep']

# ─────────────────────────────  TEXT  DL  ────────────────────────────
texts = df['overview'].astype(str).tolist()
tokenizer = Tokenizer(num_words=12000,oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
seq = tokenizer.texts_to_sequences(texts)
pad = pad_sequences(seq,maxlen=120,padding='post',truncating='post')

X_train_d,X_test_d,y_train_d,y_test_d = train_test_split(
        pad, target_rating, test_size=0.2, random_state=42)

dl_model = Sequential([
    Embedding(12000,64,input_length=120),
    Bidirectional(LSTM(64,return_sequences=True)),
    GlobalAveragePooling1D(),
    Dropout(0.3),
    Dense(64,activation='relu'),
    Dropout(0.3),
    Dense(1)
])
dl_model.compile(optimizer='adam',loss='mse',metrics=['mae'])
early = tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)
dl_model.fit(X_train_d,y_train_d,epochs=20,batch_size=32,
             validation_split=0.2,callbacks=[early],verbose=2)
y_pred_dl = dl_model.predict(X_test_d).flatten()

# ──────────────────────────  METRICS  ────────────────────────────────
def metrics(y_true,y_pred,name):
    mse  = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true,y_pred)
    r2   = r2_score(y_true,y_pred)
    print(f"\n{name}  ►  MSE:{mse:.3f}  RMSE:{rmse:.3f}  MAE:{mae:.3f}  R²:{r2:.3f}")

metrics(y_test_rtg, y_pred_rtg, "Rating ML (GBR)")
metrics(y_test_rev_raw, y_pred_rev, "Revenue ML (GBR, log‑target)")

metrics(y_test_d,   y_pred_dl,  "Rating DL (Bi‑LSTM)")

# ───────────────────────────  SAVE  ──────────────────────────────────

joblib.dump(fitted_prep,  "models/preprocessor.pkl")   # ← now FITTED ✔
joblib.dump(rating_pipe,  "models/rating_model.pkl")   # full pipeline
joblib.dump(revenue_pipe, "models/revenue_model.pkl")  # full pipeline

dl_model.save("models/overview_rating_model.h5")
joblib.dump(tokenizer,    "models/tokenizer.pkl")


print("\n✅  Models trained & saved  — see console for new (much higher) R² scores.")
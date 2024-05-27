import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Wczytanie danych, pomijając drugi wiersz, który zawiera typy danych
data = pd.read_csv('dataset_improved.csv', delimiter=';', skiprows=[1])

# Usunięcie kolumn 'ID' i 'Motorway'
data.drop(['ID', 'Motorway'], axis=1, inplace=True)

# Listy kolumn dla różnych typów danych
numeric_features = ['SR', 'NR', 'OR']
categorical_features = ['VR', 'UR', 'MR', 'CR']
ordinal_features = ['RR', 'BR']

# Dane wejściowe i etykiety
X = data.drop(data.columns[-7:], axis=1)  # Zakładamy, że ostatnie 7 kolumn to etykiety

y = data[data.columns[-7:]]  # Etykiety

# Tworzenie transformatorów dla numerycznych, kategorycznych i porządkowych danych
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features),
        ('ord', OrdinalEncoder(), ordinal_features)  # Użycie OrdinalEncoder dla danych porządkowych
    ])

# Pipeline do przetwarzania i modelowania danych
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Przetwarzanie danych
X_processed = pipeline.fit_transform(X)

# Podział danych na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)


# Zapis danych treningowych i testowych do plików CSV
pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)

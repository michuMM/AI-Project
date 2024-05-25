import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Przykładowe dane
data = pd.DataFrame({
    'kolor': ['czerwony', 'zielony', 'niebieski', 'zielony', 'czerwony']
})

# Tworzenie instancji OneHotEncoder
encoder = OneHotEncoder(sparse=False)

# Dopasowanie i transformacja danych
encoded_data = encoder.fit_transform(data[['kolor']])

# Tworzenie DataFrame z zakodowanych danych
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['kolor']))

print("Zakodowane dane:")
print(encoded_df)

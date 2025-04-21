# README: Kundenwertanalyse (CLTV) mit Python

Dieses Projekt implementiert eine vollständige CLTV (Customer Lifetime Value) Analyse für das Unternehmen FLO unter Verwendung des BG/NBD- und Gamma-Gamma-Modells. Die Analyse erfolgt mit Hilfe der Bibliothek `lifetimes` und basiert auf realistischen Kundendaten.

## 📂 Verwendete Daten
- **Datei:** `flo_data_20k.csv`
- **Kundenattribute:** Bestellanzahl online/offline, Umsatz online/offline, erste/letzte Bestellung

## 🔄 Projektstruktur
### 1. **Datenvorbereitung**
- Entfernen von Ausreißern in Bestellanzahl und Umsatz
- Erzeugung von Gesamtwerten pro Kunde:
  - `order_num_total`
  - `customer_value_total`

### 2. **Zeitbasierte Transformation**
- `recency_cltv_weekly`: Wochen seit letzter Bestellung
- `T_weekly`: Wochen seit erster Bestellung bis Analysezeitpunkt (01.06.2021)

### 3. **BG/NBD-Modell (BetaGeoFitter)**
- Prognose zukünftiger Käufe:
  - `exp_sales_3_month`
  - `exp_sales_6_month`

### 4. **Gamma-Gamma-Modell (GammaGammaFitter)**
- Vorhersage des durchschnittlichen Bestellwerts je Kunde:
  - `exp_average_value`

### 5. **CLTV-Berechnung**
- Monetärer Wert jedes Kunden für einen Zeitraum von 6 Monaten:
  - `cltv`

### 6. **Segmentierung**
- Einteilung aller Kunden in 4 Segmente (A bis D) nach ihrem CLTV:
  - `cltv_segment`

## 🚀 Anforderungen
- Python 3.8+
- pandas
- lifetimes
- scikit-learn

## ⚙️ Nutzung
```bash
pip install pandas scikit-learn lifetimes
```

```python
from cltv_analysis_de import create_cltv_df
import pandas as pd

df = pd.read_csv("flo_data_20k.csv")
cltv_df = create_cltv_df(df)
print(cltv_df.head())
```

## 🌍 Autor
- Ali Bektasoglu  
- [GitHub-Profil](https://github.com/Ali623462)

---
> Dieses Projekt basiert auf dem Trainingsmaterial des Data Science Bootcamps von Miuul und wurde erweitert und dokumentiert für Bewerbungs- und Praxiszwecke.

# flo_CLTV
Cltv_Analysis
# cltv_analysis_de.py
# Kundenwertanalyse (Customer Lifetime Value - CLTV) mit BG/NBD und Gamma-Gamma Modell
# Autor: Ali Bektasoglu
# Quelle der Daten: flo_data_20k.csv

import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter, GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.options.mode.chained_assignment = None

# Funktion zur Erkennung und Begrenzung von Ausreißern
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = round(quartile3 + 1.5 * interquantile_range)
    low_limit = round(quartile1 - 1.5 * interquantile_range)
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit

# Hauptfunktion zur Erstellung der CLTV-Tabelle
def create_cltv_df(dataframe):
    # Schritt 1: Datenvorbereitung
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline",
               "customer_value_total_ever_offline", "customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = dataframe[(dataframe["order_num_total"] > 1)]

    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # Schritt 2: CLTV-Struktur erstellen
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).dt.days) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).dt.days) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]

    # Schritt 3: BG/NBD Modell (Kaufhäufigkeit)
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])
    cltv_df["exp_sales_3_month"] = bgf.predict(12, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])
    cltv_df["exp_sales_6_month"] = bgf.predict(24, cltv_df["frequency"], cltv_df["recency_cltv_weekly"], cltv_df["T_weekly"])

    # Schritt 4: Gamma-Gamma Modell (Durchschnittlicher Gewinn)
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["monetary_cltv_avg"])

    # Schritt 5: CLTV-Berechnung (6 Monate)
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df["frequency"],
                                       cltv_df["recency_cltv_weekly"],
                                       cltv_df["T_weekly"],
                                       cltv_df["monetary_cltv_avg"],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # Schritt 6: CLTV-Segmentierung
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

# Beispielnutzung:
df = pd.read_csv("flo_data_20k.csv")
cltv_df = create_cltv_df(df)
print(cltv_df.head())

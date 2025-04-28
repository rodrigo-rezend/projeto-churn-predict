from google.cloud import bigquery
import pandas as pd

def fetch_bigquery_data():
    # Cria o cliente do BigQuery
    client = bigquery.Client()

    # Query SQL
    query = """
      SELECT
        fullVisitorId,
        visitId,
        date,
        device.deviceCategory AS device,
        device.operatingSystem AS os,
        geoNetwork.country AS country,
        trafficSource.medium AS traffic_medium,
        trafficSource.source AS traffic_source,
        totals.pageviews,
        totals.timeOnSite,
        totals.transactions,
        totals.transactionRevenue
      FROM
        `bigquery-public-data.google_analytics_sample.ga_sessions_*`
      WHERE
        _TABLE_SUFFIX BETWEEN '20160801' AND '20180131'
        AND totals.pageviews IS NOT NULL
    """

    # Executa a query
    query_job = client.query(query)

    # Resultado
    df = query_job.to_dataframe()

    # Salva em CSV
    df.to_csv("data/ga_sessions_sample.csv", index=False)
    print("Arquivo salvo em: data/ga_sessions_sample.csv")

if __name__ == "__main__":
    fetch_bigquery_data()

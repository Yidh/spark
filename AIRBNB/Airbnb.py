from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, concat, format_number, lit, avg, count, when, sum

def initialize_spark(app_name):
    return SparkSession.builder.appName(app_name).getOrCreate()

def read_data(spark, file_path):
    return spark.read.option("header", True) \
        .option("inferSchema", True) \
        .option("multiLine", True) \
        .option("escape", "\"") \
        .csv(file_path)

def clean_price_column(df):
    return df.withColumn("price", regexp_replace("price", r"[\$,£€]", "").cast("float"))

def convert_columns(df):
    cols_to_convert = ["number_of_reviews", "minimum_nights_avg_ntm", "reviews_per_month"]
    for col_name in cols_to_convert:
        df = df.withColumn(col_name, col(col_name).cast("float"))
    return df

def group_by_room_type(df):
    return df.groupBy("room_type").agg(count("*").alias("count"))

def calculate_room_type_percentage(df_room_count, total_listings):
    return df_room_count.withColumn("percentage", (col("count") / total_listings * 100)) \
        .withColumn("percentage_label", concat(format_number("percentage", 1), lit("%")))

def calculate_average_price(df, total_listings):
    total_price_sum = df.agg(sum("price").alias("total_price_sum")).first()["total_price_sum"]
    average_price_per_night = total_price_sum / total_listings
    return average_price_per_night

def calculate_occupancy_and_income(df):
    df = df.withColumn("estimated_nights_booked",
                       (col("number_of_reviews") * col("minimum_nights_avg_ntm") / 2))

    df = df.withColumn("estimated_income", col("estimated_nights_booked") * col("price"))

    average_nights_booked = df.agg(avg("estimated_nights_booked").alias("average_nights_booked")).first()[
        "average_nights_booked"]
    average_income = df.agg(avg("estimated_income").alias("average_income")).first()["average_income"]

    return average_nights_booked, average_income

def calculate_short_term_rentals_percentage(df):
    short_term_rentals = df.filter(col("minimum_nights") <= 30)
    longer_term_rentals = df.filter(col("minimum_nights") > 30)

    total_listings = df.count()
    short_term_rentals_count = short_term_rentals.count()
    longer_term_rentals_count = longer_term_rentals.count()

    short_term_rentals_percentage = (short_term_rentals_count / total_listings) * 100
    longer_term_rentals_percentage = (longer_term_rentals_count / total_listings) * 100

    return short_term_rentals_count, longer_term_rentals_count, short_term_rentals_percentage, longer_term_rentals_percentage

def calculate_multi_listings_percentage(df):
    multi_listings = df.filter(col("calculated_host_listings_count") > 1)
    single_listings = df.filter(col("calculated_host_listings_count") == 1)

    total_listings = df.count()
    multi_listings_count = multi_listings.count()
    single_listings_count = single_listings.count()

    multi_listings_percentage = (multi_listings_count / total_listings) * 100
    single_listings_percentage = (single_listings_count / total_listings) * 100

    return single_listings_count, multi_listings_count, single_listings_percentage, multi_listings_percentage

def calculate_top_hosts(df):
    df_top_hosts = df.groupBy("host_name").agg(
        count(when(col("room_type") == "Entire home/apt", True)).alias("entire_home_apts"),
        count(when(col("room_type") == "Private room", True)).alias("private_rooms"),
        count(when(col("room_type") == "Shared room", True)).alias("shared_rooms"),
        count(when(col("room_type") == "Hotel room", True)).alias("hotel_rooms"),
        count("*").alias("total_listings")
    ).orderBy(col("total_listings").desc())

    return df_top_hosts

def display_results(df_room_percentage, average_price, average_nights_booked, average_income,
                    short_term_rentals_count, longer_term_rentals_count, short_term_rentals_percentage,
                    longer_term_rentals_percentage, single_listings_count, multi_listings_count,
                    single_listings_percentage, multi_listings_percentage, df_top_hosts):
    df_room_percentage.show(truncate=False)
    print(f"Average price per night: {average_price:.2f}")
    print(f"Moyenne des nuits réservées : {average_nights_booked:.1f}")
    print(f"Revenu moyen estimé : {average_income:.0f} £\n")

    print(f"{short_term_rentals_percentage:.1f}%\nshort-term rentals")
    print(f"{short_term_rentals_count} ({short_term_rentals_percentage:.1f}%)\nshort-term rentals")
    print(f"{longer_term_rentals_count} ({longer_term_rentals_percentage:.1f}%)\nlonger-term rentals\n")


    print(f"Listings per Host")
    print(f"{multi_listings_percentage:.1f}%")
    print(f"{single_listings_count} ({single_listings_percentage:.1f}%)\nsingle listings")
    print(f"{multi_listings_count} ({multi_listings_percentage:.1f}%)\nmulti-listings\n")

    print(f"Top Hosts")
    df_top_hosts.show(truncate=False)

def main():
    spark = initialize_spark('Airbnb Data Processing')
    file_path = "listings.csv"

    df = read_data(spark, file_path)
    df = clean_price_column(df)
    df = convert_columns(df)
    #df.printSchema()

    df_room_count = group_by_room_type(df)
    total_listings = df.count()
    print("Le nombre de lignes du DataFrame est :", total_listings)

    df_room_percentage = calculate_room_type_percentage(df_room_count, total_listings)
    average_price = calculate_average_price(df, total_listings)
    average_nights_booked, average_income = calculate_occupancy_and_income(df)
    short_term_rentals_count, longer_term_rentals_count, short_term_rentals_percentage, longer_term_rentals_percentage = calculate_short_term_rentals_percentage(df)
    single_listings_count, multi_listings_count, single_listings_percentage, multi_listings_percentage = calculate_multi_listings_percentage(df)
    df_top_hosts = calculate_top_hosts(df)

    display_results(df_room_percentage, average_price, average_nights_booked, average_income,
                    short_term_rentals_count, longer_term_rentals_count, short_term_rentals_percentage,
                    longer_term_rentals_percentage, single_listings_count, multi_listings_count,
                    single_listings_percentage, multi_listings_percentage, df_top_hosts)

    spark.stop()

if __name__ == "__main__":
    main()

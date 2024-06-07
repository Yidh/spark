from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, round, lower, to_date, expr

def main():

    spark = SparkSession.builder.appName("TrumpInsultsAnalysis").getOrCreate()


    df = spark.read.csv("trump_insult_tweets_2014_to_2021.csv", header=True, inferSchema=True)


    df.printSchema()

    # Calculer le nombre total d'insultes
    total_insults = df.count()

    # PARTIE 1 : Afficher les 7 comptes les plus insultés
    top_7_insulted_targets = get_top_insulted_targets(df, total_insults, 7)
    print("Les 7 comptes les plus insultés par Donald Trump:")
    top_7_insulted_targets.show()

    # PARTIE 2 : Afficher les insultes les plus utilisées
    insult_usage_sorted = get_insult_usage(df, total_insults)
    print("Les insultes les plus utilisées par Donald Trump:")
    insult_usage_sorted.show()

    # PARTIE 3 : Insulte la plus utilisée pour Joe Biden
    top_biden_insult = get_top_insult_for_target(df, "biden")
    print("L'insulte la plus utilisée par Donald Trump pour Joe Biden:")
    top_biden_insult.show()

    # PARTIE 5 : Classer le nombre de tweets par période de 6 mois
    tweets_by_period = get_tweets_by_period(df)
    print("Nombre de tweets par période de 6 mois:")
    tweets_by_period.show()

    # PARTIE 4 : Comptage des mots spécifiques "Mexico", "China", "coronavirus"
    count_specific_words(df, ["mexico", "china", "coronavirus"])

    # Arrêter la session Spark
    spark.stop()

def get_top_insulted_targets(df, total_insults, top_n):
    return (df.groupBy("target")
              .agg(count("insult").alias("count"))
              .withColumn("percentage", round((col("count") / total_insults) * 100, 2))
              .orderBy(col("count").desc())
              .limit(top_n))

def get_insult_usage(df, total_insults):
    return (df.groupBy("insult")
              .agg(count("insult").alias("count"))
              .withColumn("percentage", round((col("count") / total_insults) * 100, 2))
              .orderBy(col("count").desc()))

def get_top_insult_for_target(df, target_name):
    return (df.filter(lower(col("target")).like(f"%{target_name.lower()}%"))
              .groupBy("insult")
              .agg(count("insult").alias("count"))
              .orderBy(col("count").desc())
              .limit(1))

def get_tweets_by_period(df):
    df = df.withColumn("date", to_date(col("date"), "yyyy-MM-dd"))
    df = df.withColumn("period", expr("""
        case 
            when month(date) between 1 and 6 then concat(year(date), '-S1')
            else concat(year(date), '-S2')
        end
    """))
    return (df.groupBy("period")
              .agg(count("*").alias("tweet_count"))
              .orderBy("period"))

def count_specific_words(df, words):
    df = df.withColumn("tweet_lower", lower(col("tweet")))
    for word in words:
        word_count = df.filter(col("tweet_lower").contains(word)).count()
        print(f"Le mot '{word}' a été tweeté {word_count} fois.")

if __name__ == "__main__":
    main()

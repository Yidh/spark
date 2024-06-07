from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, split, col, regexp_extract, avg, floor, concat_ws

# Initialiser une session Spark
spark = SparkSession.builder.appName("Netflix Analysis").getOrCreate()

# Charger le fichier CSV dans un DataFrame PySpark
file_path = "netflix_titles.csv"
df = spark.read.csv(file_path, header=True, inferSchema=True)


# Fonction pour afficher les réalisateurs les plus prolifiques
def prolific_directors(df):
    df_with_directors = df.withColumn("director", explode(split(col("director"), r",\s*")))
    directors_count = df_with_directors.groupBy("director").count().orderBy(col("count").desc())
    directors_count.show()


# Fonction pour afficher les pourcentages des pays de production
def country_percentages(df):
    df_with_countries = df.withColumn("country", explode(split(col("country"), r",\s*")))
    countries_count = df_with_countries.groupBy("country").count()
    total_count = df_with_countries.count()
    countries_percentage = countries_count.withColumn("percentage", (col("count") / total_count) * 100)
    countries_percentage = countries_percentage.orderBy(col("percentage").desc())
    countries_percentage.show()


# Fonction pour calculer la durée moyenne des films, le plus long et le plus court
def movies_duration(df):
    movies_df = df.filter(df.type == "Movie")
    movies_df = movies_df.withColumn("duration_minutes", regexp_extract(col("duration"), r"(\d+)", 1).cast("int"))

    average_duration = movies_df.agg(avg("duration_minutes").alias("average_duration")).collect()[0]["average_duration"]
    longest_movie = movies_df.orderBy(col("duration_minutes").desc()).select("title", "duration_minutes").first()
    shortest_movie = movies_df.orderBy(col("duration_minutes").asc()).select("title", "duration_minutes").first()

    print(f"Durée moyenne des films : {average_duration} minutes")
    print(f"Film le plus long : {longest_movie['title']} avec {longest_movie['duration_minutes']} minutes")
    print(f"Film le plus court : {shortest_movie['title']} avec {shortest_movie['duration_minutes']} minutes")


# Fonction pour afficher la durée moyenne des films par intervalles de 2 ans
def average_duration_by_interval(df):
    movies_df = df.filter(df.type == "Movie")
    movies_df = movies_df.withColumn("duration_minutes", regexp_extract(col("duration"), r"(\d+)", 1).cast("int"))
    movies_df = movies_df.withColumn("year_interval", (floor((col("release_year") - 1) / 2) * 2 + 1))
    avg_duration_interval = movies_df.groupBy("year_interval").agg(
        avg("duration_minutes").alias("average_duration")).orderBy("year_interval", ascending=False)
    avg_duration_interval.show(truncate=False)


# Fonction pour trouver le duo réalisateur-acteur le plus prolifique
def prolific_director_actor_duo(df):
    df_with_directors = df.withColumn("director", explode(split(col("director"), r",\s*")))
    df_with_actors = df_with_directors.withColumn("actor", explode(split(col("cast"), r",\s*")))
    df_director_actor_pairs = df_with_actors.withColumn("director_actor_pair",
                                                        concat_ws(" - ", col("director"), col("actor")))
    director_actor_count = df_director_actor_pairs.groupBy("director_actor_pair").count().orderBy(col("count").desc())
    top_duo = director_actor_count.first()

    print(
        f"Le duo réalisateur-acteur avec le plus de collaborations est {top_duo['director_actor_pair']} avec {top_duo['count']} films.")


# Exécution des fonctions
prolific_directors(df)
country_percentages(df)
movies_duration(df)
average_duration_by_interval(df)
prolific_director_actor_duo(df)

# Arrêter la session Spark
spark.stop()

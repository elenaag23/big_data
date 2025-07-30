import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import expr, size, col, when, lit
from pyspark.sql.functions import avg, stddev, count


def print_confusion_matrix(predictions, model_name, label_to_genre):
    pred_labels = predictions.select("label", "prediction")
    confusion_matrix = pred_labels.groupBy("label", "prediction").count().orderBy("label", "prediction")

    labels = sorted(label_to_genre.keys())
    confusion = confusion_matrix.collect()
    conf_dict = {(row['label'], row['prediction']): row['count'] for row in confusion}

    print(f"\nMatrice de confuzie pentru {model_name}")
    print(" " * 18 + "\t".join([label_to_genre[l] for l in labels]))
    for actual in labels:
        row_counts = [str(conf_dict.get((actual, pred), 0)) for pred in labels]
        print(f"{label_to_genre[actual]:18}:\t" + "\t".join(row_counts))


spark = SparkSession.builder.appName("MusicClassification_CV").getOrCreate()

df = spark.read.csv("features/genre_features.csv", header=True, inferSchema=True).dropna()
df.createOrReplaceTempView("audio_raw")

sql_df = spark.sql("""
    SELECT *,
           tempo * zero_crossing_rate AS rhythmic_complexity,
           chroma * spectral_centroid AS harmonic_density,
           CASE WHEN bandwidth != 0 THEN spectral_centroid / bandwidth ELSE 0.0 END AS brightness_score
    FROM audio_raw
""")

mfcc_cols = [f"mfcc{i}" for i in range(1, 14)]
sql_df = sql_df.withColumn("mfccs_array", expr(f"array({', '.join(mfcc_cols)})"))
sql_df = sql_df.withColumn("mfcc_energy", expr("aggregate(mfccs_array, 0D, (acc, x) -> acc + x) / size(mfccs_array)"))
sql_df = sql_df.withColumn("percussive_ratio", when(
    col("mfcc_energy") != 0, col("rhythmic_complexity") / col("mfcc_energy")
).otherwise(lit(0.0)))

df_proc = sql_df.drop("mfccs_array")

agg_df = df_proc.groupBy("genre").agg(
    count("*").alias("num_tracks"),
    avg("tempo").alias("avg_tempo"),
    avg("mfcc_energy").alias("avg_mfcc_energy"),
    stddev("rhythmic_complexity").alias("std_rhythmic_complexity"),
    avg("brightness_score").alias("avg_brightness"),
    avg("harmonic_density").alias("avg_harmonic_density")
)

print("Afisarea rezultatelor agregate")
agg_df.show(truncate=False)

indexer = StringIndexer(inputCol="genre", outputCol="label")
df_indexed = indexer.fit(df_proc).transform(df_proc)

label_to_genre = {i: genre for i, genre in enumerate(indexer.fit(df_proc).labels)}

feature_cols = [
    "tempo", "spectral_centroid", "zero_crossing_rate", "rmse", "bandwidth", "chroma",
    "mfcc_energy", "rhythmic_complexity", "harmonic_density", "brightness_score", "percussive_ratio",
    "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5", "mfcc6", "mfcc7", "mfcc8", "mfcc9", "mfcc10", "mfcc11", "mfcc12", "mfcc13",
    "beat_strength", "tempo_variation", "onset_density"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
scaler = StandardScaler(inputCol="features_vec", outputCol="features", withMean=True, withStd=True)

train_data, test_data = df_indexed.randomSplit([0.8, 0.2], seed=42)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

rf = RandomForestClassifier(featuresCol="features", labelCol="label")
rf_pipeline = Pipeline(stages=[assembler, scaler, rf])

rf_paramGrid = ParamGridBuilder() \
    .addGrid(rf_pipeline.getStages()[2].numTrees, [100, 150]) \
    .addGrid(rf_pipeline.getStages()[2].maxDepth, [5, 10]) \
    .addGrid(rf_pipeline.getStages()[2].maxBins, [32]) \
    .addGrid(rf_pipeline.getStages()[2].featureSubsetStrategy, ['auto', 'sqrt']) \
    .build()

rf_cv = CrossValidator(
    estimator=rf_pipeline,
    estimatorParamMaps=rf_paramGrid,
    evaluator=evaluator,
    numFolds=5
)

rf_cv_model = rf_cv.fit(train_data)
rf_preds = rf_cv_model.transform(test_data)
rf_acc = evaluator.evaluate(rf_preds)

df_binary = df_indexed.withColumn("label_disco", when(col("genre") == "disco", 1).otherwise(0))

gtb_label_to_genre = {0: "not_disco", 1: "disco"}

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
scaler = StandardScaler(inputCol="features_vec", outputCol="features", withMean=True, withStd=True)

gbt = GBTClassifier(featuresCol="features", labelCol="label_disco", maxIter=100)
gbt_pipeline = Pipeline(stages=[assembler, scaler, gbt])

gbt_train_data, gbt_test_data = df_binary.randomSplit([0.8, 0.2], seed=42)

gbt_paramGrid = ParamGridBuilder() \
                .addGrid(gbt.maxDepth, [2]) \
                .addGrid(gbt.maxIter, [50]) \
                .build()

gb_evaluator = MulticlassClassificationEvaluator(labelCol="label_disco")

gbt_cv = CrossValidator(
    estimator=gbt_pipeline,
    estimatorParamMaps=gbt_paramGrid,
    evaluator=gb_evaluator,
    numFolds=5
)

gbt_cv_model = gbt_cv.fit(gbt_train_data)
gbt_predictions = gbt_cv_model.transform(gbt_test_data)
gbt_acc = gb_evaluator.evaluate(gbt_predictions)

print(f" Random Forest Accuracy:       {rf_acc:.3f}")
print(f" GBTClassifier  Accuracy: {gbt_acc:.3f}")

print(" Matrice confuzie RF:")
print_confusion_matrix(rf_preds, "Random Forest", label_to_genre)

# print("Matrice confuzie GBT:")
# print_confusion_matrix(gbt_predictions, "GBTClassifier", gtb_label_to_genre)

final_df = rf_cv_model.bestModel.transform(df_indexed)
pandas_df = final_df.select("features", "label").toPandas()

import numpy as np

X = np.array(pandas_df["features"].tolist())  # features = list de vectori
y = pandas_df["label"].values

pd.DataFrame(X).to_csv("features.csv", index=False)
pd.DataFrame(y, columns=["label"]).to_csv("labels.csv", index=False)




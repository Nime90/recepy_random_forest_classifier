import pandas as pd
import numpy as np
import pickle
import umap.umap_ as umap
import warnings
from google.cloud import bigquery
from dotenv import load_dotenv
from google.auth import default
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
load_dotenv()

# Suppress UMAP warning about random_state overriding n_jobs
warnings.filterwarnings('ignore', message='.*n_jobs value.*overridden.*', category=UserWarning)

def reduce_embendings(embeddings_matrix,name_col):

    # Convert embedding column (list of floats) into separate columns
    emb_table = pd.DataFrame(
        embeddings_matrix['embedding'].tolist(),
        index=embeddings_matrix.index
        )
    # Add column names for each embedding dimension
    emb_table.columns = [f'emb_{i}' for i in range(emb_table.shape[1])]
    emb_table_to_reduce = emb_table.copy()
    # Add recipeId column
    emb_table['recipeId'] = embeddings_matrix['recipeId'].values

    # Reorder columns to have recipeId first
    emb_table = emb_table[['recipeId'] + [col for col in emb_table.columns if col != 'recipeId']]
        # Step 3.2: UMAP Dimensionality Reduction
    reducer = umap.UMAP(
        n_components=2,
        random_state=42, # for reproducible results
        n_neighbors=5,   # Focus on tighter, local clusters
        min_dist=0.3     # Slightly spread points for visual clarity
        )

    projected_embeddings = reducer.fit_transform(emb_table_to_reduce)
    
    umap_col1 = []
    umap_col2 = []
    for umap_ in projected_embeddings: 
        umap_col1.append(umap_[0])
        umap_col2.append(umap_[1])
    emb_table[f'{name_col}_UMAP_col1'] = umap_col1
    emb_table[f'{name_col}_UMAP_col2'] = umap_col2

    emb_table = emb_table[['recipeId', f'{name_col}_UMAP_col1', f'{name_col}_UMAP_col2']]

    return emb_table

def reduce_embeddings_pca(embeddings_matrix,name_col):
    """
    Performs Principal Component Analysis (PCA) dimensionality reduction
    on a matrix of embeddings.

    Args:
        embeddings_matrix (np.ndarray): A 2D numpy array where rows are
                                        embeddings and columns are features.

    Returns:
        np.ndarray: A 2D numpy array of the embeddings projected onto
                    the top 2 principal components.
    """
    # Convert embedding column (list of floats) into separate columns
    emb_table = pd.DataFrame(
        embeddings_matrix['embedding'].tolist(),
        index=embeddings_matrix.index
        )
    # Add column names for each embedding dimension
    emb_table.columns = [f'emb_{i}' for i in range(emb_table.shape[1])]
    emb_table_to_reduce = emb_table.copy()
    # Add recipeId column
    emb_table['recipeId'] = embeddings_matrix['recipeId'].values

    # Reorder columns to have recipeId first
    emb_table = emb_table[['recipeId'] + [col for col in emb_table.columns if col != 'recipeId']]
        # Step 3.2: UMAP Dimensionality Reduction
    pca = PCA(
        n_components=2,
        random_state=42 # for reproducible results in randomized solvers (though deterministic by default)
    )


    projected_embeddings = pca.fit_transform(emb_table_to_reduce)
    
    umap_col1 = []
    umap_col2 = []
    for umap_ in projected_embeddings: 
        umap_col1.append(umap_[0])
        umap_col2.append(umap_[1])
    emb_table[f'{name_col}_pca_col1'] = umap_col1
    emb_table[f'{name_col}_pca_col2'] = umap_col2

    emb_table = emb_table[['recipeId', f'{name_col}_pca_col1', f'{name_col}_pca_col2']]

    return emb_table

def import_training_data():
    # Run a SQL query
    query = f''' 
    WITH FLAGGED as (
      SELECT
        recipeId, recipeName, recipeTags, recipeIngredients, procedure, protein, carbohydrate, fat, minimumCalories, maximumCalories, prepareTimeInMinutes,
        CASE WHEN cookingTimeInMinutes is NULL THEN 0 ELSE cookingTimeInMinutes END as cookingTimeInMinutes,
        CASE WHEN porridge = 1 THEN 0 WHEN shake = 1 THEN 1 WHEN smoothie = 1 THEN 2 ELSE 3 END as flag
      FROM
        `dbt_nime.meal_recipes_flag_us_v2`
        where reason is not null
        )
      ,imgs as (
        select
        meal_id as recipeId,
        concat('https://eu.lenus.io/admin/meals/meals-live/bulk-edit/', meal_variation_group_id, '#', meal_id) as adminLink
        from bi-lenus-prod.dbt_staging.stg_appdb_snapshot__meal
        where meal_id in (select recipeId from FLAGGED)
        )
      select
        f.recipeId,
        f.recipeName,
        f.recipeTags,
        f.recipeIngredients,
        f.procedure,
        f.protein,      
        f.carbohydrate,
        f.fat,
        f.minimumCalories,
        f.maximumCalories,
        f.prepareTimeInMinutes,
        f.cookingTimeInMinutes,
        f.flag,
        i.adminLink
      from FLAGGED f
      left join imgs i
      on f.recipeId = i.recipeId
    '''
    df_query = client_bq.query(query).to_dataframe()
    return df_query

credentials, project_id = default()
client_bq = bigquery.Client(credentials=credentials,project=project_id )

recipeIngredients_emb = pd.read_parquet('emb_data/recipeIngredients_emb.parquet')
recipeName_emb = pd.read_parquet('emb_data/recipeName_emb.parquet')
recipeProcedure_emb = pd.read_parquet('emb_data/procedure_emb.parquet')
recipeTags_emb = pd.read_parquet('emb_data/recipeTags_emb.parquet')

recipeIngredients_UMAP = reduce_embendings(recipeIngredients_emb,'ingredients')
recipeName_UMAP = reduce_embendings(recipeName_emb,'rec_name')
recipeProcedure_UMAP = reduce_embendings(recipeProcedure_emb,'procedure')
recipeTags_UMAP = reduce_embendings(recipeTags_emb,'tags')
recipeIngredients_PCA = reduce_embeddings_pca(recipeIngredients_emb,'ingredients')
recipeName_PCA = reduce_embeddings_pca(recipeName_emb,'rec_name')
recipeProcedure_PCA = reduce_embeddings_pca(recipeProcedure_emb,'procedure')
recipeTags_PCA = reduce_embeddings_pca(recipeTags_emb,'tags')

training_data = import_training_data() 
training_data = pd.merge(training_data, recipeIngredients_UMAP, on='recipeId', how='left')
training_data = pd.merge(training_data, recipeName_UMAP, on='recipeId', how='left')
training_data = pd.merge(training_data, recipeProcedure_UMAP, on='recipeId', how='left')
training_data = pd.merge(training_data, recipeTags_UMAP, on='recipeId', how='left')
training_data = pd.merge(training_data, recipeIngredients_PCA, on='recipeId', how='left')
training_data = pd.merge(training_data, recipeName_PCA, on='recipeId', how='left')
training_data = pd.merge(training_data, recipeProcedure_PCA, on='recipeId', how='left')
training_data = pd.merge(training_data, recipeTags_PCA, on='recipeId', how='left')  



df_training = training_data[[
  'protein', 'carbohydrate', 'fat', 'minimumCalories', 'maximumCalories', 'prepareTimeInMinutes', 'cookingTimeInMinutes',
  'rec_name_UMAP_col1', 'rec_name_UMAP_col2', 'rec_name_pca_col1', 'rec_name_pca_col2',
  'ingredients_UMAP_col1', 'ingredients_UMAP_col2', 'ingredients_pca_col1', 'ingredients_pca_col2',
  'tags_UMAP_col1', 'tags_UMAP_col2', 'tags_pca_col1', 'tags_pca_col2',
  'procedure_UMAP_col1','procedure_UMAP_col2', 'procedure_pca_col1', 'procedure_pca_col2',
  'flag'
  ]]

target_column = 'flag'
X = df_training.drop(columns=[target_column])
y = df_training[target_column]
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state=42, stratify=y )
print("Training samples:",X_train.shape, "Testing samples: ",  X_test.shape)

model = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_estimators=2000, # <-- TUNE THIS
    max_depth=None,  # <-- TUNE THIS
    min_samples_leaf=1 # <-- AND THIS
)

print("Training the Random Forest model (with class_weight='balanced')...")
# Train the model on the training data
model.fit(X_train, y_train)
print("Model training complete.\n")

# --- 5. Evaluate Model ---
print("--- Model Evaluation on Test Set ---")

# Get predictions on the unseen test data
y_pred = model.predict(X_test)

# 5a. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5b. Confusion Matrix
print("\nConfusion Matrix:")
print("(Rows = Actual, Columns = Predicted)")
print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]))

unique, counts = np.unique(y_pred, return_counts=True)
print(dict(zip(unique, counts)))

unique, counts = np.unique(y_test, return_counts=True)
print(dict(zip(unique, counts)))


with open('models/random_forest_recipes_model_us_v2_2k_obs.pkl', 'wb') as f: pickle.dump(model, f)
###-------------------------------------LABEL NEW DATA-------------------------------------
from google.cloud import bigquery
from dotenv import load_dotenv
from google.auth import default
import pandas as pd, numpy as np
from google.genai import types
from google import genai
from tqdm import tqdm
import os, pickle
import umap.umap_ as umap
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated libraries
try:
    import cupy as cp
    from cuml import UMAP as cuUMAP
    from cuml.decomposition import PCA as cuPCA
    GPU_AVAILABLE = True
    print("GPU libraries loaded successfully. Using GPU acceleration.")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU libraries not available. Falling back to CPU.")
    cp = None

load_dotenv()

credentials, project_id = default()
project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
client_bq = bigquery.Client(credentials=credentials,project=project_id )
client_gemini = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

def import_new_data(limit = 1000):
  query = f"""
    with new_data as (
    SELECT
        recipeId,recipeName,recipeTags,recipeIngredients,procedure,protein, carbohydrate, fat, minimumCalories, maximumCalories, prepareTimeInMinutes,
        case when cookingTimeInMinutes is NULL then 0 else cookingTimeInMinutes end as cookingTimeInMinutes,
        FROM `bi-lenus-staging.dbt_nime.sot_meal_recipes` r
        where r.language = 'en-US' and r.owner is NULL and r.recipeId not in (select distinct recipeId from `bi-lenus-staging.dbt_nime.meal_recipes_flag_us_v2`)
        LIMIT {limit}
        )
        ,img as (
        select
            meal_id as recipeId,
            concat('https://eu.lenus.io/admin/meals/meals-live/bulk-edit/', meal_variation_group_id, '#', meal_id) as adminLink
            from bi-lenus-prod.dbt_staging.stg_appdb_snapshot__meal
            )
        select 
            new_data.*,
            img.adminLink
            from new_data
            left join img
            on new_data.recipeId = img.recipeId
        """
  df = client_bq.query(query).to_dataframe()
  print('data imported correctly')
  return df

def get_gemini_embeddings(texts: list[str]) -> np.ndarray:
    """
    Generates text embeddings using the Gemini API.

    Args:
        texts: A list of strings (documents or queries) to embed.

    Returns:
        A NumPy array where each row is an embedding vector.
    """


    # The recommended embedding model
    EMBEDDING_MODEL = 'gemini-embedding-001'

    # We specify the task_type as CLUSTERING since our goal is visualization
    # and clustering of similar texts on the UMAP plot.
    config = types.EmbedContentConfig(task_type="CLUSTERING")

    #print(f"Generating embeddings for {len(texts)} texts using {EMBEDDING_MODEL}...")

    # The API call to generate embeddings
    response = client_gemini.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=config
    )

    # Extract the vector values and convert to a NumPy array
    embeddings_list = [e.values for e in response.embeddings]
    return np.array(embeddings_list)

def txt_embender_nd(training_data, col = 'recipeName', save_emb = False):
    texts_emb = []
    recipe_ids = []

    for idx in tqdm(range(len(training_data)), desc=f"Processing {col}"):
        text = training_data.iloc[idx][col]
        recipe_id = training_data.iloc[idx]['recipeId']
        
        # Convert list/array to string if needed (check this first before pd.isna)
        if isinstance(text, (list, tuple, np.ndarray)):
            try:
                if len(text) > 0:
                    text = ", ".join(str(item) for item in text)
                else:
                    text = ""
            except (TypeError, AttributeError):
                text = str(text)
        # Handle None/NaN values (after checking for arrays)
        elif pd.isna(text):
            text = ""
        # Ensure it's a string
        else:
            text = str(text)
        
        text_emb = get_gemini_embeddings(texts=[text])  # Pass as list
        texts_emb.append(text_emb[0].tolist())  # Convert numpy array to list
        recipe_ids.append(recipe_id)
    
    df_emb = pd.DataFrame({'recipeId': recipe_ids, 'embedding': texts_emb})
    
    if save_emb==True:
        df_emb.to_parquet(f'emb_data/{col}_emb.parquet')
    
    return df_emb

def reduce_embeddings(embeddings_matrix,name_col):

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
    
    # Step 3.2: UMAP Dimensionality Reduction (GPU-accelerated if available)
    if GPU_AVAILABLE:
        # Convert to GPU array
        emb_array_gpu = cp.asarray(emb_table_to_reduce.values, dtype=cp.float32)
        
        # Use GPU-accelerated UMAP
        reducer = cuUMAP(
            n_components=2,
            random_state=42, # for reproducible results
            n_neighbors=5,   # Focus on tighter, local clusters
            min_dist=0.3     # Slightly spread points for visual clarity
        )
        
        projected_embeddings = reducer.fit_transform(emb_array_gpu)
        
        # Convert back to CPU numpy array
        projected_embeddings = cp.asnumpy(projected_embeddings)
    else:
        # Fallback to CPU UMAP
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
    on a matrix of embeddings using GPU acceleration if available.

    Args:
        embeddings_matrix (pd.DataFrame): DataFrame with 'embedding' column containing lists of floats
        name_col (str): Name prefix for output columns

    Returns:
        pd.DataFrame: DataFrame with recipeId and PCA-reduced columns
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
    
    # Step 3.2: PCA Dimensionality Reduction (GPU-accelerated if available)
    if GPU_AVAILABLE:
        # Convert to GPU array
        emb_array_gpu = cp.asarray(emb_table_to_reduce.values, dtype=cp.float32)
        
        # Use GPU-accelerated PCA
        pca = cuPCA(
            n_components=2,
            random_state=42 # for reproducible results in randomized solvers
        )
        
        projected_embeddings = pca.fit_transform(emb_array_gpu)
        
        # Convert back to CPU numpy array
        projected_embeddings = cp.asnumpy(projected_embeddings)
    else:
        # Fallback to CPU PCA
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

def write_data(df, table_id, if_exists='replace'):

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        if if_exists == 'replace' else bigquery.WriteDisposition.WRITE_APPEND
    )

    # Start the load job
    job = client_bq.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # Wait for the job to complete

    print(f"Data successfully written to {table_id}")


#import training embeddings
recipeIngredients_emb = pd.read_parquet('emb_data/recipeIngredients_emb.parquet')
recipeName_emb = pd.read_parquet('emb_data/recipeName_emb.parquet')
recipeProcedure_emb = pd.read_parquet('emb_data/procedure_emb.parquet')
recipeTags_emb = pd.read_parquet('emb_data/recipeTags_emb.parquet')
print('training embeddings imported')

#import new data
new_data = import_new_data(limit = 1)
data_redu = pd.DataFrame()
start_index = 0
batch_size = 10
end_index = start_index + batch_size

#loop through new data in batches to create embeddings and reduce them with UMAP and PCA
for i in range(0, len(new_data), batch_size): 
    print(f'Processing batch {i} to {i + batch_size}')
    new_data_t = new_data[start_index:end_index]

    recipe_name_emb = txt_embender_nd(col = 'recipeName', training_data = new_data_t, save_emb = False)
    recipe_tags_emb = txt_embender_nd(col = 'recipeTags', training_data = new_data_t, save_emb = False)
    recipe_ingredients_emb = txt_embender_nd(col = 'recipeIngredients', training_data = new_data_t, save_emb = False)
    recipe_procedure_emb = txt_embender_nd(col = 'procedure', training_data = new_data_t, save_emb = False)
    print(f'Embeddings generated for batch {start_index} to {end_index}')


    #stuck new data on top of training embeddings
    recipeIngredients_emb_full = pd.concat([recipeIngredients_emb, recipe_ingredients_emb])
    recipeName_emb_full = pd.concat([recipeName_emb, recipe_name_emb])
    recipeProcedure_emb_full = pd.concat([recipeProcedure_emb, recipe_procedure_emb])
    recipeTags_emb_full = pd.concat([recipeTags_emb, recipe_tags_emb])

    recipeIngredients_UMAP = reduce_embeddings(recipeIngredients_emb_full,'ingredients')
    recipeName_UMAP = reduce_embeddings(recipeName_emb_full,'rec_name')
    recipeProcedure_UMAP = reduce_embeddings(recipeProcedure_emb_full,'procedure')
    recipeTags_UMAP = reduce_embeddings(recipeTags_emb_full,'tags')
    recipeIngredients_PCA = reduce_embeddings_pca(recipeIngredients_emb_full,'ingredients')
    recipeName_PCA = reduce_embeddings_pca(recipeName_emb_full,'rec_name')
    recipeProcedure_PCA = reduce_embeddings_pca(recipeProcedure_emb_full,'procedure')
    recipeTags_PCA = reduce_embeddings_pca(recipeTags_emb_full,'tags')

    print(f'Embeddings reduced for batch {start_index} to {end_index}')

    new_data_t = pd.merge(new_data_t, recipeIngredients_UMAP, on='recipeId', how='left')
    new_data_t = pd.merge(new_data_t, recipeName_UMAP, on='recipeId', how='left')
    new_data_t = pd.merge(new_data_t, recipeProcedure_UMAP, on='recipeId', how='left')
    new_data_t = pd.merge(new_data_t, recipeTags_UMAP, on='recipeId', how='left')
    new_data_t = pd.merge(new_data_t, recipeIngredients_PCA, on='recipeId', how='left')
    new_data_t = pd.merge(new_data_t, recipeName_PCA, on='recipeId', how='left')
    new_data_t = pd.merge(new_data_t, recipeProcedure_PCA, on='recipeId', how='left')
    new_data_t = pd.merge(new_data_t, recipeTags_PCA, on='recipeId', how='left')  

    data_redu = pd.concat([data_redu, new_data_t])
    print(f'Data reduced for batch {start_index} to {end_index}')
    start_index = end_index
    end_index = start_index + batch_size


df_nd = data_redu[[
  'protein', 'carbohydrate', 'fat', 'minimumCalories', 'maximumCalories', 'prepareTimeInMinutes', 'cookingTimeInMinutes',
  'rec_name_UMAP_col1', 'rec_name_UMAP_col2', 'rec_name_pca_col1', 'rec_name_pca_col2',
  'ingredients_UMAP_col1', 'ingredients_UMAP_col2', 'ingredients_pca_col1', 'ingredients_pca_col2',
  'tags_UMAP_col1', 'tags_UMAP_col2', 'tags_pca_col1', 'tags_pca_col2',
  'procedure_UMAP_col1','procedure_UMAP_col2', 'procedure_pca_col1', 'procedure_pca_col2',
  ]]

with open('models/random_forest_recipes_model_us_v2_2k_obs.pkl', 'rb') as f: model = pickle.load(f)
print('model loaded')

flag_pred = model.predict(df_nd)

print('recipeId: ', new_data.iloc[0]['recipeId'])
print('recipeName: ', new_data.iloc[0]['recipeName'])
print('recipeTags: ', new_data.iloc[0]['recipeTags'])
print('recipeIngredients: ', new_data.iloc[0]['recipeIngredients'])
print('protein: ', new_data.iloc[0]['protein'])
print('carbohydrate: ', new_data.iloc[0]['carbohydrate'])
print('fat: ', new_data.iloc[0]['fat'])
print('minimumCalories: ', new_data.iloc[0]['minimumCalories'])
print('maximumCalories: ', new_data.iloc[0]['maximumCalories'])
print('prepareTimeInMinutes: ', new_data.iloc[0]['prepareTimeInMinutes'])
print('cookingTimeInMinutes: ', new_data.iloc[0]['cookingTimeInMinutes'])
print('flag predicted: ', flag_pred)


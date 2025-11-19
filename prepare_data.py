import os
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from google.cloud import bigquery
from google.cloud import bigquery
from google.auth import default
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

credentials, project_id = default()
client_bq = bigquery.Client(credentials=credentials,project=project_id )
client_gemini = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

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

def txt_embender(training_data, col = 'recipeName'):
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
    df_emb.to_parquet(f'emb_data/{col}_emb.parquet')
    return df_emb

training_data = import_training_data()
recipe_name_emb = txt_embender(training_data, col = 'recipeName')
recipe_tags_emb = txt_embender(training_data, col = 'recipeTags')
recipe_ingredients_emb = txt_embender(training_data, col = 'recipeIngredients')
recipe_procedure_emb = txt_embender(training_data, col = 'procedure')
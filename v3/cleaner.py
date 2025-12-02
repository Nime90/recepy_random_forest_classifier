#this cleaner is created to clean the porridge, the smoothies and the shakes after they have been categorized by the model
from google.cloud import bigquery
from dotenv import load_dotenv
from google.auth import default
import os, time, pandas as pd
import warnings
from google import genai
from tqdm import tqdm
warnings.filterwarnings('ignore')

load_dotenv()

credentials, project_id = default()
client_bq = bigquery.Client(credentials=credentials,project=project_id )
client_gemini = genai.Client(api_key = os.getenv("GEMINI_API_KEY"))

def import_flaged_data():
    query = f"""
    SELECT recipeId, 
    concat(
        'Recipe Name: ',recipeName,'\\n'
        'Recipe Tags: ',ARRAY_TO_STRING(recipeTags, ', '),'\\n'
        'Recipe Ingredients: ',ARRAY_TO_STRING(recipeIngredients, ', '),'\\n'
        'Procedure: ',procedure) as recipe_description,
    case when porridge = 1 then 'porridge'
        when smoothie = 1 then 'smoothie'
        when shake = 1 then 'shake'
        else 'Other'
    end as original_flag

    FROM `bi-lenus-staging.dbt_nime.meal_recipes_flag_us_v3`
    where (smoothie=1 or shake=1 or porridge=1) and recipeId not in (select distinct recipeId from  `bi-lenus-staging.dbt_nime.meal_recipes_flag_us_v3_accuracy_gemini`)
    """
    df = client_bq.query(query).to_dataframe()
    return df

def cleaner(recipeId,recipe_description,flag):
    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents= f'''The provided recipe have been flagged as {flag}. Please carefully read the provided recipe and flag the recipe as porridge, smoothie, shake or other. 
        Only return the correct flag for the recipe. No intro nor explanation is needed. 
        ###Output###
        return a single word between porridge, smoothie, shake or other.
        ###Recipe###
        Recipe: {recipe_description}'''
        )
        
    df = pd.DataFrame()
    df['recipeId'] = [recipeId]
    df['new_flag'] = str(response.text).replace('```json','').replace('```','')
    
    return df

def write_data(df, table_id, if_exists='append'):

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        if if_exists == 'replace' else bigquery.WriteDisposition.WRITE_APPEND
    )

    # Start the load job
    job = client_bq.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # Wait for the job to complete

    print(f"Data successfully written to {table_id}")

df = import_flaged_data()

start_batch = 0
batch = 100
end_batch = start_batch + batch

for round in tqdm(range(int(df.shape[0]/batch)+1)):
    try:
        df_round = df[start_batch:end_batch]
        df_round.reset_index(drop=True,inplace=True)
        df_cleaned = pd.DataFrame()
        for i in tqdm(range(df_round.shape[0])):
            df_t = cleaner(recipeId=df_round.recipeId[i],recipe_description=df_round.recipe_description[i],flag=df_round.original_flag[i])
            df_cleaned = pd.concat([df_cleaned,df_t]).reset_index(drop=True)

        df_cleaned['new_flag'] = df_cleaned['new_flag'].str.lower()
        df_new = pd.merge(df_round,df_cleaned,on='recipeId',how='left')
        df_new = df_new.drop_duplicates().reset_index(drop=True)
        write_data(df_new, table_id = f'bi-lenus-staging.dbt_nime.meal_recipes_flag_us_v3_accuracy_gemini', if_exists='append')
        start_batch = end_batch
        end_batch = start_batch + batch
    except Exception as e:
        print(e)
        time.sleep(10)
        continue

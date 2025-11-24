###-------------------------------------STREAMLIT APP - CREATE ONE RECIPE-------------------------------------
import streamlit as st
from google.cloud import bigquery
from google.genai import types
from dotenv import load_dotenv
from google.auth import default
import pandas as pd
import numpy as np
import os
import uuid
import json
from google import genai
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Recipe Creator",
    page_icon="🍳",
    layout="wide"
)

# Initialize session state
if 'recipe_generated' not in st.session_state:
    st.session_state.recipe_generated = False
if 'recipe_df' not in st.session_state:
    st.session_state.recipe_df = None
if 'recipe_examples' not in st.session_state:
    st.session_state.recipe_examples = None

# Load environment variables
load_dotenv()

# Initialize clients
@st.cache_resource
def init_clients():
    credentials, project_id = default()
    client_bq = bigquery.Client(credentials=credentials, project=project_id)
    client_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return client_bq, client_gemini, project_id

try:
    client_bq, client_gemini, project_id = init_clients()
except Exception as e:
    st.error(f"Error initializing clients: {str(e)}")
    st.stop()

def import_recipe_examples(limit=20):
    query = f"""
        with 
        porridges as (SELECT * FROM `bi-lenus-staging.dbt_nime.meal_recipes_flag_us_v2` where reason is not null and porridge =1 and recipeId not in ('513f9904-d807-44a7-a434-45fe2204e140','6322bfde-a8c3-4296-99f2-d503a9e65476') limit {limit}),
        shakes    as (SELECT * FROM `bi-lenus-staging.dbt_nime.meal_recipes_flag_us_v2` where reason is not null and shake =1 limit {limit}),
        smoothies as (SELECT * FROM `bi-lenus-staging.dbt_nime.meal_recipes_flag_us_v2` where reason is not null and smoothie =1 and recipeId not in ('79add589-bfb0-4fec-9541-b5db2dcf6216') limit {limit}),
        other     as (SELECT * FROM `bi-lenus-staging.dbt_nime.meal_recipes_flag_us_v2` where reason is not null and smoothie =0 and porridge =0 and shake =0 limit {limit})
        select * from porridges union all select * from shakes union all select * from smoothies union all select * from other
        """
    df = client_bq.query(query).to_dataframe()
    return df

def generate_recipe(new_recipe_idea, recipe_examples):
    prompt = f'''Given this recipe idea: {new_recipe_idea}, please suggest a 
        1.recipeName 2.recipeTags 3.recipeIngredients 4.procedure 
        **important: include the quantity of the ingredients in grams or ml in the procedure**
        After that, please compute the following nutritional information per portion: 
        5.protein, 6.carbohydrate, 7.fat, 8.minimumCalories, 9.maximumCalories, 10.prepareTimeInMinutes, 11.cookingTimeInMinutes.
        Finally suggest the 12.flag of this recipe, choosing from the following options: porridge, shake, smoothie, other.

        In returning your answer please use the following table as reference:{recipe_examples.to_string()}

        ###OUTPUT - structure###
        please only return the table in json format with one row and the following columns: recipeName, recipeTags, recipeIngredients, procedure, protein, carbohydrate, fat, minimumCalories, maximumCalories, prepareTimeInMinutes, cookingTimeInMinutes, flag, reason.
        NOT INTRO NOT COMMENTS are needed.
        '''
    response = client_gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0)
    )
    #st.write(response.text)
    #response_nutritional_info = client_gemini.models.generate_content(
    #    model="gemini-2.5-flash",
    #    contents=f'''Given this recipe: {response.text}, evaluate the nutritional information per portion : protein, carbohydrate, fat, minimumCalories, maximumCalories, prepareTimeInMinutes, cookingTimeInMinutes. 
    #    for each nutritional information evaluation, please return the computation steps and the final value.
    #    Finally return the same table with the correct values for the nutritional information.
    #    ###OUTPUT - structure###
    #    please only return the table in json format with one row and the following columns: recipeName, recipeTags, recipeIngredients, procedure, protein, carbohydrate, fat, minimumCalories, maximumCalories, prepareTimeInMinutes, cookingTimeInMinutes, flag, reason, nutritional_info_evaluation.
    #    NOT INTRO NOT COMMENTS are needed.
    #    ''',
    #    config=types.GenerateContentConfig(temperature=0)
    #)
    #st.write(response_nutritional_info.text)
    # Clean and parse JSON response
    #response_text = response_nutritional_info.text.replace("```json", "").replace("```", "").strip()
    response_text = response.text.replace("```json", "").replace("```", "").strip()
    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON response: {e}\nResponse text: {response_text[:500]}")
        raise
    
    # Handle different JSON formats
    if isinstance(response_json, list):
        # If it's a list, take the first element
        if len(response_json) > 0:
            response_json = response_json[0]
        else:
            raise ValueError("Empty JSON response list")
    
    # Ensure response_json is a dict with one row
    if not isinstance(response_json, dict):
        raise ValueError(f"Unexpected JSON format: {type(response_json)}")
    
    # Normalize the dict: ensure all values are lists of length 1 for DataFrame creation
    normalized_dict = {}
    for key, value in response_json.items():
        if isinstance(value, list):
            # For list fields like recipeTags and recipeIngredients, keep as list but wrap in outer list
            normalized_dict[key] = [value]
        else:
            # For scalar values, wrap in a list
            normalized_dict[key] = [value]
    
    # Create DataFrame from normalized dict (one row)
    df_response = pd.DataFrame(normalized_dict)
    
    return df_response

def write_data(df, table_id, if_exists='append'):
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        if if_exists == 'replace' else bigquery.WriteDisposition.WRITE_APPEND
    )
    job = client_bq.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # Wait for the job to complete
    return f"Data successfully written to {table_id}"

def prepare_data(new_recipe_df, coach_slug, comment=None):
    new_recipe_df = new_recipe_df.copy()
    new_recipe_df['recipeId'] = str(uuid.uuid4())
    new_recipe_df['language'] = 'en-US'
    new_recipe_df['comment'] = comment if comment else None
    new_recipe_df['owner'] = coach_slug
    new_recipe_df['flag_proba'] = [[0.0, 0.0, 0.0, 0.0]]
    new_recipe_df['adminLink'] = None

    porridge = []
    shake = []
    smoothie = []

    f = new_recipe_df.flag.values[0] if 'flag' in new_recipe_df.columns else 'other'
    if f == 'porridge':
        porridge.append(1)
        shake.append(0)
        smoothie.append(0)
    elif f == 'shake':
        porridge.append(0)
        shake.append(1)
        smoothie.append(0)
    elif f == 'smoothie':
        porridge.append(0)
        shake.append(0)
        smoothie.append(1)
    else:
        porridge.append(0)
        shake.append(0)
        smoothie.append(0)

    new_recipe_df['porridge'] = porridge
    new_recipe_df['shake'] = shake
    new_recipe_df['smoothie'] = smoothie

    cols_order = ['recipeId', 'recipeName', 'language', 'recipeTags', 'recipeIngredients', 'protein', 'carbohydrate', 'fat', 'minimumCalories', 'maximumCalories', 'prepareTimeInMinutes', 'cookingTimeInMinutes', 'comment', 'procedure', 'owner', 'porridge', 'shake', 'smoothie', 'reason', 'adminLink', 'flag_proba']
    df_for_bq = new_recipe_df[cols_order]
    return df_for_bq

# Main UI
st.title("🍳 Recipe Creator")
st.markdown("Generate and customize recipes using AI")

# Step 1: Input section
st.header("Step 1: Recipe Input")
col1, col2 = st.columns(2)

with col1:
    coach_slug = st.text_input("Coach Slug", value="coach-nime", help="Enter your coach slug identifier")

with col2:
    new_recipe_idea = st.text_area("New Recipe Idea", placeholder='e.g., "Healthy Tiramisu": Overnight oats soaked in black espresso with light mascarpone and chocolate chips', height=100)

# Generate recipe button
if st.button("🚀 Generate Recipe", type="primary", width='stretch'):
    if not coach_slug:
        st.error("Please enter a coach slug")
    elif not new_recipe_idea:
        st.error("Please enter a recipe idea")
    else:
        with st.spinner("Generating recipe... This may take a moment."):
            try:
                # Load recipe examples (cached)
                if st.session_state.recipe_examples is None:
                    st.session_state.recipe_examples = import_recipe_examples(limit=20)
                
                # Generate recipe
                recipe_df = generate_recipe(new_recipe_idea, st.session_state.recipe_examples)
                st.session_state.recipe_df = recipe_df
                st.session_state.recipe_generated = True
                st.success("Recipe generated successfully! ✅")
            except Exception as e:
                st.error(f"Error generating recipe: {str(e)}")
                st.session_state.recipe_generated = False

# Step 2: Edit recipe section
if st.session_state.recipe_generated and st.session_state.recipe_df is not None:
    st.header("Step 2: Edit Recipe Details")
    
    recipe_df = st.session_state.recipe_df.copy()
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Text fields
        recipe_name = st.text_input("Recipe Name", value=str(recipe_df['recipeName'].iloc[0]) if 'recipeName' in recipe_df.columns else "")
        
        # Handle recipeTags (could be list or string, possibly nested)
        recipe_tags_raw = recipe_df['recipeTags'].iloc[0] if 'recipeTags' in recipe_df.columns else ""
        # Handle nested lists (if DataFrame stored it as [[list]])
        while isinstance(recipe_tags_raw, list) and len(recipe_tags_raw) == 1 and isinstance(recipe_tags_raw[0], list):
            recipe_tags_raw = recipe_tags_raw[0]
        if isinstance(recipe_tags_raw, (list, np.ndarray)):
            recipe_tags_str = ", ".join(str(tag) for tag in recipe_tags_raw if tag)
        else:
            recipe_tags_str = str(recipe_tags_raw) if pd.notna(recipe_tags_raw) else ""
        recipe_tags = st.text_input("Recipe Tags (comma-separated)", value=recipe_tags_str)
        
        # Handle recipeIngredients (could be list or string, possibly nested)
        recipe_ingredients_raw = recipe_df['recipeIngredients'].iloc[0] if 'recipeIngredients' in recipe_df.columns else ""
        # Handle nested lists (if DataFrame stored it as [[list]])
        while isinstance(recipe_ingredients_raw, list) and len(recipe_ingredients_raw) == 1 and isinstance(recipe_ingredients_raw[0], list):
            recipe_ingredients_raw = recipe_ingredients_raw[0]
        if isinstance(recipe_ingredients_raw, (list, np.ndarray)):
            recipe_ingredients_str = ", ".join(str(ing) for ing in recipe_ingredients_raw if ing)
        else:
            recipe_ingredients_str = str(recipe_ingredients_raw) if pd.notna(recipe_ingredients_raw) else ""
        recipe_ingredients = st.text_area("Recipe Ingredients (comma-separated)", value=recipe_ingredients_str, height=150)
        
        # Procedure
        procedure = st.text_area("Procedure", value=str(recipe_df['procedure'].iloc[0]) if 'procedure' in recipe_df.columns else "", height=200)
    
    with col2:
        # Nutritional information
        st.subheader("Nutritional Information")
        
        protein = st.number_input("Protein (g)", value=float(recipe_df['protein'].iloc[0]) if 'protein' in recipe_df.columns and pd.notna(recipe_df['protein'].iloc[0]) else 0.0, min_value=0.0, step=0.1)
        carbohydrate = st.number_input("Carbohydrate (g)", value=float(recipe_df['carbohydrate'].iloc[0]) if 'carbohydrate' in recipe_df.columns and pd.notna(recipe_df['carbohydrate'].iloc[0]) else 0.0, min_value=0.0, step=0.1)
        fat = st.number_input("Fat (g)", value=float(recipe_df['fat'].iloc[0]) if 'fat' in recipe_df.columns and pd.notna(recipe_df['fat'].iloc[0]) else 0.0, min_value=0.0, step=0.1)
        minimum_calories = st.number_input("Minimum Calories", value=int(recipe_df['minimumCalories'].iloc[0]) if 'minimumCalories' in recipe_df.columns and pd.notna(recipe_df['minimumCalories'].iloc[0]) else 0, min_value=0, step=1)
        maximum_calories = st.number_input("Maximum Calories", value=int(recipe_df['maximumCalories'].iloc[0]) if 'maximumCalories' in recipe_df.columns and pd.notna(recipe_df['maximumCalories'].iloc[0]) else 0, min_value=0, step=1)
        
        st.subheader("Time Information")
        prepare_time = st.number_input("Prepare Time (minutes)", value=int(recipe_df['prepareTimeInMinutes'].iloc[0]) if 'prepareTimeInMinutes' in recipe_df.columns and pd.notna(recipe_df['prepareTimeInMinutes'].iloc[0]) else 0, min_value=0, step=1)
        cooking_time = st.number_input("Cooking Time (minutes)", value=int(recipe_df['cookingTimeInMinutes'].iloc[0]) if 'cookingTimeInMinutes' in recipe_df.columns and pd.notna(recipe_df['cookingTimeInMinutes'].iloc[0]) else 0, min_value=0, step=1)
    
    # Step 3: Comment section
    st.header("Step 3: Add Comment (Optional)")
    comment = st.text_area("Comment", placeholder="Add any additional notes or comments about this recipe", height=100)
    
    # Step 4: Save to BigQuery
    st.header("Step 4: Save Recipe")
    save_to_bigquery = st.checkbox("Save to BigQuery", value=False)
    
    if save_to_bigquery:
        table_id = "bi-lenus-staging.dbt_nime.meal_recipes_flag_us_v3"
    
    # Update recipe dataframe with edited values
    if st.button("💾 Save Recipe", type="primary", width='stretch'):
        try:
            # Update the dataframe with edited values
            recipe_df['recipeName'] = recipe_name
            # Wrap lists in another list to match DataFrame row count (1 row)
            recipe_df['recipeTags'] = [[tag.strip() for tag in recipe_tags.split(",") if tag.strip()]] if recipe_tags else [[]]
            recipe_df['recipeIngredients'] = [[ing.strip() for ing in recipe_ingredients.split(",") if ing.strip()]] if recipe_ingredients else [[]]
            recipe_df['procedure'] = procedure
            recipe_df['protein'] = protein
            recipe_df['carbohydrate'] = carbohydrate
            recipe_df['fat'] = fat
            recipe_df['minimumCalories'] = minimum_calories
            recipe_df['maximumCalories'] = maximum_calories
            recipe_df['prepareTimeInMinutes'] = prepare_time
            recipe_df['cookingTimeInMinutes'] = cooking_time
            
            # Handle flag if it exists
            if 'flag' not in recipe_df.columns:
                recipe_df['flag'] = 'other'
            if 'reason' not in recipe_df.columns:
                recipe_df['reason'] = None
            
            # Prepare data for BigQuery
            df_for_bq = prepare_data(recipe_df, coach_slug, comment)
            
            if save_to_bigquery:
                if not table_id:
                    st.error("Please enter a BigQuery table ID")
                else:
                    with st.spinner("Saving to BigQuery..."):
                        result = write_data(df=df_for_bq, table_id=table_id, if_exists='append')
                        st.success(result)
            else:
                st.info("✅ Recipe prepared successfully! Check 'Save to BigQuery' above and click 'Save Recipe' again to save to BigQuery.")
            
            # Display the final recipe
            st.header("Final Recipe Preview")
            st.dataframe(df_for_bq, width='stretch')
            
        except Exception as e:
            st.error(f"Error saving recipe: {str(e)}")
            st.exception(e)


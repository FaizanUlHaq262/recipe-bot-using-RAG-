# Food Recipe Chatbot
#### (The streamlit version of the GUI will be uploaded soon...)
## Project Overview
This project involves developing a chatbot specialized in food recipes. It uses the Retrieval Augmented Generation (RAG) approach, combining LangChain, Chroma DB, and Ollama for training the model on specific input retrieval from the vector database. The chatbot is deployed locally using Flask.

## Domain
The chatbot focuses exclusively on food recipes, providing a specialized and expert system in this domain.

## Dataset
The dataset, recipeNLG, was sourced from Kaggle. It contains over a hundred thousand recipes but was narrowed down to a manageable subset of 3,000 for efficiency. The dataset features include title, ingredients, and directions, with links and sources removed for simplification.

## Cleaning Process
The data was cleaned to remove any unknown ASCII texts. ASCII for degree symbols was replaced with 'degrees', and 'c.' with 'cups'. The clean data was then concatenated and embedded into a vector database.

## Embedding Experiments
Three different embedding methods were tested:
1. Separate fields for title, ingredients, and directions.
2. Concatenated fields without new lines.
3. A single string of all data concatenated together.

The third method showed the best cosine similarity results and was chosen for the final model.

## How to Run the Project
Ensure you have Python and Flask installed, then run the Flask server script to start the application locally.

### Supported Commands
- `/recipe <food name>`: Retrieve a specific recipe.
- `/recommend <ingredient names>`: Get recipe recommendations based on available ingredients.
- `/how`: Inquire about the quantity needed for specific ingredients.

These inputs are the only allowed formats for interacting with the chatbot.

## Installation
```bash
git clone https://github.com/yourusername/food-recipe-chatbot.git
cd food-recipe-chatbot
pip install -r requirements.txt
python app.py

from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
import boto3
import os
import streamlit as st

os.environ["AWS_PROFILE"] = "trevor"

# Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

modelID = "anthropic.claude-v2"

llm = Bedrock(
    model_id=modelID,
    client=bedrock_client,
    model_kwargs={"max_tokens_to_sample": 2000, "temperature": 0.9}
)

def nutrition_chatbot(food_item):
    prompt = PromptTemplate(
        input_variables=["food_item"],
        template="You are a nutrition expert. Provide detailed nutritional information about the following food item: {food_item}."
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)

    response = bedrock_chain({'food_item': food_item})
    return response

# Streamlit application
st.title("Nutrition Information Chatbot")

food_item = st.sidebar.text_input(label="Enter a food item to get nutritional information:")

if food_item:
    response = nutrition_chatbot(food_item)
    st.write(response['text'])

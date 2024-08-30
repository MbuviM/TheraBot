# Install necessary libraries
import os
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage
from datasets import load_dataset
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions 
from chromadb.config import Settings
import uuid
import logging 

# Disable Azure SDK logging
logging.getLogger('azure').setLevel(logging.WARNING)

# Set up logging to help track execution
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


#Create load dotenv instance
load_dotenv()

# Create a client to consume the model
client = ChatCompletionsClient(
    endpoint=os.getenv("AZURE_INFERENCE_ENDPOINT"),
    credential=AzureKeyCredential(os.getenv("AZURE_INFERENCE_CREDENTIAL")),
)

# Load dataset
mental_health_data = load_dataset("Amod/mental_health_counseling_conversations")

# Create a Chroma client
chromaClient = chromadb.Client()

# Create chroma collection to store the embeddings
therabotCollection = chromaClient.create_collection(name="therabot")

# Setting up the embedding model
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Function to process and embed the dataset into ChromaDB
def embed_and_store_data():
    try:
        logger.info(f"Total examples: {len(mental_health_data['train'])}")
        for i, example in enumerate(mental_health_data['train']):
            logger.info(f"Processing example {i}")
            
            context = example['Context']
            response = example['Response']
            
            logger.info(f"Context: {context[:50]}...")
            logger.info(f"Response: {response[:50]}...")
            
            context_embedding = default_ef([context])
            response_embedding = default_ef([response])
            
            logger.info(f"Embeddings created. Shapes: {len(context_embedding)}, {len(response_embedding)}")
            
            try:
                therabotCollection.add(
                    documents=[context],
                    embeddings=context_embedding,
                    ids=[str(uuid.uuid4())]
                )
                therabotCollection.add(
                    documents=[response],
                    embeddings=response_embedding,
                    ids=[str(uuid.uuid4())]
                )
            except Exception as e:
                logger.error(f"Error adding to collection: {e}")
                
    except Exception as e:
        logger.error(f"An error occurred: {e}")

# Define predefined scripts for specific scenarios
scripts = {
    "anxiety": [
        "It seems like you're feeling anxious. Let's take a deep breath together.",
        "Focus on your breathing: Inhale deeply through your nose, hold for a few seconds, and exhale slowly through your mouth.",
        "Try to ground yourself by naming five things you can see around you right now."
    ],
    "panic attack": [
        "It sounds like you're experiencing a panic attack. Remember, it will pass. Try to focus on slowing down your breath.",
        "Place your hand on your stomach and breathe deeply, feeling your hand rise and fall with each breath.",
        "Can you identify where you feel the panic in your body? This can help you reconnect with the present moment."
    ],

}

# Function to detect if a script should be used based on a user's response
def use_script(user_input):
    # Check for trigger words
    for trigger, script_lines in scripts.items():
        if trigger in user_input.lower():
            return script_lines
    return None

# Function to generate response based on scripts, chroma retrieval and user input
def generate_response(user_input):
    # Check if a script is needed
    script_response = use_script(user_input)

    if script_response is not None:
        # If a script is found, return the script as the response
        return "\n".join(script_response)
    
    else:
        question_embedding = default_ef([user_input])

     # Query the ChromaDB for the most relevant response
    results = therabotCollection.query(
        query_embeddings=question_embedding,
        n_results=1
    )
    
    # Get the response linked to the most similar context
    retrieved_response = results['documents'][0]
    
    # Generate a chatbot response using the retrieved context
    system_message = SystemMessage(content=f"Contextual Response: {retrieved_response}")
    response = client.complete(
        messages=[
            system_message,
            UserMessage(content=user_input),
        ],
        temperature=0.7,
        max_tokens=150
    )

    return response.choices[0].message.content

# Example usage
user_input = "I'm feeling anxious right now."
response = generate_response(user_input)
print(response)

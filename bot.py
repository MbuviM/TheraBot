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

# Define the datasets to be used
datasets = [
    {"name": "mental_health_counseling", "path": "Amod/mental_health_counseling_conversations"},
    {"name": "mental_health_chat", "path": "mpingale/mental-health-chat-dataset"},
]

# Function to load multiple datasets
def load_multiple_datasets():
    loaded_datasets = {}
    for dataset_info in datasets:
        try:
            loaded_datasets[dataset_info['name']] = load_dataset(dataset_info['path'])
            print(f"Successfully loaded dataset: {dataset_info['name']}")
        except Exception as e:
            print(f"Error loading dataset {dataset_info['name']}: {e}")
    return loaded_datasets

# Load the datasets
mental_health_data = load_multiple_datasets()

# Create a Chroma client
chromaClient = chromadb.Client()

# Create chroma collection to store the embeddings
therabotCollection = chromaClient.create_collection(name="therabot")

# Setting up the embedding model
default_ef = embedding_functions.DefaultEmbeddingFunction()

def embed_and_store_data(dataset_name):
    try:
        if dataset_name == 'mental_health_counseling':
            column_context = 'Context'
            column_response = 'Response'
        elif dataset_name == 'mental_health_chat':
            column_context = 'questionText'
            column_response = 'answerText'
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        logger.info(f"Total examples: {len(mental_health_data['train'])}")
        for i, example in enumerate(mental_health_data['train']):
            logger.info(f"Processing example {i}")

            context = example[column_context]
            response = example[column_response]

            logger.info(f"Context: {context[:50]}...")
            logger.info(f"Response: {response[:50]}...")

            context_embedding = default_ef.embed([context])
            response_embedding = default_ef.embed([response])

            logger.info(f"Embeddings created. Shapes: {len(context_embedding[0])}, {len(response_embedding[0])}")

            try:
                therabotCollection.add(
                    documents=[context],
                    embeddings=[context_embedding[0]],
                    ids=[str(uuid.uuid4())]
                )
                therabotCollection.add(
                    documents=[response],
                    embeddings=[response_embedding[0]],
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
    system_message = SystemMessage(content=f"You are a helpful therapy bot called TheraBot in Kenya. Provide supportive and empathetic responses. Contextual Response: {retrieved_response}")
    response = client.complete(
        messages=[
            system_message,
            UserMessage(content=user_input),
        ],
        temperature=1,   #  Controls the randomness and creativity of the output
        top_p=1,      # Controls the diversity
        max_tokens=2048     # Controls the content amount in the responses
    )

    return response.choices[0].message.content

# Example usage
user_input = "I experienced GBV."
response = generate_response(user_input)
print(response)

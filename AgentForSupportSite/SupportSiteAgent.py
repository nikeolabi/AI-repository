import os
import logging
import getpass
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langgraph.checkpoint.memory import MemorySaver
import requests
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import warnings
from urllib.parse import urljoin, urlparse
from unstructured.partition.auto import partition
from unstructured.partition.image import partition_image
from unstructured.partition.pdf import partition_pdf
import pytesseract
import magic


# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# API Keys
"""
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
"""

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

#print(f"DEBUG: GROQ_API_KEY Retrieved: {'SET' if GROQ_API_KEY else 'NOT SET'}")
#print(f"DEBUG: ANTHROPIC_API_KEY Retrieved: {'SET' if ANTHROPIC_API_KEY else 'NOT SET'}")

##########################################################
# Initialize LLMs
"""
retrieval_llm = ChatGroq(
    model_name="llama-3.3-70b-versatile", 
    temperature=0.6, 
    api_key=GROQ_API_KEY)
"""
structured_llm = ChatAnthropic(
    model_name="claude-3-5-sonnet-20241022",
    temperature=0.5, 
    api_key=ANTHROPIC_API_KEY)

#########################################################
# Define Support Documentation/Linksto retrieve
start_urls = [
    "https://support.neonode.com/docs/display/AIRTSUsersGuide/Introduction"
    #"https://support.neonode.com/docs/display/AIRTSUsersGuide",
    #"https://support.neonode.com/docs/display/NIPB",
    #"https://support.neonode.com/docs/display/ZFPUG",
    #"https://support.neonode.com/docs/display/workbench",
    #"https://support.neonode.com/docs/display/QA/Neonode+Help+Center"
]


def get_all_subpages(start_urls, max_depth):
    visited = set()
    urls_to_visit = [(url, 0) for url in start_urls]
    while urls_to_visit:
        url, depth = urls_to_visit.pop(0)
        if url in visited or depth > max_depth:
            continue
        visited.add(url)
        try:
            response = requests.get(url, timeout=2)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
        except requests.RequestException:
            continue
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        for link in soup.find_all("a", href=True):
            href = link["href"].strip()
            full_url = urljoin(domain, href)
            
             # Skip unsupported file types
            # Convert to lowercase and check if the URL contains an unsupported file type
            unsupported_types = ['.svg', '.mp4', '.zip', '.exe', '.dmg']
            if any(ext in full_url.lower() for ext in unsupported_types):
                logging.warning(f"Skipping unsupported file: {full_url}")
                continue
            
            if full_url.startswith(domain) and full_url not in visited:
                urls_to_visit.append((full_url, depth + 1))
    return list(visited)

############################################################
# Support Documentation Loading and Retrieval
urls = get_all_subpages(start_urls, max_depth=1)
#loader = UnstructuredURLLoader(urls, mode="text") # Load text content from URLs
loader = UnstructuredURLLoader(urls)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
vector_store = InMemoryVectorStore.from_documents(chunks, embeddings_model)
retriever = vector_store.as_retriever()


# Predefined FAQ Data
FAQ_DATA = {
    "password_reset": "To reset your password, go to the login page and click 'Forgot Password'. Follow the instructions sent to your email.",
    "account_locked": "If your account is locked due to multiple failed login attempts, please wait 30 minutes or contact support.",
    "subscription_cancel": "To cancel your subscription, navigate to your account settings and select 'Manage Subscription'."
}

########################################################
# Define Support Tools
def fetch_documentation(query: str) -> str:
    """Retrieve documentation-related answers."""
    results = retriever.invoke(query)
    if results:
        return results[0].page_content
    return "No relevant documentation found."

########################################################
# Simulated support ticket database
ticket_database = {}

def fetch_faq(topic: str) -> str:
    """Retrieve an FAQ response based on user query."""
    return FAQ_DATA.get(topic.lower(), "I'm sorry, I don't have information on that topic. Please contact support.")


def create_support_ticket(company: str, contact_name: str, email: str, product: str, issue: str, description: str) -> str:
    """Simulate support ticket creation with detailed information."""
    ticket_id = len(ticket_database) + 1
    ticket_database[ticket_id] = {
        "company": company,
        "contact_name": contact_name,
        "email": email,
        "product": product,
        "issue": issue,
        "description": description,
        "status": "Open"
    }
    logging.info(f"Support ticket #{ticket_id} created for {contact_name} at {company}: {issue}")
    return f"Support ticket #{ticket_id} has been created for {contact_name} at {company}. Our team will contact you at {email}."


def close_support_ticket(ticket_id: int) -> str:
    """Simulate closing a support ticket."""
    if ticket_id in ticket_database:
        ticket_database[ticket_id]["status"] = "Closed"
        logging.info(f"Support ticket #{ticket_id} closed.")
        return f"Support ticket #{ticket_id} has been closed. Thank you for reaching out."
    return "Invalid ticket ID. Please check and try again."


def check_ticket_status(ticket_id: int) -> str:
    """Check the status of a support ticket."""
    if ticket_id in ticket_database:
        status = ticket_database[ticket_id]["status"]
        return f"Support ticket #{ticket_id} is currently {status}."
    return "Invalid ticket ID. Please check and try again."



########################################################
# Create Unified Agent Prompt
prompt = """
You are a Support Site Agent for company called Neonode, designed to assist users with troubleshooting, FAQs, documentation, and support inquiries.
The product you support is a so called Touch Senor Module (TSM) that enables touch functionality on various screens. The company has also another vertical for "in cabim monitoring" (ICM) that is used in vehicles.
You do not have information about the ICM product, but you can assist with general support inquiries.
You have access to a knowledge base of support documentation and FAQs related to the TSM product.
Your responses should be professional, concise, and helpful.

Response Guidelines:

1. For greetings (hi, hello, etc.):
   - Respond with a polite and professional greeting.
   - Example: "Hello! How can I assist you today?"

2. When to use tools:
   - fetch_faq: ONLY when the user asks about common support topics.
   - fetch_documentation: ONLY when the user asks a technical question.
   - create_support_ticket: ONLY when the user explicitly requests to open a support ticket.

3. Tool response format:
   - Use the tool
   - Provide a brief and clear response based on the tool's output
   - Offer additional assistance if needed

4. Always define acronyms when they're first mentioned

5. Provide clear, direct answers without requiring multiple follow-up questions

6. Anticipate that users might not be familiar with our technical terminology

Remember:
- Keep responses concise and solution-focused.
- Avoid unnecessary information or overly technical language.
- When unsure, suggest contacting support for further assistance.
"""

########################################################
# Create Unified Agent
try:
    print("DEBUG: Creating agent...")
    agent = create_react_agent(
        model=structured_llm,
        tools=[fetch_faq, create_support_ticket, check_ticket_status, close_support_ticket, fetch_documentation],
        prompt=prompt,
        checkpointer=MemorySaver()
    )
    print("DEBUG: Agent created successfully!")
except Exception as e:
    print(f"ERROR: Failed to create agent - {e}")


def chat():
    print("\nWelcome to the Support Site Assistant! How can I assist you today? Type 'quit' to exit.\n")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nThank you for reaching out. Have a great day!")
            break
        
        #print(f"DEBUG: User input received: {user_input}")

        try:
            #print("DEBUG: Invoking agent...")
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": "thread1"}}
            )
            #print("DEBUG: Agent invoked successfully!")
            #print(f"DEBUG: Raw result: {result}")
            
            if result.get("messages"):
                last_message = result["messages"][-1]
                if hasattr(last_message, "content"):
                    print("\nSupport Agent:", last_message.content)
                else:
                    print("DEBUG: Last message has no content attribute.")
            else:
                print("DEBUG: No messages in the result.")
                
        except Exception as e:
            print(f"ERROR: Agent invocation failed - {e}")

  
""" 
agent = create_react_agent(
    model=structured_llm,
    tools=[fetch_faq, create_support_ticket, check_ticket_status, close_support_ticket, fetch_documentation],
    prompt=prompt,
    checkpointer=MemorySaver()
)

def chat():
    print("\nWelcome to the Support Site Assistant! How can I assist you today? Type 'quit' to exit.\n")
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nThank you for reaching out. Have a great day!")
            break
        result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config={"configurable": {"thread_id": "thread1"}})
        if result.get("messages"):
            last_message = result["messages"][-1]
            if hasattr(last_message, "content"):
                print("\nSupport Agent:", last_message.content)
"""

if __name__ == "__main__":
    chat()
    
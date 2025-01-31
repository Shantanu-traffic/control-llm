from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from typing import Optional, List, Dict
import g4f
import requests
import spacy
import json
import re
from typing import Set
import asyncio
import httpx




# --- FastAPI Application ---
app = FastAPI()

# Load spaCy model for entity extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("SpaCy model 'en_core_web_sm' is not installed.")

# WebSocket connections set
connections = set()

# Base API URL
BASE_URL = "https://v2.refresh.controlfms.com/"

# Global header for requests
HEADERS = {
    "Cookie": ""
}

# Global stage mapping
stage_mapping = {
    "Build Stage": "74f83310-878a-49f1-92c1-f3a759677080",
    "Working Drawings & Costing": "490436fb-5a1a-4357-bc9d-3693b8a89c24",
    "Project Closure": "d4ad2e6f-97f0-4aff-b3a3-69ff786302bb",
    "Lead In": "9db081ef-6281-489f-945e-b7d8ef487e4d",
    "Concept & Feasibility": "f2ea2be6-50ee-40f9-a316-9b5560fca7fb",
    "Initial Consultation": "d51ecf97-d6db-42c4-8cb0-bc055bb5f129"
}

# Utility for API tools
class CustomTool:
    def __init__(self, name: str, api_endpoint: str, description: str):
        self.name = name
        self.api_endpoint = api_endpoint
        self.description = description

    async def execute_get(self, context: Dict[str, str]) -> str:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self.api_endpoint, headers=HEADERS, params=context)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                return {"error": f"Unable to fetch data from API. Details: {e}"}
            except ValueError as ve:
                return {"error": "Response could not be parsed as JSON. Details: {ve}"}

    async def execute_post(self, data: Dict[str, str]) -> str:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.api_endpoint, headers=HEADERS, json=data)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                return {"error": f"Unable to fetch data from API. Details: {e}"}
            except ValueError as ve:
                return {"error": "Response could not be parsed as JSON. Details: {ve}"}

    async def execute_put(self, data: Dict[str, str]) -> str:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(self.api_endpoint, headers=HEADERS, json=data)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                return {"error": f"Unable to fetch data from API. Details: {e}"}
            except ValueError as ve:
                return {"error": "Response could not be parsed as JSON. Details: {ve}"}

# API-related functions
async def get_pipeline_projects(context: Dict[str, str]) -> str:
    url = f"{BASE_URL}api/v13/deal/pipeline/initial?limit=10&meta=true"
    tool = CustomTool(
        name="pipeline_projects",
        api_endpoint=url,
        description="Get all projects or deals or leads or jobs available in pipeline.",
    )
    return await tool.execute_get(context)

async def get_completed_activities(context: Dict[str, str]) -> str:
    url = f"{BASE_URL}api/v13/deal/manage-activities/master?date_completed=true&order_direction=desc"
    tool = CustomTool(
        name="completed_activities",
        api_endpoint=url,
        description="Get all completed activity list.",
    )
    return await tool.execute_get(context)

async def get_non_completed_activities(context: Dict[str, str]) -> str:
    url = f"{BASE_URL}api/v13/deal/manage-activities/master?order_direction=desc"
    tool = CustomTool(
        name="non_completed_activities",
        api_endpoint=url,
        description="Get all non-completed activity list.",
    )
    return await tool.execute_get(context)

async def get_completed_projects(context: Dict[str, str]) -> str:
    url = f"{BASE_URL}api/v13/deal/lists/completed/master?page_size=25&order_direction=desc&order_by=date_completed"
    tool = CustomTool(
        name="completed_projects",
        api_endpoint=url,
        description="Get all completed projects or deals or leads or jobs.",
    )
    return await tool.execute_get(context)

async def process_stage_wise_counts(meta_data: List[Dict[str, str]]) -> str:
    """
    Process stage-wise project counts from the meta_data.

    Args:
        meta_data (List[Dict[str, str]]): The meta data list from the API response.

    Returns:
        str: Formatted stage-wise project counts.
    """
    stage_counts = []
    for stage_name, stage_id in stage_mapping.items():
        count = next((item["count"] for item in meta_data if item["stage_id"] == stage_id), "0")
        stage_counts.append(f"{stage_name}: {count}")
    return "\n".join(stage_counts)

async def move_to_initial_stage(data: Dict[str, str]) -> str:
    url = f"{BASE_URL}api/v13/pricing/move-stage-initial"
    tool = CustomTool(
        name="move_stage",
        api_endpoint=url,
        description="Move a project or deal or lead or job to the initial consultation stage.",
    )
    return await tool.execute_post(data)
async def put_project_on_hold(data: Dict[str, str]) -> str:
    url = f"{BASE_URL}api/v13/deal/change-deal-status/onhold"
    tool = CustomTool(
        name="put_on_hold",
        api_endpoint=url,
        description="Put a project or deal or lead or job on hold.",
    )
    return await tool.execute_put(data)

# Entity extraction
async def extract_entities(user_input: str) -> List[tuple]:
    doc = await nlp(user_input)
    return [(ent.text, ent.label_) for ent in doc.ents]


# Custom LLM with integrated logic
class EducationalLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    async def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        match = re.search(r"Conversation History:(.*?)User Query:(.+)", prompt, re.S)
        conversation_history = match.group(1).strip() if match else ""
        user_query = match.group(2).strip() if match else ""
        
         # You can check the kwargs to see what additional context might be provided
        print(f"Additional arguments received: {kwargs}")

        # Intent detection
        intent = await detect_intent(user_query)
        print(f"Detected Intent: {intent}")

        # Handle intents
        if intent == "LIST_PROJECTS_FROM_PIPELINE" or intent == "GET_PROJECTS_FROM_PIPELINE":
            response = await get_pipeline_projects({})
            return json.dumps(response, indent=2) if isinstance(response, dict) else response
        elif intent == "STAGE_WISE_PROJECT_COUNT":
          # Fetch pipeline projects
          response = await get_pipeline_projects({})
          if isinstance(response, dict) and response.get("status"):
              meta_data = response["data"]["meta",[]]
              # Call the processing function
              final_response = process_stage_wise_counts(meta_data)
              print(type(final_response))
          return json.dumps(final_response, indent=2) if isinstance(final_response, dict) else final_response

        elif intent == "GET_COMPLETED_ACTIVITIES" or intent == "LIST_COMPLETED_ACTIVITIES":
            response = await get_completed_activities({})
            return json.dumps(response, indent=2) if isinstance(response, dict) else response
        elif intent == "GET_NON_COMPLETED_ACTIVITIES" or intent == "LIST_NON_COMPLETED_ACTIVITIES" or intent == "LIST_UNCOMPLETED_ACTIVITIES":
            response = await get_non_completed_activities({})
            return json.dumps(response, indent=2) if isinstance(response, dict) else response
        elif intent == "GET_COMPLETED_PROJECTS" or intent == "LIST_COMPLETED_PROJECTS":
            response = await get_completed_projects({})
            return json.dumps(response, indent=2) if isinstance(response, dict) else response
        elif intent == "MOVE_TO_INITIAL_STAGE" or intent == "CHANGE_STAGE_OF_PROJECT":
            # Prompt user for required data
            alternative_id = input("Enter Project ID: ").strip()
            stage_id = "9db081ef-6281-489f-945e-b7d8ef487e4d"  # Fixed value #
            data = {"operation": {"ubiquity": False}, "alternative_id": alternative_id, "stage_id": stage_id}
            response = await move_to_initial_stage(data)
            return json.dumps(response, indent=2) if isinstance(response, dict) else response
        elif intent == "PUT_PROJECT_ON_HOLD" or intent == "PROJECT_ON_HOLD_LIST":
            # Prompt user for required data
            alternative_id = input("Enter Deal ID: ").strip()
            review_date = input("Enter On-Hold Review Date (YYYY-MM-DD): ").strip() + "T18:30:00.000Z"
            data = {"alternative_id": alternative_id, "onhold_review_date": review_date}
            response =  await put_project_on_hold(data)
            return json.dumps(response, indent=2) if isinstance(response, dict) else response
        elif intent == "UNKNOWN_INTENT":
            return "I'm sorry, I couldn't understand your query. Could you please rephrase it?"
        else:
            return "An unexpected error occurred while processing your query."

# --- Intent Detection ---
# Intent detection (simplified for this example)
async def detect_intent(user_query: str) -> str:
  intent_list = [
        "LIST_PROJECTS_FROM_PIPELINE",
        "GET_PROJECTS_FROM_PIPELINE",
        "STAGE_WISE_PROJECT_COUNT",
        "LIST_COMPLETED_ACTIVITIES",
        "GET_COMPLETED_ACTIVITIES",
        "LIST_NON_COMPLETED_ACTIVITIES",
        "GET_NON_COMPLETED_ACTIVITIES",
        "LIST_UNCOMPLETED_ACTIVITIES",
        "GET_COMPLETED_PROJECTS",
        "LIST_COMPLETED_PROJECTS",
        "MOVE_TO_INITIAL_STAGE",
        "CHANGE_STAGE_OF_PROJECT",
        "PUT_PROJECT_ON_HOLD",
        "PROJECT_ON_HOLD_LIST",
        "GENERAL_KNOWLEDGE"]


  try:
      prompt = f"""
      You are a system designed to classify user intents. Analyze the following query and provide the intent as one of the following:
      - LIST_PROJECTS_FROM_PIPELINE
      - GET_PROJECTS_FROM_PIPELINE
      - STAGE_WISE_PROJECT_COUNT
      - LIST_COMPLETED_ACTIVITIES
      - GET_COMPLETED_ACTIVITIES
      - LIST_NON_COMPLETED_ACTIVITIES
      - GET_NON_COMPLETED_ACTIVITIES
      - LIST_UNCOMPLETED_ACTIVITIES
      - GET_COMPLETED_PROJECTS
      - LIST_COMPLETED_PROJECTS
      - MOVE_TO_INITIAL_STAGE
      - CHANGE_STAGE_OF_PROJECT
      - PUT_PROJECT_ON_HOLD
      - PROJECT_ON_HOLD_LIST
      - GENERAL_KNOWLEDGE
      If the intent doesn't match these, return UNKNOWN_INTENT.

      User Query: "{user_query}"

      Detected Intent:
      """
      response = await g4f.ChatCompletion.create(
          model=g4f.models.gpt_4,  # Use a lighter model like gpt_3.5 for efficiency
          messages=[{"role": "user", "content": prompt}],
          verbose=False,
      )
      print(f"Detected Intent from promt: {response}")  # Debugging

      matching_intents = [intent for intent in intent_list if intent in response]
      if matching_intents:
          print(matching_intents[0])
          return matching_intents[0]
      else:
          return "UNKNOWN_INTENT"
  except Exception as e:
      return f"Error: {e}"



# --- LangChain Chain Setup ---
memory = ConversationBufferMemory()
prompt = PromptTemplate(
    input_variables=["history", "query"],
    template="""
        You are an assistant integrated with APIs. Respond using API data or generate a general answer.

        Conversation History:
        {history}

        User Query: {query}

        Response:
        """
)
chain = LLMChain(llm=EducationalLLM(), prompt=prompt, memory=memory)

async def process_user_query(user_input: str) -> str:
    try:
        response = await chain.ainvoke({"query": user_input, "history": memory.buffer})
        return response["text"]
    except Exception as e:
        return f"An error occurred: {e}"

# WebSocket Connection Manager
class WebSocketConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = WebSocketConnectionManager()


# --- FastAPI WebSocket Events ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    connections.add(websocket)

    try:
        # Step 1: Send a welcome message
        await websocket.send_text("I'm your Control Assistant, please tell me your query?")

        # Step 2: Ask for the Cookie value
        await websocket.send_text("Please provide the value for the Cookie in JSON format: {'message': '<cookie_value>'}")

        # Step 3: Receive the Cookie value from the client
        cookie_data = await websocket.receive_text()
        if cookie_data.strip().lower() == "exit":
            await websocket.send_text("Goodbye! Disconnecting from the server.")
            connections.remove(websocket)
            await websocket.close()
            return

        # Step 4: Parse the JSON and extract the `message` value
        try:
            cookie_dict = json.loads(cookie_data)
            cookie_value = cookie_dict.get("message")
            if not cookie_value:
                raise ValueError("Cookie value is missing in the provided JSON.")

            # Update the global HEADERS
            global HEADERS
            HEADERS["Cookie"] = cookie_value

            # Step 5: Show the updated headers back to the client
            await websocket.send_text(f"Cookie value set successfully. Updated headers: {HEADERS}")
        except json.JSONDecodeError:
            await websocket.send_text("Invalid JSON format. Please send the Cookie in the format: {'message': '<cookie_value>'}")
            return
        except ValueError as e:
            await websocket.send_text(f"Error: {e}")
            return

        # Step 6: Handle user queries in a loop
        while True:
            data = await websocket.receive_text()

            # Check if the client wants to exit
            if data.strip().lower() == "exit":
                await websocket.send_text("Goodbye! Disconnecting from the server.")
                connections.remove(websocket)
                await websocket.close()
                break

            # Process the user query
            response = await process_user_query(data)
            await websocket.send_text(response)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await websocket.close()

# --- Run the Application ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

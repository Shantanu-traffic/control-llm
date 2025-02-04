import asyncio
import json
import re
from typing import Optional, List, Dict

import httpx
import spacy
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

# --- Configuration and Setup ---

# Create FastAPI app
app = FastAPI()

# Load spaCy model for entity extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("SpaCy model 'en_core_web_sm' is not installed.")

# Global WebSocket connection set
connections = set()

# For UAT – you can change BASE_URL as needed
BASE_URL = "https://v2.refresh.controlfms.com/"
# BASE_URL = "http://localhost:3200/"

# Global headers; the Cookie header will be updated from the client
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


# --- HTTP API Tool Wrapper using async httpx ---

class CustomTool:
    def __init__(self, name: str, api_endpoint: str, description: str):
        self.name = name
        self.api_endpoint = api_endpoint
        self.description = description

    async def execute_get(self, context: Dict[str, str]) -> dict:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self.api_endpoint, headers=HEADERS, params=context)
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                return {"error": f"Unable to fetch data from API. Details: {e}"}
            except ValueError as ve:
                return {"error": f"Response could not be parsed as JSON. Details: {ve}"}

    async def execute_post(self, data: Dict[str, str]) -> dict:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.api_endpoint, headers=HEADERS, json=data)
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                return {"error": f"Unable to fetch data from API. Details: {e}"}
            except ValueError as ve:
                return {"error": f"Response could not be parsed as JSON. Details: {ve}"}

    async def execute_put(self, data: Dict[str, str]) -> dict:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.put(self.api_endpoint, headers=HEADERS, json=data)
                response.raise_for_status()
                return response.json()
            except httpx.RequestError as e:
                return {"error": f"Unable to fetch data from API. Details: {e}"}
            except ValueError as ve:
                return {"error": f"Response could not be parsed as JSON. Details: {ve}"}

# --- API Functions (Tools) ---

async def get_pipeline_projects(context: Dict[str, str]) -> dict:
    url = f"{BASE_URL}api/v13/deal/pipeline/initial?limit=10&meta=true"
    tool = CustomTool(
        name="get_pipeline_projects",
        api_endpoint=url,
        description="Get all projects (or deals, leads, jobs) available in pipeline.",
    )
    return await tool.execute_get(context)

async def get_completed_activities(context: Dict[str, str]) -> dict:
    url = f"{BASE_URL}api/v13/deal/manage-activities/master"
    tool = CustomTool(
        name="get_completed_activities",
        api_endpoint=url,
        description="Get all completed activity list.",
    )
    return await tool.execute_get(context)

async def get_non_completed_activities(context: Dict[str, str]) -> dict:
    url = f"{BASE_URL}api/v13/deal/manage-activities/master"
    tool = CustomTool(
        name="get_non_completed_activities",
        api_endpoint=url,
        description="Get all non-completed activity list.",
    )
    return await tool.execute_get(context)

async def get_completed_projects(context: Dict[str, str]) -> dict:
    url = f"{BASE_URL}api/v13/deal/lists/completed/master"
    tool = CustomTool(
        name="get_completed_projects",
        api_endpoint=url,
        description="Get all completed projects (or deals, leads, jobs).",
    )
    return await tool.execute_get(context)

async def move_to_initial_stage(data: Dict[str, str]) -> dict:
    url = f"{BASE_URL}api/v13/pricing/move-stage-initial"
    tool = CustomTool(
        name="move_to_initial_stage",
        api_endpoint=url,
        description="Move a project (or deal/lead/job) to the initial consultation stage.",
    )
    return await tool.execute_post(data)

async def put_project_on_hold(data: Dict[str, str]) -> dict:
    url = f"{BASE_URL}api/v13/deal/change-deal-status/onhold"
    tool = CustomTool(
        name="put_project_on_hold",
        api_endpoint=url,
        description="Put a project (or deal/lead/job) on hold.",
    )
    return await tool.execute_put(data)

def process_stage_wise_counts(meta_data: List[Dict[str, str]]) -> str:
    """
    Process stage-wise project counts from meta_data.
    """
    stage_counts = []
    for stage_name, stage_id in stage_mapping.items():
        count = next((item["count"] for item in meta_data if item["stage_id"] == stage_id), "0")
        stage_counts.append(f"{stage_name}: {count}")
    return "\n".join(stage_counts)

def get_scope_by_aoh(area_of_house: str) -> dict:
    with open('data.json', 'r') as file:
        data = json.load(file)
    
    return data[area_of_house]
    

def get_bathroom_scopes(stage: str) -> dict:
    data = get_scope_by_aoh('bathroom')
    return data['stages'][stage]

def get_bedroom_scopes(stage: str) -> dict:
    data = get_scope_by_aoh('bedroom')
    return data['stages'][stage]

def get_basement_scopes(stage: str) -> dict:
    data = get_scope_by_aoh('basement')
    return data['stages'][stage]

    

# --- Entity Extraction Utility ---
def extract_entities(user_input: str) -> List[tuple]:
    doc = nlp(user_input)
    return [(ent.text, ent.label_) for ent in doc.ents]

# --- Custom LLM with Function Calling Capability ---
#
# This custom LLM (EducationalLLM) analyzes the query to detect an intent.
# For demonstration, if the detected intent requires external API data,
# the _call method returns a special marker "CALL_FUNCTION:" followed by JSON 
# describing which function to call and with what arguments.
#
class EducationalLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Here we simulate extraction of conversation history and user query.
        # In a production system, you could use a real model to determine
        # whether a function call is needed.
        match = re.search(r"Conversation History:(.*?)User Query:(.+)", prompt, re.S)
        conversation_history = match.group(1).strip() if match else ""
        user_query = match.group(2).strip() if match else ""

        # Simple (simulated) intent detection based on keywords:
        if "pipeline" in user_query.lower():
            # Instead of immediately calling the API, return a function call instruction
            # (the agent indicates that it wants to call get_pipeline_projects)
            return (
                "CALL_FUNCTION: " +
                json.dumps({
                    "name": "get_pipeline_projects",
                    "arguments": {}  # if additional parameters are needed, include them here
                })
            )
        elif "completed activities" in user_query.lower():
            return (
                "CALL_FUNCTION: " +
                json.dumps({
                    "name": "get_completed_activities",
                    "arguments": {'date_completed':True, 'order_direction':'desc'}
                })
            )
        elif "on hold" in user_query.lower():
            # Here we assume the user did not provide required information.
            # The agent asks for missing information.
            # return "It seems you want to put a project on hold. Please provide the Deal ID and On-Hold Review Date (YYYY-MM-DD)."
            return (
                "CALL_FUNCTION: " +
                json.dumps({
                    "name": "put_project_on_hold",
                    "arguments": {}
                })
            )
        elif "to initial consultation" in user_query.lower():
            # Here we assume the user did not provide required information.
            # The agent asks for missing information.
            return (
                "CALL_FUNCTION: " +
                json.dumps({
                    "name": "move_to_initial_stage",
                    "arguments": {}
                })
            )
        elif "stage" in user_query.lower():
            return (
                "CALL_FUNCTION: " +
                json.dumps({
                    "name": "get_pipeline_projects",
                    "arguments": {'meta':True}
                })
            )
        elif "completed projects" in user_query.lower():
            return (
                "CALL_FUNCTION: " +
                json.dumps({
                    "name": "get_completed_projects",
                    "arguments": {'page_size':25, 'order_direction':'desc', 'order_by':'date_completed'}
                })
            )
        elif "non-completed activities" in user_query.lower():
            return (
                "CALL_FUNCTION: " +
                json.dumps({
                    "name": "get_non_completed_activities",
                    "arguments": {'order_direction':'desc'}
                })
            )
        elif "bathroom" in user_query.lower():
            return (
                "CALL_FUNCTION: " +
                json.dumps({
                    "name": "get_bathroom_scopes",
                    "arguments": {}
                })
            )
        elif "bedroom" in user_query.lower():
            return (
                "CALL_FUNCTION: " +
                json.dumps({
                    "name": "get_bedroom_scopes",
                    "arguments": {}
                })
            )
        elif "basement" in user_query.lower():
            return (
                "CALL_FUNCTION: " +
                json.dumps({
                    "name": "get_basement_scopes",
                    "arguments": {}
                })
            )
        # Fallback: general answer (could use general knowledge here)
        return "I'm sorry, I couldn't understand your query. Could you please rephrase it?"

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Wrap synchronous _call in an async function
        return await asyncio.to_thread(self._call, prompt, stop)

# --- LangChain Chain Setup with Conversation Memory ---
memory = ConversationBufferMemory()

prompt = PromptTemplate(
    input_variables=["history", "query"],
    template="""
You are an assistant integrated with external APIs. Use the conversation context provided to decide whether to call a function (tool) or provide a general answer.

Conversation History:
{history}

User Query: {query}

Response:
"""
)

# Create a chain that uses our custom LLM, the prompt, and conversation memory.
chain = LLMChain(llm=EducationalLLM(), prompt=prompt, memory=memory)

# --- Streaming Utility ---
async def stream_response(websocket: WebSocket, text: str):
    """
    Simulate streaming of a text response by sending one line at a time.
    """
    for line in text.splitlines():
        await websocket.send_text(line)
        await asyncio.sleep(0.1)  # simulate a brief delay between chunks

# --- Function Calling Agent ---
#
# This function uses the chain to obtain an initial response.
# If the response indicates a function call (by containing "CALL_FUNCTION:"), 
# it parses the request, calls the appropriate API tool, and feeds the function 
# result back into the chain to generate a final answer.
#
async def function_calling_agent(query: str, websocket: WebSocket):
    # Prepare the prompt using current conversation history and the new query
    conversation_context = memory.buffer
    full_prompt = prompt.format(history=conversation_context, query=query)
    # Get initial response from the LLM (using our async wrapper)
    initial_response_text = await chain.llm._acall(full_prompt)
    # print(f"init resp {initial_response_text}")
    
    # Check if the response instructs to call a function
    if initial_response_text.startswith("CALL_FUNCTION:"):
        try:
            # Extract the JSON details after the marker
            function_call_str = initial_response_text[len("CALL_FUNCTION:"):].strip()
            function_call_data = json.loads(function_call_str)
            function_name = function_call_data.get("name")
            function_args = function_call_data.get("arguments", {})

            # Based on the function name, call the corresponding API function
            if function_name == "get_pipeline_projects":
                function_result = await get_pipeline_projects(function_args)
            elif function_name == "get_completed_activities":
                function_result = await get_completed_activities(function_args)
            elif function_name == "get_non_completed_activities":
                function_result = await get_non_completed_activities(function_args)
            elif function_name == "get_completed_projects":
                function_result = await get_completed_projects(function_args)
            elif function_name == "process_stage_wise_counts":
                api_response = await get_pipeline_projects(function_args)
                meta_data = api_response["data"]["meta"]

                # Call the processing function
                final_response = process_stage_wise_counts(meta_data)
                function_result = json.dumps(final_response, indent=2) if isinstance(final_response, dict) else final_response
            elif function_name == "move_to_initial_stage":
                # For demonstration, we simulate prompting for missing data.
                # In practice, you might send a message to the user requesting this info.
                await websocket.send_text("Please provide the Project ID to move to initial stage:")
                alternative_id = (await websocket.receive_text()).strip()
                # Using a fixed stage_id here (could be dynamic)
                stage_id = "9db081ef-6281-489f-945e-b7d8ef487e4d"
                function_args = {"operation": {"ubiquity": False}, "alternative_id": alternative_id, "stage_id": stage_id}
                function_result = await move_to_initial_stage(function_args)
            elif function_name == "put_project_on_hold":
                await websocket.send_text("Please provide the Deal ID:")
                alternative_id = (await websocket.receive_text()).strip()
                await websocket.send_text("Please provide the On-Hold Review Date (YYYY-MM-DD):")
                review_date = (await websocket.receive_text()).strip() + "T18:30:00.000Z"
                function_args = {"alternative_id": alternative_id, "onhold_review_date": review_date}
                function_result = await put_project_on_hold(function_args)
            elif function_name == "get_bathroom_scopes":
                await websocket.send_text("Please provide the stage:")
                stage = (await websocket.receive_text()).strip()
                function_args = stage
                function_result = get_bathroom_scopes(function_args)
            elif function_name == "get_bedroom_scopes":
                await websocket.send_text("Please provide the stage:")
                stage = (await websocket.receive_text()).strip()
                function_args = stage
                function_result = get_bedroom_scopes(function_args)
            elif function_name == "get_basement_scopes":
                await websocket.send_text("Please provide the stage:")
                stage = (await websocket.receive_text()).strip()
                function_args = stage
                function_result = get_basement_scopes(function_args)
            else:
                await websocket.send_text("Unknown function call requested.")
                return

            # Incorporate the function result back into the conversation and re-run the chain
            new_query = f"Function {function_name} returned: {json.dumps(function_result)}"
            # Update conversation memory with the function call result
            memory.save_context({"query": query}, {"output": initial_response_text})
            memory.save_context({"query": new_query}, {"output": ""})
            final_prompt = prompt.format(history=memory.buffer, query=new_query)
            final_response_text = await chain.llm._acall(final_prompt)
            await stream_response(websocket, final_response_text)
            await websocket.send_text(f"API Response: {json.dumps(function_result, indent=2)}")
        except Exception as e:
            await websocket.send_text(f"Error processing function call: {e}")
    else:
        # If no function call was triggered, stream the response directly.
        await stream_response(websocket, initial_response_text)

# --- WebSocket Endpoint ---
#
# The WebSocket first asks for a Cookie (in JSON format) so the global headers can be updated.
# Then it enters a loop to process user queries using the function calling agent.
#
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connections.add(websocket)

    try:
        # Step 1: Send a welcome message.
        await websocket.send_text("I'm your Control Assistant. Please tell me your query:")

        # Step 2: Ask for the Cookie value.
        await websocket.send_text("Please provide the Cookie value in JSON format, e.g.: {'message': '<cookie_value>'}")

        # Step 3: Receive and process the Cookie value.
        cookie_data = await websocket.receive_text()
        if cookie_data.strip().lower() == "exit":
            await websocket.send_text("Goodbye! Disconnecting from the server.")
            connections.remove(websocket)
            await websocket.close()
            return

        try:
            cookie_dict = json.loads(cookie_data)
            cookie_value = cookie_dict.get("message")
            if not cookie_value:
                raise ValueError("Cookie value is missing in the provided JSON.")

            # Update the global HEADERS
            global HEADERS
            HEADERS["Cookie"] = cookie_value

            # Confirm update with the client.
            await websocket.send_text(f"Cookie value set successfully. Updated headers: {HEADERS}")
        except json.JSONDecodeError:
            await websocket.send_text("Invalid JSON format. Please send the Cookie as: {'message': '<cookie_value>'}")
            return
        except ValueError as e:
            await websocket.send_text(f"Error: {e}")
            return

        # Step 4: Handle user queries in a loop.
        while True:
            data = await websocket.receive_text()
            if data.strip().lower() == "exit":
                await websocket.send_text("Goodbye! Disconnecting from the server.")
                connections.remove(websocket)
                await websocket.close()
                break

            # Process the query via the function calling agent.
            await function_calling_agent(data, websocket)

    except WebSocketDisconnect:
        if websocket in connections:
            connections.remove(websocket)
        await websocket.close()

# --- Run the Application ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

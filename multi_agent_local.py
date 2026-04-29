from typing import Annotated, Literal, Sequence, TypedDict
import operator
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --- 1. Define State ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


# using llama3.1 as LLM 
llm = ChatOllama(model="llama3.1", temperature=0)

# --- 3. Define Nodes (Agents) ---

# A. Supervisor Node
class Router(BaseModel):
    next: Literal["Researcher", "Writer", "FINISH"]
supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a supervisor managing a Researcher and a Writer. "
               "Given the conversation history, decide who should act next. "
               "If the user asks a new question, route to the 'Researcher' to gather facts. "
               "If the Researcher has provided facts, route to the 'Writer' to draft the response. "
               "If the Writer has provided a satisfactory draft that answers the user, respond with 'FINISH'."),
    MessagesPlaceholder(variable_name="messages"),
])

# LangChain's ChatOllama 
supervisor_chain = supervisor_prompt | llm.with_structured_output(Router)

def supervisor_node(state: AgentState):
    result = supervisor_chain.invoke(state)
    return {"next": result.next}

# B. Researcher Node
researcher_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Researcher. Provide detailed, bulleted facts about the user's request. "
               "Prefix your response with 'RESEARCHER NOTES:\n'."),
    MessagesPlaceholder(variable_name="messages"),
])

researcher_chain = researcher_prompt | llm

def researcher_node(state: AgentState):
    result = researcher_chain.invoke(state)
    return {"messages": [AIMessage(content=result.content, name="Researcher")]}

# C. Writer Node
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Writer. Using the facts provided by the Researcher, draft a concise, engaging response for the user. "
               "Prefix your response with 'FINAL DRAFT:\n'."),
    MessagesPlaceholder(variable_name="messages"),
])

writer_chain = writer_prompt | llm

def writer_node(state: AgentState):
    result = writer_chain.invoke(state)
    return {"messages": [AIMessage(content=result.content, name="Writer")]}

# --- 4. Build the Graph ---
builder = StateGraph(AgentState)

builder.add_node("Supervisor", supervisor_node)
builder.add_node("Researcher", researcher_node)
builder.add_node("Writer", writer_node)

# --- 5. Add Edges (Logic Loops) ---
builder.add_edge(START, "Supervisor")

builder.add_conditional_edges(
    "Supervisor",
    lambda state: state["next"],
    {
        "Researcher": "Researcher",
        "Writer": "Writer",
        "FINISH": END
    }
)

builder.add_edge("Researcher", "Supervisor")
builder.add_edge("Writer", "Supervisor")

# --- 6. Compile with Memory ---
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# --- 7. Execution Run ---
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "local_llama_run"},"recursion_limit": 10}

    print("=== RUN 1: INITIAL REQUEST ===")
    request1 = HumanMessage(content="I need a quick summary of the Apollo 11 mission.")
    for event in graph.stream({"messages": [request1]}, config):
        for node, values in event.items():
            print(f"\n[{node}] Action:")
            if "messages" in values:
                print(values["messages"][-1].content)
            elif "next" in values:
                print(f"Routing task to -> {values['next']}")
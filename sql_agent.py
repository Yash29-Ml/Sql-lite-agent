import streamlit as st
from typing import TypedDict

from sqlalchemy import create_engine, text

from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, END


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="SQLite SQL Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 SQLite SQL Agent")
st.caption("LangGraph + Ollama + SQLite + Streamlit")


# ============================================================
# 1. DATABASE
# ============================================================

engine = create_engine("sqlite:///company.db")

db = SQLDatabase(engine)


# ============================================================
# 2. CREATE DATABASE / TABLE
# ============================================================

def create_database():

    with engine.begin() as connection:

        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT NOT NULL,
                salary REAL NOT NULL
            )
        """))

        # Check whether table already has data

        result = connection.execute(
            text("SELECT COUNT(*) FROM employees")
        )

        count = result.scalar()

        if count == 0:

            connection.execute(text("""
                INSERT INTO employees
                (name, department, salary)
                VALUES
                ('Rahul', 'IT', 75000),
                ('Priya', 'HR', 60000),
                ('Amit', 'IT', 85000),
                ('Sneha', 'Finance', 70000),
                ('Rohit', 'Sales', 55000),
                ('Neha', 'IT', 90000),
                ('Vikas', 'Finance', 80000),
                ('Pooja', 'HR', 65000)
            """))


create_database()


# ============================================================
# 3. OLLAMA MODEL
# ============================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ============================================================
# 4. STATE
# ============================================================

class State(TypedDict):
    question: str
    sql: str
    result: str
    answer: str
    error: str
    attempts: int


# ============================================================
# 5. HELPER
# ============================================================

def get_text(response):

    """
    Works whether the model returns:
    
    AIMessage -> response.content
    
    or
    
    string -> response
    """

    if hasattr(response, "content"):
        return response.content

    return str(response)


def clean_sql(sql):

    sql = sql.strip()

    sql = sql.replace("```sql", "")
    sql = sql.replace("```SQL", "")
    sql = sql.replace("```", "")

    return sql.strip()


# ============================================================
# 6. GENERATE SQL
# ============================================================

def generate_sql(state: State):

    schema = db.get_table_info()

    prompt = f"""
You are an expert SQLite SQL generator.

Database schema:
{schema}

User question:
{state["question"]}

Your task:
Convert the user's question into ONE valid SQLite SELECT query.

Rules:
1. Use ONLY tables and columns from the schema.
2. Generate ONLY SELECT statements.
3. Never generate INSERT.
4. Never generate UPDATE.
5. Never generate DELETE.
6. Never generate DROP.
7. Never generate ALTER.
8. Never generate TRUNCATE.
9. Do not explain the query.
10. Do not use markdown.
11. Return ONLY SQL.

SQL:
"""

    response = llm.invoke(prompt)

    sql = clean_sql(get_text(response))

    return {
        "sql": sql,
        "error": "",
        "attempts": 0
    }


# ============================================================
# 7. EXECUTE SQL
# ============================================================

def execute_sql(state: State):

    try:

        sql = state["sql"].strip()

        sql_upper = sql.upper()

        # -----------------------------------------
        # Security
        # -----------------------------------------

        forbidden = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "ALTER",
            "TRUNCATE",
            "REPLACE",
            "CREATE"
        ]

        # Must start with SELECT

        if not sql_upper.startswith("SELECT"):

            return {
                "result": "",
                "error": "Only SELECT queries are allowed."
            }

        # Check forbidden operations

        if any(word in sql_upper for word in forbidden):

            return {
                "result": "",
                "error": "Forbidden SQL operation."
            }

        # -----------------------------------------
        # Execute
        # -----------------------------------------

        with engine.connect() as connection:

            result = connection.execute(
                text(sql)
            )

            rows = result.fetchall()

            columns = result.keys()

        # Convert result to readable format

        result_data = []

        for row in rows:

            result_data.append(
                dict(zip(columns, row))
            )

        return {
            "result": str(result_data),
            "error": ""
        }

    except Exception as e:

        return {
            "result": "",
            "error": str(e)
        }


# ============================================================
# 8. CHECK RESULT
# ============================================================

def check_result(state: State):

    if not state["error"]:
        return "answer"

    # Prevent infinite loop

    if state["attempts"] >= 3:
        return "answer"

    return "fix"


# ============================================================
# 9. FIX SQL
# ============================================================

def fix_sql(state: State):

    schema = db.get_table_info()

    prompt = f"""
You are an expert SQLite debugging agent.

Database schema:
{schema}

User question:
{state["question"]}

Previous SQL:
{state["sql"]}

SQLite error:
{state["error"]}

Fix the SQL query.

Rules:
1. Return ONLY one valid SQLite SELECT query.
2. Use ONLY tables and columns from the schema.
3. Never use INSERT.
4. Never use UPDATE.
5. Never use DELETE.
6. Never use DROP.
7. Never use ALTER.
8. Never use TRUNCATE.
9. Do not explain anything.
10. Do not use markdown.

Correct SQL:
"""

    response = llm.invoke(prompt)

    sql = clean_sql(get_text(response))

    return {
        "sql": sql,
        "error": "",
        "attempts": state["attempts"] + 1
    }


# ============================================================
# 10. FINAL ANSWER
# ============================================================

def final_answer(state: State):

    # If SQL failed after retries

    if state["error"]:

        return {
            "answer": (
                "I couldn't execute the SQL query successfully. "
                f"Error: {state['error']}"
            )
        }

    prompt = f"""
You are a helpful data analyst.

Answer the user's question using ONLY the database result.

User question:
{state["question"]}

SQL:
{state["sql"]}

Database result:
{state["result"]}

Rules:
1. Give a simple natural-language answer.
2. Use the database result as the source of truth.
3. Do not invent information.
4. Do not mention internal reasoning.
5. Do not unnecessarily explain SQL.
6. If there are no rows, say that no matching data was found.

Answer:
"""

    response = llm.invoke(prompt)

    answer = get_text(response).strip()

    return {
        "answer": answer
    }


# ============================================================
# 11. BUILD LANGGRAPH
# ============================================================

graph = StateGraph(State)

graph.add_node(
    "generate_sql",
    generate_sql
)

graph.add_node(
    "execute_sql",
    execute_sql
)

graph.add_node(
    "fix_sql",
    fix_sql
)

graph.add_node(
    "final_answer",
    final_answer
)


# ============================================================
# 12. GRAPH EDGES
# ============================================================

graph.set_entry_point("generate_sql")


graph.add_edge(
    "generate_sql",
    "execute_sql"
)


graph.add_conditional_edges(
    "execute_sql",
    check_result,
    {
        "fix": "fix_sql",
        "answer": "final_answer"
    }
)


graph.add_edge(
    "fix_sql",
    "execute_sql"
)


graph.add_edge(
    "final_answer",
    END
)


# Compile

app = graph.compile()


# ============================================================
# 13. STREAMLIT CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# 14. CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about employees..."
)


if question:

    # -----------------------------------------
    # USER MESSAGE
    # -----------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------------
    # AGENT
    # -----------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🤔 Thinking..."):

            result = app.invoke({
                "question": question,
                "sql": "",
                "result": "",
                "answer": "",
                "error": "",
                "attempts": 0
            })


        # -----------------------------------------
        # ANSWER
        # -----------------------------------------

        st.markdown(result["answer"])


        # -----------------------------------------
        # SQL
        # -----------------------------------------

        with st.expander("🔍 Generated SQL"):

            st.code(
                result["sql"],
                language="sql"
            )


        # -----------------------------------------
        # DATABASE RESULT
        # -----------------------------------------

        with st.expander("📊 Database Result"):

            st.write(result["result"])


        # -----------------------------------------
        # ERROR
        # -----------------------------------------

        if result["error"]:

            with st.expander("⚠️ Error"):

                st.error(result["error"])


    # -----------------------------------------
    # SAVE ANSWER
    # -----------------------------------------

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })
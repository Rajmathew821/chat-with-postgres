import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from contextlib import contextmanager
from dotenv import load_dotenv
import uvicorn
import re
from langchain_openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# FastAPI app
app = FastAPI(title="RAG Safety Stock API", version="1.0.0")

# Pydantic model for request validation
class QueryRequest(BaseModel):
    question: str

# Pydantic model for response
class QueryResponse(BaseModel):
    table_name: str
    sql_query: str
    results: List[Dict]
    summary: str

@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str

class DatabaseConnection:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.conn = None
        self.cursor = None
    
    @contextmanager
    def get_cursor(self):
        try:
            self.conn = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                cursor_factory=RealDictCursor
            )
            self.cursor = self.conn.cursor()
            yield self.cursor
            self.conn.commit()
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            if self.conn:
                self.conn.rollback()
            raise
        finally:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()

class RAGSafetyStockSystem:
    def __init__(self, db_config: DatabaseConfig, model_name: str = "meta-llama/Llama-2-7b"):
        self.db_config = db_config
        self.db_connection = DatabaseConnection(db_config)
        
        try:            
            self.llm = OpenAI(temperature=0.1, max_tokens=500)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise

        self.find_table_template = """You are a highly intelligent AI assistant that helps map user queries to the correct database table names based on intent and domain knowledge.

        Context:
        You are working with a business application that manages inventory, sales, and purchases. The database includes the following table names:
        - `current_stock`: Contains data about available products and their quantities.
        - `ageing_report`: Contains records of aged article bucket wise aging value like 0-30 days ,above 90 days etc.
        - `excess_stock`: Contains information on excess value and over stock article related.
        - `dead_stock`: Contains information on dead stock or out_dated article related.
        - `out_of_stock`: Contains information on out of stock article value.
        - `safty_stock`: Contains information on safty stock report and value of article with safty and reorder points

        Instructions:
        1. Read the user's input carefully and summarize their intent in one sentence.
        2. Based on that intent, return the most appropriate table name from the list above.
        3. If multiple tables are relevant, list all of them in order of relevance.
        4. If the intent does not clearly map to any table, respond with `unknown`.
        5. output should provide the exacy column name , dont provide any unnecessory content exact table in above table names is enough is string type

        Question : {question}

        Output :
        """
        self.find_table_prompt = PromptTemplate(
            input_variables=["question"],
            template=self.find_table_template)
        
        self.nl_to_sql_prompt = PromptTemplate(
            input_variables=["table_name","schema", "question"],
            template="""
        You are an expert SQL query generator for a PostgreSQL database. Convert the natural language question into a valid SQL query using ONLY the provided table schema. Follow these rules strictly:
        Schema for {table_name}:
        {schema}
        Rules:
            1. Generate only valid PostgreSQL syntax.
            2. Use proper table and column names from the schema.
            3. Include appropriate WHERE clauses and JOINs if needed.
            4. Avoid SQL injection vulnerabilities.
            5. Return only the SQL query without any explanation.
            6. Strictly provide the correct SQL query based on the user-provided input question, ensuring it matches PostgreSQL syntax.
            7. Dont use underscore for any of the column name , column names should follow exact from table schema 
            8. If user query is invalid then give the Response : 'No queries matched from the your input'
            9. column name in generated sql query should be same in table schema, dont provide unnecessory column names exact column names from shema is enough.
            
        - If a column name contains a space (e.g., 'stock value'), enclose it in double quotes (e.g., "stock value").
        - Map 'laptops' or similar terms to article_name (e.g., article name = 'laptop').
        - Map 'premium' or similar terms to segment name (e.g., segment name = 'premium').
        - Avoid SQL injection.
        - Return clear, readable results.
        - Handle aggregations, filters, and joins appropriately.
        - If the question references columns not in the schema (e.g., department_name) or is unrelated to the database (e.g., general knowledge), return EXACTLY: No relevant SQL query can be generated.
        - Output the SQL query as plain text, without quotes, markdown, comments, or explanations.
        - For questions asking for "top N" or "more" of a value, order by that column in DESCENDING order and use LIMIT N.

        Question: {question}

        Response : SQL Query
        """
        )
        self.context_to_summary_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            Given the following user question and database context:
            Question: {question}
            Context: {context}
            
            Generate a concise, business-oriented summary of the data in natural language with accurate quantities or values from the context. Then, provide a list of values with their corresponding values based on the user's query.
            Rules:
            1. Provide accurate information based on the context
            2. Use professional business language
            3. Summarize key insights concisely
            4. Avoid technical jargon
            5. if the user query is to give list then you should provide he answer in bulletin format.
            Summary:
            """
        )
        self.find_table_chain = LLMChain(llm=self.llm, prompt=self.find_table_prompt)
        self.nl_to_sql_chain = LLMChain(llm=self.llm, prompt=self.nl_to_sql_prompt)
        self.summary_chain = LLMChain(llm=self.llm, prompt=self.context_to_summary_prompt)

    def find_table(self, question: str) -> str:
        try:
            response = self.find_table_chain.run(
                question=question
            )
            # matched_part = response.split("Matched Table:")[-1]
            # table_list = [table.strip().strip('`') for table in matched_part.split(',')]
            # table_list = table_list[0]
            return response
        except Exception as e:
            logger.error(f"Error Finding the table name: {str(e)}")
            raise

    def find_schema(self, table_name: str) -> str:
        try:
            with self.db_connection.get_cursor() as cursor:
                schema_query = f"""
                SELECT column_name, data_type FROM information_schema.columns
                WHERE table_name = '{table_name}'
                AND table_schema = 'public'
                ORDER BY ordinal_position;
                """
                cursor.execute(schema_query)


                results = cursor.fetchall()
                type_mapping = {
                    'text': 'TEXT',
                    'bigint': 'BIGINT',
                    'double precision': 'DOUBLE'
                }
                column_definitions = []
                for col in results:
                    col_name_raw = col['column_name']
                    col_name = f'"{col_name_raw}"' if ' ' in col_name_raw else col_name_raw
                    col_type = type_mapping.get(col['data_type'], col['data_type'])
                    column_definitions.append(f"{col_name} {col_type}")
                formatted = ", \n".join(column_definitions)    

                schema_query = f"CREATE TABLE {table_name}({str(formatted)});"
                schema_query = schema_query.replace('\n','')

                converted_sql = re.sub(r'"([^"]+)"', r"'\1'", schema_query)
                return converted_sql
            
        except Exception as e:
            logger.error(f"Error for finding the table schema: {str(e)}")
            raise

    def execute_sql_query(self, query: str) -> List[Dict]:
        try:
            with self.db_connection.get_cursor() as cursor:
                cursor.execute(query)
                results = cursor.fetchall()
                return results
        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            raise
    
    def generate_sql_query(self, question: str,table_schema: str,table_name: str) -> str:
        try:
            response = self.nl_to_sql_chain.run(
                schema=table_schema,
                table_name=table_name,
                question=question
            )
            sql_query = response.split("SQL Query:")[-1].strip()
            sql_query = sql_query.split(";")[0].strip().replace("\n"," ")
            return sql_query
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            raise

    def generate_summary(self, question: str, context: List[Dict]) -> str:
        try:
            context_str = "\n".join([str(row) for row in context])
            response = self.summary_chain.run(
                question=question,
                context=context_str
            )
            summary = response.split("Summary:")[-1].strip()
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise            

    def process_query(self, question: str) -> Dict[str, Any]:
        try:
            table_name = self.find_table(question)

            table_name = table_name.lower().strip()
            logger.info(f"Found Table Name: {table_name}")

            table_schema = self.find_schema(table_name)
            logger.info(f"Found Table Schema: {table_schema}")

            sql_query = self.generate_sql_query(question,table_schema,table_name)
            logger.info(f"Generated SQL query: {sql_query}")
            
            results = self.execute_sql_query(sql_query)
            logger.info(f"Retrieved {len(results)} records from database")
            
            summary = self.generate_summary(question, results)
            
            return {
                "table_name": table_name,
                "sql_query": sql_query,
                "results": results,
                "summary": summary
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

db_config = DatabaseConfig(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    database=os.getenv("DB_NAME", "rag_db"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASSWORD", "123456")
)

rag_system = RAGSafetyStockSystem(db_config)

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        result = rag_system.process_query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8018)  

import os
from pinecone import Pinecone
from google import genai
from google.genai import types # Added for configuration types

class CareerAdviceRAG:
    def __init__(self, pinecone_api_key, google_api_key):
        # 1. Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index_name = "index"
        self.index = self.pc.Index(self.index_name)

        # 2. Initialize Google GenAI
        self.client = genai.Client(api_key=google_api_key)

    def generate_response(self, user_input):
        # 3. Create Embedding for the user's question
        # CRITICAL: Added output_dimensionality=1024 to match your Pinecone index
        embed_response = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=user_input,
            config=types.EmbedContentConfig(
                output_dimensionality=1024,
                task_type="RETRIEVAL_QUERY"
            )
        )
        query_vector = embed_response.embeddings[0].values

        # 4. Query Pinecone
        query_results = self.index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )

        # 5. Extract text from metadata
        relevant_chunks = []
        for match in query_results['matches']:
            content = match['metadata'].get('text', 'Snippet found, but no text field available.')
            relevant_chunks.append(content)

        # 6. Generate Career Advice
        context = "\n\n".join(relevant_chunks)
        prompt = f"""
        You are a career advisor. Use these YFIOB podcast insights:
        {context}

        Suggest career paths for a student with these interests: {user_input}
        """

        # Using the latest stable gemini-2.5-flash
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return relevant_chunks, response.text

    def clear_conversation(self):
        pass
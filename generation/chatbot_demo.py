import os
import logging
from openai import OpenAI
import chromadb
from chromadb import Client
from chromadb.config import Settings 
from dotenv import load_dotenv
#from build_vector_db import get_embedding
import datetime # 시간 기록용 (선택적)


dbclient = chromadb.PersistentClient(path="./data/chatbot2/chroma_db")
collection = dbclient.get_or_create_collection("rag_collection")

load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
client=OpenAI(api_key=api_key) 

def get_embedding(text, model="text-embedding-3-large"):
    # do it
    response=client.embeddings.create(input=[text], model=model)
    embedding=response.data[0].embedding
    return embedding


# --- ChromaDB 설정 ---
# ***** ChromaDB 로깅 레벨 설정 (INFO 메시지 숨기기) *****
# ChromaDB 클라이언트 초기화 전에 로거 레벨을 WARNING으로 설정
logging.getLogger('chromadb').setLevel(logging.WARNING)
# 경우에 따라 하위 모듈 로거 레벨도 조정해야 할 수 있습니다.
logging.getLogger('chromadb.db.duckdb').setLevel(logging.WARNING)
logging.getLogger('chromadb.api.segment').setLevel(logging.WARNING)
# ----------------------------------------------------

chroma_path = "./data/chatbot2/chroma_db"
if not os.path.exists(chroma_path):
    os.makedirs(chroma_path)
    #print(f"Created ChromaDB directory at: {chroma_path}")

try:
    dbclient = chromadb.PersistentClient(path=chroma_path)
    collection = dbclient.get_or_create_collection("rag_collection")
    #print(f"ChromaDB client connected. Collection 'rag_collection' loaded/created.")
except Exception as e:
    print(f"Error connecting to ChromaDB or getting collection: {e}")
    exit()

# --- OpenAI 클라이언트 설정 ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable not set.")
    exit()
openai_client = OpenAI(api_key=api_key)
print("OpenAI client initialized.")


# --- 함수 정의 ---

def retrieve(query, top_k=5):
    """query를 임베딩해 chroma에서 가장 유사도가 높은 top-k개의 문서 가져오는 함수"""
    try:
        query_embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas'] # 필요한 정보 명시
        )
         # 결과 구조 확인 및 안전한 접근
        if results and results.get("documents") and results.get("metadatas"):
             # print(f"Retrieved {len(results['documents'][0])} documents for query: {query[:50]}...") # 디버깅용
             return results
        else:
             print(f"Warning: No results or unexpected format from ChromaDB for query: {query[:50]}...")
             return {"ids": [[]], "embeddings": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    except Exception as e:
        print(f"Error during ChromaDB retrieval: {e}")
        return {"ids": [[]], "embeddings": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

# ***** 수정된 함수 *****
def generate_answer_with_context(query, conversation_history, top_k=5):
    """
    1) query에 대해 벡터 DB에서 top_k개 문서 retrieval
    2) 검색된 문서들과 대화 기록(context)을 함께 GPT에 prompt
    3) 최종 답변 반환하는 함수
    """
    #print(f"\n[generate_answer_with_context] Generating answer for query: {query[:50]}...")

    # 1. Retrieve relevant documents (기존 로직 유지)
    results = retrieve(query, top_k)
    found_docs = results["documents"][0] if results and results["documents"] else []
    found_metadatas = results["metadatas"][0] if results and results["metadatas"] else []
    #print(f"Retrieved {len(found_docs)} static documents.")

    # 2. Construct context string from retrieved documents (기존 로직 유지)
    context_texts = []
    if found_docs:
        for doc_text, meta in zip(found_docs, found_metadatas):
            filename = meta.get('filename', 'N/A') # filename 없을 경우 대비
            context_texts.append(f"<<문서 출처: {filename}>>\n{doc_text}")
        document_context_str = "\n\n".join(context_texts)
    else:
        document_context_str = "관련 문서를 찾지 못했습니다."
        print("No relevant static documents found.")

    # 3. Prepare the prompt including conversation history
    system_prompt = """
    당신은 주어진 문서 정보와 이전 대화 내용을 바탕으로 사용자 질문에 답변하는 지능형 어시스턴트입니다. 다음 원칙을 지키세요:

    1. 제공된 **문서 내용**과 **이전 대화**에 근거해서 답변을 작성하세요.
    2. 문서나 **이전 대화**에 언급되지 않은 내용이라면, 잘 모르겠다고 답변해줘.
    4. 지나치게 장황하지 않게, 간결하고 알기 쉽게 설명하세요.
    5. 사용자가 질문을 한국어로 한다면, 한국어로 답변하고, 다른 언어로 질문한다면 해당 언어로 답변하도록 노력하세요.
    6. 답변이 특정 문서 내용을 참조했다면, 가능한 출처(예: <<문서 출처: 파일명>>)를 언급하세요.
    7. **이전 대화**에 대하여 직접적으로 언급한다면, 그 내용을 바탕으로 답변을 생성하세요.
    8. **문서 내용**과 직접적으로 관련이 없는 내용이더라도 최대한 친절하게 설명해줘. 다만 정보에 관해서 묻는 내용이라면 잘 모르겠다고 답변해줘.
    9. 만약 사용자가 본인에 대하여 얘기한다면 그 내용에 공감하고 질문을 해주세요.

    당신은 전문적인 지식을 갖춘 듯 정확하고, 동시에 친절하고 이해하기 쉬운 어투를 구사합니다. 이전 대화의 맥락을 잘 파악하여 답변하세요.
    """

    # Construct messages list for OpenAI API
    messages = [{"role": "system", "content": system_prompt}]

    # Add limited conversation history to messages
    # conversation_history 는 [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}] 형식
    history_limit = 10 # 최근 N개 턴 (user + assistant = 1턴) 기억 -> 실제 메시지 수는 *2
    limited_history = conversation_history[-(history_limit * 2):]
    messages.extend(limited_history) # ***** 대화 기록 추가 *****
    #print(f"Added {len(limited_history)} messages from conversation history to prompt.")

    # Add the current user query along with the retrieved document context
    user_prompt_content = f"""--- 검색된 문서 내용 ---
{document_context_str}
---
위 문서 내용과 이전 대화를 바탕으로 다음 질문에 답변해 주세요: {query}"""

    messages.append({"role": "user", "content": user_prompt_content})

    # 4. Call OpenAI API
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7, # 필요시 조절
        )
        answer = response.choices[0].message.content
        #print("[generate_answer_with_context] Successfully received response from OpenAI.")
        return answer

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."


# ***** 수정된 메인 실행 부분 *****
if __name__ == "__main__":
    # ***** 대화 기록을 저장할 리스트 초기화 *****
    conversation_history = []
    print("\n멀티턴 RAG 챗봇을 시작합니다. (종료: 'quit' 또는 '종료')")

    while True:
        user_query = input("\n당신: ")
        if user_query.lower() in ["quit", "종료"]:
            #print("챗봇: 대화를 종료합니다. 이용해주셔서 감사합니다.")
            break

        # ***** 대화 기록과 함께 답변 생성 함수 호출 *****
        answer = generate_answer_with_context(user_query, conversation_history, top_k=3)

        print("\n챗봇:", answer)

        # ***** 현재 턴의 사용자 질문과 봇 답변을 대화 기록에 추가 *****
        conversation_history.append({"role": "user", "content": user_query})
        conversation_history.append({"role": "assistant", "content": answer})

        # (선택적) 대화 기록 길이 제한 (메모리 관리 및 토큰 제한 방지)
        MAX_HISTORY_LENGTH = 15 # 최근 10턴 (user+assistant) 유지
        if len(conversation_history) > MAX_HISTORY_LENGTH * 2:
            # 오래된 기록부터 제거 (리스트 앞부분 제거)
            conversation_history = conversation_history[-(MAX_HISTORY_LENGTH * 2):]
            # print(f"[History trimmed. Current length: {len(conversation_history)} messages]") # 디버깅용
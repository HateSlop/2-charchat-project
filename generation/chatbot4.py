import os
from openai import OpenAI
from dotenv import load_dotenv

# 환경변수 로드 및 OpenAI 클라이언트 초기화
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def generate_response(user_message):
    # (1) 시스템 프롬프트 정의 (원하는 설정 유지)
    system_prompt = (
        "너는 서강대학교 새내기 여학생 알로스야. 호기심 많은 평범한 말투를 사용해. 너의 뻔선과 대화를 하는 상황이야. 뻔선이란, 학번이 같은 선배를 의미해. 서강대학교에서는 뻔선뻔후 제도가 존재하는데, 학번이 같은 선후배끼리 가족처럼 챙겨주는 관습이야."
        "너는 자유전공학부 신입생이야. 따라서 2년 뒤에 전공을 정해야해. 그리고 너는 새로운 환경에서의 적응, 다양한 인간관계, 학업과 동아리 활동에 관심이 많아."
    )
    
    # (2) OpenAI Chat API 호출 (간단 버전)
    response = client.chat.completions.create(
        model="gpt-4o",  # 실제 사용하는 모델명에 맞게 조정
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    )
    
    # (3) 응답 추출
    reply = response.choices[0].message.content
    return reply

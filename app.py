import os
from flask import Flask, request, render_template, jsonify, url_for
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# 초기 화면: 챗봇 선택 페이지
@app.route('/')
def index():
    chatbots = [
        {
            'id': 1,
            'name': 'chatbot1',
            'image': url_for('static', filename='images/chatbot1/thumbnail.png'),
            'tags': ['#챗봇', '#유머', '#일상']
        },
        {
            'id': 2,
            'name': 'chatbot2',
            'image': url_for('static', filename='images/chatbot2/thumbnail.png'),
            'tags': ['#챗봇', '#유머', '#일상']
        },
        {
            'id': 3,
            'name': 'chatbot3',
            'image': url_for('static', filename='images/chatbot3/thumbnail.png'),
            'tags': ['#챗봇', '#유머', '#일상']
        },
        {
            'id': 4,
            'name': '내 뻔후는 알로스',
            'image': url_for('static', filename='images/chatbot4/thumbnail.png'),
            'tags': ['#서강대', '#자유전공학부', '#알로스']
        }
    ]
    return render_template('index.html', chatbots=chatbots)

# 챗봇 상세정보 페이지 (새로운 HTML로 구현)
@app.route('/detail/<int:bot_id>')
def detail(bot_id):
    chatbot_data = {
       1: {
           "name": "chatbot1",
           'image': url_for('static', filename='images/chatbot1/thumbnail.png'),
           "description": "chatbot1의 설명입니다.",
           'tags': ['#챗봇', '#유머', '#일상']
       },
       2: {
           "name": "chatbot2",
           'image': url_for('static', filename='images/chatbot2/thumbnail.png'),
           "description": "chatbot2의 설명입니다.",
           'tags': ['#챗봇', '#유머', '#일상']
       },
       3: {
           "name": "chatbot3",
           'image': url_for('static', filename='images/chatbot3/thumbnail.png'),
           "description": "chatbot3의 설명입니다.",
           'tags': ['#챗봇', '#유머', '#일상']
       },
       4: {
           "name": "내 뻔후는 알로스",
           'image': url_for('static', filename='images/chatbot4/thumbnail.png'),
           "description": "알로스는 서강대학교에 갓 입학한 자유전공학부 신입생이다. 새로운 환경에서의 적응, 다양한 인간관계, 학업과 동아리 활동의 과정을 거치며 자신의 전공을 선택해간다. 알로스는 과연 어떤 전공을 선택하고 적응해갈까? 사용자의 선택에 따라 다양한 미래가 펼쳐진다!.",
           'tags': ['#서강대', '#자유전공학부', '#알로스']
       },
    }
    bot = chatbot_data.get(bot_id)
    if not bot:
        return "Invalid bot id", 404
    return render_template('chatbot_detail.html', bot=bot, bot_id=bot_id)

# 공용 채팅 화면: URL의 bot_id에 따라 제목 등을 변경
@app.route('/chat/<int:bot_id>')
def chat(bot_id):
    chatbot_names = {
        1: "chatbot1",
        2: "chatbot2",
        3: "chatbot3",
        4: "내 뻔후는 알로스"
    }
    bot_name = chatbot_names.get(bot_id, "챗봇")
    return render_template('chat.html', bot_id=bot_id, bot_name=bot_name)

# API 엔드포인트: 전달된 bot_id에 따라 해당 모듈의 응답 생성 함수를 호출
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json()
    user_message = data.get('message')
    try:
        bot_id = int(data.get('bot_id'))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid bot id'}), 400

    try:
        if bot_id == 1:
            from generation.chatbot1 import generate_response
            reply = generate_response(user_message)
        elif bot_id == 2:
            from generation.chatbot2 import generate_response
            reply = generate_response(user_message)
        elif bot_id == 3:
            from generation.chatbot3 import generate_response
            reply = generate_response(user_message)
        elif bot_id == 4:
            from generation.chatbot4 import generate_response
            reply = generate_response(user_message)
        else:
            return jsonify({'error': 'Invalid bot id'}), 400

        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

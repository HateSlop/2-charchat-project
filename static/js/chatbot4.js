// chat-area에서 bot_id와 bot_image_url 정보 꺼내기
const chatArea = document.querySelector('.chat-area');
const botId = chatArea.dataset.botId;
const botImageUrl = chatArea.dataset.botImageUrl;

// 주요 DOM 요소
const chatLog = document.getElementById('chat-log');
const userMessageInput = document.getElementById('user-message');
const sendBtn = document.getElementById('send-btn');
const videoBtn = document.getElementById('videoBtn');
const imageBtn = document.getElementById('imageBtn');

// 메시지 전송 함수
async function sendMessage() {
  const message = userMessageInput.value.trim();
  if (!message) return;

  appendMessage('user', message);
  userMessageInput.value = '';

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bot_id: botId, message: message })
    });
    const data = await response.json();
    if (data.reply) {
      appendMessage('bot', data.reply);
    } else if (data.error) {
      appendMessage('bot', 'Error: ' + data.error);
    }
  } catch (err) {
    appendMessage('bot', 'Error: 요청 실패');
  }
}

// 메시지 DOM에 추가
function appendMessage(sender, text) {
  const messageElem = document.createElement('div');
  messageElem.classList.add('message', sender);

  if (sender === 'user') {
    // 사용자 메시지
    messageElem.textContent = text;
  } else {
    // 봇 메시지
    const botImg = document.createElement('img');
    botImg.classList.add('bot-big-img');
    botImg.src = botImageUrl;
    botImg.alt = "챗봇 이미지";

    const messageText = document.createElement('div');
    messageText.classList.add('bot-text');
    messageText.textContent = text;

    messageElem.appendChild(botImg);
    messageElem.appendChild(messageText);
  }

  chatLog.appendChild(messageElem);
  chatLog.scrollTop = chatLog.scrollHeight;
}

// 엔터키 또는 전송 버튼으로 전송
userMessageInput.addEventListener('keyup', (event) => {
  if (event.key === 'Enter') {
    sendMessage();
  }
});
sendBtn.addEventListener('click', sendMessage);

// 모달 열기/닫기
function openModal(modalId) {
  document.getElementById(modalId).style.display = 'block';
}
function closeModal(modalId) {
  document.getElementById(modalId).style.display = 'none';
}

// 미디어 버튼
videoBtn.addEventListener('click', () => {
  openModal('videoModal');
});
imageBtn.addEventListener('click', () => {
  openModal('imageModal');
});

// 모달 닫기 버튼들
document.querySelectorAll('.modal-close').forEach(btn => {
  btn.addEventListener('click', () => {
      const modal = btn.closest('.modal');
      closeModal(modal.id);
  });
});

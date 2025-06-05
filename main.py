import os
import asyncio
import logging
from dotenv import load_dotenv
import httpx
from datetime import datetime, timedelta

from ai import OpenAIAPI

AI_MODEL="openai/gpt-4o-mini"  # Модель ИИ с сайта vsegpt.ru
AI_ASSISTANT=OpenAIAPI()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("avito_bot")

# Загрузка переменных окружения
load_dotenv()

class AvitoClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://api.avito.ru"
        self.token = None
        self.token_expires = None
        self.http_client = None
    
    async def __aenter__(self):
        self.http_client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        if self.http_client:
            await self.http_client.aclose()
    
    async def get_token(self):
        """Получение и обновление OAuth токена для Avito API"""
        url = f"{self.base_url}/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        try:
            response = await self.http_client.post(url, data=data)
            response.raise_for_status()
            token_data = response.json()
            
            self.token = token_data['access_token']
            # Устанавливаем время истечения токена (обычно 1 час)
            self.token_expires = datetime.now() + timedelta(seconds=token_data.get('expires_in', 3600) - 300)  # -5 минут для запаса
            logger.info("Успешно получен новый access_token для Avito API")
            return self.token
        except Exception as e:
            logger.error(f"Ошибка при получении токена: {e}")
            raise

    async def ensure_token_valid(self):
        """Проверяет валидность токена и обновляет при необходимости"""
        if not self.token or (self.token_expires and datetime.now() >= self.token_expires):
            await self.get_token()
    
    async def request(self, method: str, endpoint: str, **kwargs):
        """Выполнение запроса к API Avito"""
        await self.ensure_token_valid()
        
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        if "headers" in kwargs:
            headers.update(kwargs["headers"])
            del kwargs["headers"]
        
        try:
            response = await getattr(self.http_client, method.lower())(
                url, headers=headers, **kwargs
            )
            
            # Если получили 401, пробуем обновить токен и повторить запрос
            if response.status_code == 401:
                logger.warning("Токен недействителен, пробуем обновить...")
                await self.get_token()
                headers["Authorization"] = f"Bearer {self.token}"
                response = await getattr(self.http_client, method.lower())(
                    url, headers=headers, **kwargs
                )
            
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP ошибка {e.response.status_code}: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при выполнении запроса: {e}")
            raise
    
    async def get_chats(self, user_id: int, limit: int = 50, offset: int = 0):
        """Получение списка непрочитанных чатов"""
        endpoint = f"/messenger/v2/accounts/{user_id}/chats"
        params = {"unread_only": True,"limit": limit, "offset": offset}
        return await self.request("GET", endpoint, params=params)
    
    async def get_messages(self, user_id: int, chat_id: str, limit: int = 50, offset: int = 0):
        """Получение сообщений в чате"""
        endpoint = f"/messenger/v3/accounts/{user_id}/chats/{chat_id}/messages"
        params = {"limit": limit, "offset": offset}
        return await self.request("GET", endpoint, params=params)
    
    async def send_message(self, user_id: int, chat_id: str, text: str):
        """Отправка текстового сообщения"""
        endpoint = f"/messenger/v1/accounts/{user_id}/chats/{chat_id}/messages"
        payload = {
            "message": {
                "text": text
            },
            "type": "text"
        }
        return await self.request("POST", endpoint, json=payload)
    
    async def mark_chat_as_read(self, user_id: int, chat_id: str):
        """Отметить чат как прочитанный"""
        endpoint = f"/messenger/v1/accounts/{user_id}/chats/{chat_id}/read"
        return await self.request("POST", endpoint, json={})

async def generate_response(text: str, chat_id: str) -> str:
    """Генерация ответа с помощью ChatGPT"""
    response = AI_ASSISTANT.get_response(text=text, chat_id=chat_id, model=AI_MODEL, max_token=5000)
    response_message = response['message']
    return response_message

async def process_new_messages(avito_client: AvitoClient, user_id: int):
    """Обработка новых сообщений"""
    try:
        # Получаем список чатов
        chats_response = await avito_client.get_chats(user_id)
        chats = chats_response.get('chats', [])
        
        if not chats:
            logger.info("Нет активных чатов")
            return
        
        logger.info(f"Найдено {len(chats)} чатов")
        
        for chat in chats:
            chat_id = chat.get('id')
            if not chat_id:
                continue
            
            # Получаем сообщения в чате
            messages_response = await avito_client.get_messages(user_id, chat_id)
            messages = messages_response.get('messages', [])
            
            if not messages:
                continue
            
            logger.info(f"Чат {chat_id}: {len(messages)} сообщений")
            
            for message in messages:
                # Обрабатываем только входящие непрочитанные сообщения
                if message.get('direction') != 'in' or message.get('is_read'):
                    continue
                
                text = message.get('content', {}).get('text', '')
                if not text:
                    continue
                
                logger.info(f"Новое сообщение в чате {chat_id}: {text[:50]}...")
                
                # Генерация ответа
                response = await generate_response(text, chat_id)
                
                # Отправка ответа
                try:
                    await avito_client.send_message(user_id, chat_id, response)
                    logger.info(f"Ответ отправлен в чат {chat_id}")
                except Exception as e:
                    logger.error(f"Ошибка при отправке сообщения: {e}")
                
                # Отмечаем чат как прочитанный
                try:
                    await avito_client.mark_chat_as_read(user_id, chat_id)
                    logger.info(f"Чат {chat_id} отмечен как прочитанный")
                except Exception as e:
                    logger.error(f"Ошибка при отметке чата как прочитанного: {e}")
    
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщений: {e}")

async def polling_loop():
    """Основной цикл опроса новых сообщений"""
    client_id = os.getenv("AVITO_CLIENT_ID")
    client_secret = os.getenv("AVITO_CLIENT_SECRET")
    user_id = os.getenv("AVITO_USER_ID")  # ID пользователя в Авито
    
    if not all([client_id, client_secret, user_id]):
        raise ValueError("Не все переменные окружения установлены")
    
    try:
        async with AvitoClient(client_id, client_secret) as avito_client:
            while True:
                try:
                    await process_new_messages(avito_client, int(user_id))
                except Exception as e:
                    logger.error(f"Ошибка в цикле опроса: {e}")
                
                # Пауза между запросами (60 секунд)
                await asyncio.sleep(60)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(polling_loop())
    except KeyboardInterrupt:
        logger.info("Бот остановлен")
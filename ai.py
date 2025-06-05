import os
import base64

import dotenv
import openai

dotenv.load_dotenv()

PROMPT = """Ты помощник-ассистент Авито. Тебе нужно отвечать на вопросы людей. Отвечай вежливо, не сильно много. Если что-то не знаешь - говори, что передашь владельцу."""

openai.api_key = os.getenv("OPENAI_API_KEY")

ASSISTANT_PROMPT = PROMPT
ANALYTIC_PROMPT = 'settings.ANALYTIC_PROMPT'

openai.base_url = "https://api.vsegpt.ru:6070/v1/"


class BaseAIAPI:
    def __init__(self, ) -> None:
        self._ASSISTANT_PROMPT: str = ASSISTANT_PROMPT
        self.chat_history: dict = {}
        self._TEMPERATURE = 0.7

    def clear_chat_history(self, chat_id: int) -> None:
        self.chat_history.pop(chat_id)


class OpenAIAPI(BaseAIAPI):
    """API for working with https://vsegpt.ru/Docs/API"""

    def __init__(self, ) -> None:
        super().__init__()

    def _get_or_create_user_chat_history(self, chat_id: int, new_user_message: str = "") -> list:
        if not self.chat_history.get(chat_id, False):
            self.chat_history[chat_id] = []
            self.chat_history[chat_id].append({"role": "system", "content": self._ASSISTANT_PROMPT})
            self.chat_history[chat_id].append({"role": "user", "content": new_user_message})
            return self.chat_history[chat_id]

        self.chat_history[chat_id].append({"role": "user", "content": new_user_message})
        chat_history = self.chat_history[chat_id]
        return chat_history

    def get_response(self, chat_id: int, text: str, model: str, max_token: int =1024) -> dict:
        """
        Make request to AI and write answer to message_history.
        Usually working in chats with AI.
        """
        user_chat_history = self._get_or_create_user_chat_history(chat_id, text)

        try:
            response = (
                openai.chat.completions.create(
                    model=model,
                    messages=user_chat_history,
                    temperature=self._TEMPERATURE,
                    n=1,
                    max_tokens=max_token, )
            )

            answer = {"message": response.choices[0].message.content, "total_cost": response.usage.total_cost}
            self.chat_history[chat_id].append({"role": "assistant", "content": answer["message"]})

            return answer

        except Exception as e:
            #self.clear_chat_history(chat_id)
            print(e)

    def add_txt_to_user_chat_history(self, chat_id: int, text: str) -> None:
        try:
            self._get_or_create_user_chat_history(chat_id, text)
        except Exception as e: 
            #logger.error(f'Error occurred while adding text: {e} to user chat history')
            print("Error occurred while adding text to user chat history")

    """def get_single_response(self, text: str, model: str, meta_prompt: str = ANALYTIC_PROMPT) -> str:
       
        Get response from AI without message_history.
        Working with parsing payments.
        
        try:
            response = (
                openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "assistant", "content": meta_prompt},
                        {"role": "user", "content": f'{text}'}, ],
                    temperature=self._TEMPERATURE,
                    n=1,
                    max_tokens=self._MAX_TOKENS + 1000,
                ).choices[0].message.content
            )
            return response

        except Exception as e:
            print(e)

    def generate_image(self, prompt: str, model: str) -> str:
        try:
            response = (
                openai.images.generate(
                    model=model,
                    prompt=prompt,
                    n=1,
                    response_format="b64_json",
                    size="1024x1024"
                )
            )
            
            imageb64 = response.data[0].b64_json

            if "data:image" in imageb64:
                imageb64 = imageb64.split(",")[1]
            image_bytes = base64.b64decode(imageb64)
            image_stream = BytesIO(image_bytes)
            image = Image.open(image_stream)
            return image

        except Exception as e:
            print(e)
        """
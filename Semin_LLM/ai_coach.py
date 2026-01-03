import os
import io
import json
import numpy as np
from PIL import Image as PILImage

# Google Vertex AI 관련 임포트
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image


class LLMCoach:
    def __init__(self, config):
        self.config = config
        self.enabled = config.get("use_ai_coach", False)

        if self.enabled:
            #인증키 설정
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.get("key_path", "service_account.json")

            #Gemini 사용 // Vertex AI 서버 접속
            try:
                vertexai.init(project=config["project_id"], location=config["location"])

                #모델 불러옴
                self.model = GenerativeModel(config["model_name"])
                print(f"LLM 코치 초기화 완료! ({config['model_name']})")

            except Exception as e:
                print(f"LLM 초기화 실패: {e}")
                print("   -> key_path 경로와 프로젝트 ID를 확인하세요.")
                self.enabled = False #AI 없어도 실행가능하게 함.

    #게임 실행 도중에 LLM과 대화를 가능하게 함
    def ask_advice(self, screen_array, context_text=""):
        if not self.enabled:
            return None

        try:
            image_part = self._process_image(screen_array)
            prompt = f"Context: {context_text}\nWhere am I? What to do next? Short advice."

            response = self.model.generate_content([image_part, prompt])
            return response.text.strip()

        except Exception as e:
            print(f"⚠️ [Gemini] 조언 요청 실패: {e}")
            return None

    #LLM의 보상 조건을 관리하고 이유를 같이 설명하도록 함.
    def evaluate_screen(self, screen_array, current_reward):
        if not self.enabled:
            return 0.0, "AI Coach Disabled"

        try:
            #게임 화면을 이미지 파일처럼 만듬.
            image_part = self._process_image(screen_array)

            #나의 실제 채점 기준표 LLM이 이를 기반으로 적용하여 보상을 매김. 당연히 마이너스도 됨
            prompt = f"""
            너는 포켓몬 골드 버전 AI를 평가하는 심판이야.
            현재 화면을 보고 게임 진행 상황을 다음 채점 기준에 맞춰 평가해 줘.

            [채점 기준]:
            - 0점: 구석에 갇혀 있거나, 의미 없는 행동 반복, 검은 화면.
            - 2점: 탐험 중 (새로운 장소로 이동).
            - 5점: 상호작용 (NPC 대화, 표지판 읽기, 아이템 줍기).
            - 8점: 전투 중이거나 메뉴 조작 중.
            - 10점: 큰 성과 (전투 승리, 레벨업, 새로운 도시 도착, 중요 이벤트).

            Output must be strict JSON:
            {{
                "score": <0~10 사이의 숫자>,
                "reason": "<점수를 준 이유를 한국어로 짧게 한 문장으로>"
            }}
            """

            #위 img와 prompt를 기반으로 실제 요청.
            response = self.model.generate_content(
                [image_part, prompt],
                generation_config={"response_mime_type": "application/json"}
            )

            #결과 받아와서 해석
            result = json.loads(response.text)

            score = float(result.get("score", 0))
            reason = result.get("reason", "No reason provided")

            return score, reason

        except Exception as e:
            print(f"LLM 평가 실패: {e}")
            # 에러 발생 시 0점과 에러 메시지 반환
            return 0.0, f"Error: {e}"

    def _process_image(self, screen_array):

        #PPO형식에 맞는 방식을 원래 img 방식으로 변환
        if screen_array.shape[0] == 3:
            screen_img = np.transpose(screen_array, (1, 2, 0))
        else:
            screen_img = screen_array

        #숫자 이미지 저장
        pil_img = PILImage.fromarray(screen_img)
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')

        # Vertex AI Image 객체 생성
        return Image.from_bytes(img_byte_arr.getvalue())
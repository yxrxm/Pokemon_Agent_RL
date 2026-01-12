import os
import io
import json
import numpy as np
from PIL import Image as PILImage

# Google Vertex AI 관련 임포트
import vertexai
from vertexai.generative_models import GenerativeModel, Image


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
            prompt = f"Context: {context_text}\n"

            response = self.model.generate_content([image_part, prompt])
            return response.text.strip()

        except Exception as e:
            print(f"LLM 조언 요청 실패: {e}")
            return None

    #LLM의 보상 조건을 관리하고 이유를 같이 설명하도록 함.
    # [수정] game_status 인자를 추가했습니다. (기본값은 빈 딕셔너리)
    def evaluate_screen(self, screen_array, current_reward, game_status={}):
        if not self.enabled:
            return 0.0, "AI Coach Disabled"

        try:
            # 1. 이미지 처리
            image_part = self._process_image(screen_array)

            # 2. [추가] 텍스트 정보(game_status)를 보기 좋게 문자열로 변환
            # 예: "- HP: 20/20 \n - Battle Mode: No ..." 형태가 됩니다.
            status_text = "\n".join([f"- {k}: {v}" for k, v in game_status.items()])

            # 3. [수정] 프롬프트 보강 (이미지 + 텍스트 정보 + 현재 보상 상황)
            prompt = f"""
            너는 포켓몬 골드 버전 AI를 평가하는 심판이야.
            제공된 [게임 화면]과 [내부 데이터]를 종합해서 현재 상황을 판단해 줘.

            [현재 게임 내부 데이터]:
            {status_text}

            [현재 스텝에서 받은 보상 합계]: {current_reward}

            위 정보를 바탕으로 AI의 행동을 평가해 줘.
            특히, 'HP가 없어서 치료를 잘했다'거나 '전투 중인데 도망쳤다' 같은 인과관계를 잘 봐줘.

            [채점 기준]:
            - 0점: 별다른 성과 없음, 의미 없는 행동 반복.
            - 5점: 전투 승리, 유의미한 탐험, 적절한 회복.
            - 10점: 레벨업, 배지 획득, 새로운 마을 도착, 중요 이벤트 클리어.
            - 감점(-1 ~ -5): HP가 꽉 찼는데 또 치료함(낭비), 벽에 막혀 제자리걸음.

            Output must be strict JSON:
            {{
                "score": <-5~10 사이의 숫자 (소수점 가능)>,
                "reason": "<점수를 준 이유를 한국어로 짧게 한 문장으로>"
            }}
            """

            # 4. LLM 요청
            response = self.model.generate_content(
                [image_part, prompt],
                generation_config={"response_mime_type": "application/json"}
            )

            # 5. [추가] 결과 파싱 안전장치 (마크다운 백틱 제거)
            text_response = response.text.strip()

            # 가끔 AI가 ```json { ... } ``` 형태로 줄 때가 있어서 이를 제거함
            if text_response.startswith("```"):
                text_response = text_response.strip("`").replace("json", "").strip()

            # JSON 변환
            result = json.loads(text_response)

            score = float(result.get("score", 0))
            reason = result.get("reason", "No reason provided")

            return score, reason

        except Exception as e:
            print(f"LLM 평가 실패: {e}")
            # 에러 발생 시 0점 반환 (프로그램이 멈추지 않도록)
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
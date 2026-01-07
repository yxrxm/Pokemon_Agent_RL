import google.generativeai as genai

#실제 사용할 수 있는 모델을 확인할 수 있는 파일
api_key = ""  # <-- 여기에 본인 키 입력

genai.configure(api_key=api_key) #여기서는 일단 gemini google 것을 사용

try:
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            available_models.append(m.name)

    if not available_models:
        print("목록이 비어있음. 키가 잘못되었을 수 있음.")

except Exception as e:
    print(f"Error: {e}")
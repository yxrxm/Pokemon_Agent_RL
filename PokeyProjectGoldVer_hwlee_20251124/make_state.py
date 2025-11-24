import os
from pyboy import PyBoy

# 현재 폴더에 있는 골드 버전 롬 파일 이름으로 변경했습니다.
rom_path = "PokeGold.gbc"

if not os.path.exists(rom_path):
    print(f"오류: {rom_path} 파일을 찾을 수 없습니다. 같은 폴더에 롬 파일이 있는지 확인하세요.")
    exit()

print("=== 안내 ===")
print("1. 게임 창이 열리면 엔터(Enter, Start버튼) 키 등을 눌러서 오프닝을 넘기세요.")
print("2. 'New Game' 메뉴가 보이는 타이틀 화면이 나오면 잠시 기다리세요.")
print("3. 그 상태에서 게임 창을 닫으면(X버튼), 자동으로 init.state 파일이 생성됩니다.")
print("============")

# PyBoy 실행 (SDL2 윈도우 모드)
pyboy = PyBoy(rom_path, window="SDL2")
pyboy.set_emulation_speed(1) # 사람이 플레이하기 편한 정상 속도

# 창이 열려있는 동안 계속 실행 (유저가 X를 눌러 끌 때까지 대기)
while pyboy.tick():
    pass

# 창을 닫으면 저장
print("창이 닫혔습니다. 현재 상태를 init.state로 저장합니다...")
with open("init.state", "wb") as f:
    pyboy.save_state(f)
print(f"저장 완료! 생성 위치: {os.path.abspath('init.state')}")
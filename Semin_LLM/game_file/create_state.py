from pyboy import PyBoy
import os
import sys

# --- 설정 ---
ROM_FILE = "./PokeGold.gbc"  # 1. 본인의 골드 버전 롬 파일 경로
STATE_FILE = "init.state"  # 2. 저장할 파일 이름 (train_gold.py와 일치시킴)
EMULATION_SPEED = 1  # 3. 게임 플레이 속도 (1 = 1배속, 6 = 6배속)
# ------------

# 롬 파일이 있는지 확인
if not os.path.exists(ROM_FILE):
    print(f"오류: '{ROM_FILE}' 파일을 찾을 수 없습니다.")
    print("ROM_FILE 변수의 경로를 올바르게 수정해주세요.")
    sys.exit()

try:
    pyboy = PyBoy(ROM_FILE, window="SDL2")
except Exception as e:
    print(f"PyBoy 초기화 중 오류 발생: {e}")
    print("SDL2 라이브러리 관련 문제가 있을 수 있습니다.")
    sys.exit()

# 게임을 플레이할 수 있도록 정상 속도(1배속) 또는 그 이상으로 설정합니다.
pyboy.set_emulation_speed(EMULATION_SPEED)

print("--- PyBoy 에뮬레이터가 실행되었습니다 ---")
print(f"\n1. 게임 창에서 원하는 지점까지 '수동으로' 플레이하세요.")
print(f"   (현재 속도: {EMULATION_SPEED}배속)")
print(f"\n2. 원하는 시작 지점에 멈춘 후, **PyBoy 게임 창(X 버튼)을 닫으세요**.")
print("   (창을 닫으면 그 즉시 현재 상태가 저장됩니다.)")

# 게임 루프: pyboy.tick()은 창이 닫히면 False를 반환합니다.
# 이 루프가 게임을 계속 실행시킵니다.
try:
    while pyboy.tick():
        pass  # tick() 자체가 모든 것을 처리합니다.
except OSError as e:
    # 게임 창을 강제로 닫을 때 가끔 발생하는 OSERROR를 무시합니다.
    print(f"OS 오류 발생 (정상 종료 과정일 수 있음): {e}")

# --- 사용자가 창을 닫으면 이 코드가 실행됩니다 ---
try:
    with open(STATE_FILE, "wb") as f:
        pyboy.save_state(f)

    print(f"\n게임 창이 닫혔습니다. '{STATE_FILE}' 파일이 성공적으로 저장되었습니다.")

except Exception as e:
    print(f"상태 저장 중 오류 발생: {e}")

finally:
    pyboy.stop()
    print("스크립트를 종료합니다.")
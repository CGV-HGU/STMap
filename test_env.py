
import sys
print("Python version:", sys.version)
try:
    import habitat
    print("habitat imported")
except Exception as e:
    print(f"habitat failed: {e}")

try:
    import google.generativeai as genai
    print("genai imported")
except Exception as e:
    print(f"genai failed: {e}")

try:
    import gpt4v_planner
    print("gpt4v_planner imported")
except Exception as e:
    print(f"gpt4v_planner failed: {e}")

pass

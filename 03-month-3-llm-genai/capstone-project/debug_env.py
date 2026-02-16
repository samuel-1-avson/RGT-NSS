import sys
import os

try:
    import langchain_openai
    print("langchain_openai module found:", langchain_openai.__file__)
except ImportError as e:
    print("ImportError:", e)

api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print("OPENAI_API_KEY is SET")
else:
    print("OPENAI_API_KEY is NOT set")

print("python executable:", sys.executable)

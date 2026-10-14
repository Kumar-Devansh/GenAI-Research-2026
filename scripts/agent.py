"""LLM agent (open-weight model via Ollama) with 2 tools: forecast + knowledge search.
Run:  python scripts/agent.py "Will tomorrow be hotter than today in Delhi?" """
import sys, inspect, ollama
from core import predict_next_day, search_knowledge

MODEL = "llama3.2"   # any tool-capable model: `ollama pull llama3.2`
TOOLS = {"predict_next_day": predict_next_day, "search_knowledge": search_knowledge}
SYSTEM = ("You are a weather assistant for Delhi and the Indian Southwest Monsoon. "
          "Use predict_next_day for forecasts and search_knowledge for facts. "
          "When reporting a forecast, state the exact forecast_date and temperature "
          "from the tool, and say it is a research-model estimate based on the local archive, not live. "
          "For knowledge answers cite retrieved title, page and URL; if no evidence is returned, say so. "
          "Never invent numbers. Answer concisely in the user's language.")


def run_agent(question: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    resp = ollama.chat(model=MODEL, messages=msgs, tools=list(TOOLS.values()))
    for _ in range(4):
        if not resp.message.tool_calls:
            break
        msgs.append(resp.message)
        for call in resp.message.tool_calls:
            fn = TOOLS.get(call.function.name)
            if fn is None:
                out = "Unknown tool."
            else:
                allowed = inspect.signature(fn).parameters
                args = {k: v for k, v in call.function.arguments.items() if k in allowed}
                try:
                    out = fn(**args)
                except Exception as e:
                    out = f"Tool error: {e}"
            msgs.append({"role": "tool", "tool_name": call.function.name, "content": str(out)})
        resp = ollama.chat(model=MODEL, messages=msgs, tools=list(TOOLS.values()))
    return resp.message.content


if __name__ == "__main__":
    print(run_agent(" ".join(sys.argv[1:]) or "Forecast tomorrow's temperature in Delhi."))

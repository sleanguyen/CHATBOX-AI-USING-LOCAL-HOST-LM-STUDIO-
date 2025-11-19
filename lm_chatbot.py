import requests
import json


class LMChatBot:
    def __init__(self, api_url="http://localhost:1234/v1"):
        self.api_url = api_url
        self.conversation_history = []

        # Define tools/functions the model can call
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_summary",
                    "description": "Returns a summary of the Spotify dataset, including total songs and average popularity.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_popularity",
                    "description": "Analyzes and visualizes the distribution of song popularity and its correlation with audio features.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "show_correlations",
                    "description": "Generates a heatmap to show the correlation between all major audio features.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_feature",
                    "description": "Analyzes and visualizes a specific audio feature's distribution and its relation to popularity.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feature_name": {
                                "type": "string",
                                "description": "The name of the feature to analyze. Must be one of: danceability, energy, valence, acousticness, instrumentalness, liveness, speechiness, tempo.",
                            }
                        },
                        "required": ["feature_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "predict_song",
                    "description": "Predicts the popularity category of a song based on its audio features.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "danceability": {"type": "number"},
                            "energy": {"type": "number"},
                            "valence": {"type": "number"},
                            "acousticness": {"type": "number"},
                        },
                    },
                },
            }
        ]

    def test_connection(self):
        """Test connection to LM Studio"""
        try:
            response = requests.get(f"{self.api_url}/models", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def send_message(self, user_message):
        """Send message to LM Studio and handle function calling"""
        try:
            # System Prompt
            system_message = """You are a smart Spotify Data Assistant.
            - ONLY use the provided tools if the user EXPLICITLY asks for data, summary, analysis, or prediction.
            - For general questions (hello, jokes, fun facts, school), DO NOT use tools. Just chat normally.
            - If you act as a tool, output the JSON. If not, output text."""


            messages = [{"role": "system", "content": system_message}]


            for role, content in self.conversation_history[-4:]:
                messages.append({"role": role, "content": str(content)})


            messages.append({"role": "user", "content": user_message})

            payload = {
                "model": "local-model",
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.7,
                "tools": self.tools
            }

            response = requests.post(
                f"{self.api_url}/chat/completions",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                choice = result['choices'][0]['message']


                self.conversation_history.append(("user", user_message))



                if choice.get("tool_calls"):
                    tool_call = choice['tool_calls'][0]
                    return {
                        "type": "function_call",
                        "name": tool_call['function']['name'],
                        "args": json.loads(tool_call['function']['arguments']),
                    }


                content = choice.get('content', '')
                if content and '{' in content and '"name":' in content:
                    try:

                        start = content.find('{')
                        end = content.rfind('}') + 1
                        json_str = content[start:end]
                        data = json.loads(json_str)


                        if "name" in data:
                            return {
                                "type": "function_call",
                                "name": data["name"],
                                "args": data.get("parameters", {}) or data.get("args", {})
                            }
                    except:
                        pass



                self.conversation_history.append(("assistant", content))
                return {"type": "text", "content": content}

            else:
                return {"type": "text", "content": f"Error: API returned status {response.status_code}"}

        except Exception as e:
            return {"type": "text", "content": f"Error connecting to AI: {str(e)}"}
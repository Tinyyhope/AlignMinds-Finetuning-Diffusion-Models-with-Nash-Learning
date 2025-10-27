import os
import json
import time
import re
from tqdm import tqdm
from openai import OpenAI

class DeepSeekPromptSlotFillerFeasibility:
    """
    This class performs automated slot-filling classification of image prompts using the DeepSeek API.
    It extracts attributes such as 'Character', 'Scene/Location', 'Action', 'Emotion', 'Feasibility', and 'Style' 
    from free-form textual prompts and saves the structured output to a JSON file.
    """

    def __init__(self, api_key, input_json_path, output_json_path, max_samples=1000, sleep_time=1.0, max_retries=3):
        """
        Initialize the classification system.

        Parameters:
            api_key (str): API key for authenticating with DeepSeek.
            input_json_path (str): Path to input JSON file containing prompts.
            output_json_path (str): Path to write classified output.
            max_samples (int): Maximum number of prompts to process.
            sleep_time (float): Delay (in seconds) between API requests.
            max_retries (int): Maximum number of retry attempts for API failures.
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            default_headers={"Content-Type": "application/json; charset=utf-8"}
        )
        self.input_json_path = input_json_path
        self.output_json_path = output_json_path
        self.max_samples = max_samples
        self.sleep_time = sleep_time
        self.max_retries = max_retries
        self.allowed_styles = {"paintings", "anime and cartoon", "real photo", "concept-art"}

    def _load_data(self):
        """Load the prompt data from JSON and truncate to `max_samples`."""
        with open(self.input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data[:self.max_samples]

    def _safe_ascii(self, text):
        """Ensure that non-ASCII characters are removed to prevent console printing errors."""
        return text.encode('ascii', 'ignore').decode('ascii')

    def _parse_slots_from_response(self, response_text):
        """
        Parse slot values from the raw LLM response.

        Ensures constraints on values for 'Feasibility' (yes/no) and 'Style' (from allowed set).
        Defaults to fallback values if parsing fails.
        """
        slots = {
            "Character": "none",
            "Scene/Location": "none",
            "Action": "none",
            "Emotion": "none",
            "Feasibility": "no",
            "Style": "others"
        }
        for slot in slots.keys():
            pattern = rf"{slot}:\s*(.+)"
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if slot == "Style":
                    value = value.lower()
                    if value not in self.allowed_styles:
                        value = "others"
                if slot == "Feasibility":
                    value = value.lower()
                    if value not in {"yes", "no"}:
                        value = "no"
                slots[slot] = value
        return slots

    def _classify_prompt(self, prompt_text):
        """
        Send a prompt to the DeepSeek API and parse the returned slot-filling output.

        Retries on failure, up to `max_retries` times.
        """
        system_prompt = (
            "You will be given a description of an image.\n"
            "Please fill the following slots based only on the description:\n\n"
            "- Character: [describe the main character, or 'none']\n"
            "- Scene/Location: [describe the scene or location, or 'none']\n"
            "- Action: [describe the main action, or 'none']\n"
            "- Emotion: [describe the main emotion, or 'none']\n"
            "- Feasibility: [Answer 'yes' if the described scene is plausible in the real world, otherwise 'no']\n"
            "- Style: [choose one of: paintings, anime and cartoon, real photo, concept-art, others]\n\n"
            "Strictly output in the following format:\n"
            "Character: <answer>\n"
            "Scene/Location: <answer>\n"
            "Action: <answer>\n"
            "Emotion: <answer>\n"
            "Feasibility: <yes or no>\n"
            "Style: <answer>\n\n"
            "Do not add any extra explanation or comment."
        )

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": prompt_text.strip()}
        ]

        for attempt in range(1, self.max_retries + 1):
            try:
                safe_prompt_to_print = self._safe_ascii(prompt_text[:50])
                print(f"[Attempt {attempt}] Classifying prompt: {safe_prompt_to_print}...")
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0.0,
                    max_tokens=150,
                    timeout=10
                )
                response_text = response.choices[0].message.content.strip()
                print(f"[Raw Response] {self._safe_ascii(response_text)}")

                slots = self._parse_slots_from_response(response_text)
                print(f"[Success] Extracted feasibility: {slots['Feasibility']}, style: {slots['Style']}")

                return slots

            except Exception as e:
                print(f"[Error] Error on attempt {attempt}: {e}")
                time.sleep(self.sleep_time * attempt)

        # Return fallback slot values if all attempts fail
        print("[Warning] Failed to classify after multiple attempts.")
        return {
            "Character": "none",
            "Scene/Location": "none",
            "Action": "none",
            "Emotion": "none",
            "Feasibility": "no",
            "Style": "others"
        }

    def run(self):
        """
        Main execution function.

        Loads prompt data, classifies each prompt using the DeepSeek API, 
        and saves the extracted slot information to an output JSON file.
        """
        data = self._load_data()
        classified_data = []

        print(f"[Info] Loaded {len(data)} prompts to classify.")

        for idx, entry in enumerate(tqdm(data, desc="Classifying Prompts")):
            prompt_text = entry.get('prompt', '')
            if not prompt_text:
                continue

            slots = self._classify_prompt(prompt_text)

            # Save original prompt and extracted style/feasibility attributes
            classified_entry = {
                "prompt": prompt_text,
                "style": slots["Style"],
                "feasibility": slots["Feasibility"]
            }
            classified_data.append(classified_entry)
            time.sleep(self.sleep_time)

        with open(self.output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(classified_data, outfile, ensure_ascii=False, indent=2)

        print(f"[Info] Finished! Classified prompts saved to {self.output_json_path}")


if __name__ == "__main__":
    classifier = DeepSeekPromptSlotFillerFeasibility(
        api_key="Deepseek api path",  
        input_json_path="input_dat_path",
        output_json_path="output_path",
        max_samples=500,
        sleep_time=0.05,
        max_retries=3
    )
    classifier.run()


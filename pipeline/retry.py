from utils import make_file_contents_str

def retry_installation(file_contents: dict[str, str], commit_hash: str, commit_date: str, error_message: str):
    # load prompt from prompts/trial_and_error.txt
    with open("prompts/trial_and_error.txt", "r") as f:
        prompt = f.read()
    prompt = prompt.replace("{file_contents_str}", make_file_contents_str(file_contents))
    prompt = prompt.replace("{commit_hash}", commit_hash)
    prompt = prompt.replace("{commit_date}", commit_date)
    prompt = prompt.replace("{error_message}", error_message)
    print(prompt)
    return prompt

if __name__ == "__main__":
    pass
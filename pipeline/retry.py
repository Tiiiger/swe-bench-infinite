def retry_installation(tree_result, commit_hash, commit_date, error_message):
    # load prompt from prompts/trial_and_error.txt
    with open("prompts/trial_and_error.txt", "r") as f:
        prompt = f.read()
    prompt = prompt.replace("{tree_result}", tree_result)
    prompt = prompt.replace("{commit_hash}", commit_hash)
    prompt = prompt.replace("{commit_date}", commit_date)
    prompt = prompt.replace("{error_message}", error_message)
    print(prompt)
    return prompt

if __name__ == "__main__":
    pass
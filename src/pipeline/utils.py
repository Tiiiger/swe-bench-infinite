def make_file_contents_str(file_contents: dict[str, str]) -> str:
    """
    Make a string from a dictionary of file contents.
    """
    # format each file content with the file path
    file_contents_str = ""
    for file_path, content in file_contents.items():
        file_contents_str += f"==== {file_path} ====\n{content}\n\n"
    return file_contents_str

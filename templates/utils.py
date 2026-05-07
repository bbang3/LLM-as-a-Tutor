import os

TEMPLATE_FILES_DIR = os.path.join(os.path.dirname(__file__), "template_files")


def load_template(name: str, extension: str = "txt"):
    if "." not in name or name.split(".")[-1] not in ["txt", "jinja"]:
        # name does not include extension, add it
        basename = f"{name}.{extension}"
    else:
        basename = name
    template_path = os.path.join(TEMPLATE_FILES_DIR, basename)
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read().strip()

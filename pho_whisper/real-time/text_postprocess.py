def postprocess_text(text):
    """
    Xóa khoảng trắng
    Thêm dấu chấm vào cuối câu nếu không có
    """
    text = " ".join(text.split())
    if not text.endswith("."):
        text += "."
    return text

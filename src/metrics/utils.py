import editdistance


def calc_cer(target_text: str, predicted_text: str) -> float:
    return 1 if not target_text else editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text: str, predicted_text: str) -> float:
    return 1 if not target_text.split() else editdistance.eval(target_text.split(), predicted_text.split()) / len(target_text.split())

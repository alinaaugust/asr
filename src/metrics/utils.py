import editdistance


def calc_cer(target_text, predicted_text) -> float:
    target_text = target_text.lower()
    predicted_text = predicted_text.lower()

    if len(target_text) == 0:
        return 1

    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target_text = target_text.lower()
    predicted_text = predicted_text.lower()
    target_text = target_text.split()
    predicted_text = predicted_text.split()

    if len(target_text) == 0:
        return 1

    return editdistance.eval(target_text, predicted_text) / len(target_text)

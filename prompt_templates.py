PROMPTS_WITHOUT_TARGET_QUESTION_MASKING = ['NoP', 'WP-QM', "WP-TQM", "WP-v2.2-TP-QM", "WP-v2.2-TP-TQM", "FP-QM", "FP-TQM"]


def prompting(option, context_text, history_questions, history_answers, question_text, answer_text, question_mask="<mask>", question_mask_dec="<mask>"):
    """
    No Prompt: <a> A1 <q> Q1 ... <a> At-1 <q> Qt-1 <a> At <sep> C
    Prompt: conversation: answer: A1 question: Q1 ... answer: At-1 question: Qt-1 answer: At question: <mask> context: C
    """

    # tokenizer.add_tokens(['<sep>', '<mask>', '<h>', '<\h>', '<q>', '<\q>', '<a>', '<\\a>'])

    if option == "NoP":
        if not history_questions:
            source_text = f"{answer_text} <sep> {context_text}"
        else:
            history_text = ""
            for h_q, h_a in zip(history_questions, history_answers):
                history_text += f"<a> {h_a} <q> {h_q} "

            source_text = f"{history_text} <a> {answer_text} <sep> {context_text}"

        target_text = f"{question_text}"

    elif option == "FP+DiffQM":
        # with different enc * dec prompt
        history_text = ""
        for h_q, h_a in zip(history_questions, history_answers):
            history_text += f"answer: {h_a} question: {h_q} "

        source_text = f"conversation: {history_text} answer: {answer_text} question: {question_mask} context: {context_text}"
        target_text = f"{question_mask_dec} {question_text}"

    elif option == "FP-v2":
        history_text = ""
        for h_q, h_a in zip(history_questions, history_answers):
            history_text += f"answer: {h_a} question: {h_q} "

        source_text = f"conversation: {history_text} answer: {answer_text} question: <mask> context: {context_text}"
        target_text = f"<mask> {question_mask} {question_text}"

    elif option == "FP":
        history_text = ""
        for h_q, h_a in zip(history_questions, history_answers):
            history_text += f"answer: {h_a} question: {h_q} "

        source_text = f"conversation: {history_text} answer: {answer_text} question: {question_mask} context: {context_text}"
        target_text = f"{question_mask} {question_text}"

    elif option == "FP-SP":
        history_text = ""
        for h_q, h_a in zip(history_questions, history_answers):
            history_text += f"<a> {h_a} <q> {h_q} "

        source_text = f"{history_text} <a> {answer_text} <q> {question_mask} <sep> {context_text}"
        target_text = f"{question_mask} {question_text}"

    elif option == "FP-QM":
        history_text = ""
        for h_q, h_a in zip(history_questions, history_answers):
            history_text += f"answer: {h_a} question: {h_q} "

        source_text = f"conversation: {history_text} answer: {answer_text} context: {context_text}"
        target_text = f"{question_text}"

    elif option == "FP-SQM":
        history_text = ""
        for h_q, h_a in zip(history_questions, history_answers):
            history_text += f"answer: {h_a} question: {h_q} "

        source_text = f"conversation: {history_text} answer: {answer_text} context: {context_text}"
        target_text = f"{question_mask} {question_text}"

    elif option == "FP-TQM":
        history_text = ""
        for h_q, h_a in zip(history_questions, history_answers):
            history_text += f"answer: {h_a} question: {h_q} "

        source_text = f"conversation: {history_text} answer: {answer_text} question: {question_mask} context: {context_text}"
        target_text = f"{question_text}"

    else:
        raise NotImplementedError

    return source_text, target_text
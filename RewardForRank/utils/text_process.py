def get_context(msc, msg_index, window_size): 
    context_sentence_list = []
    start_index = msg_index - window_size
    if start_index < 0:
        start_index = 0
    for index in range(start_index, msg_index):
        from_user = msc[index][0]
        msg_text = msc[index][1]
        if from_user:
            character_name = "[USER]" 
        else:
            character_name = "[ADVI]"
        if msg_text is None:
            msg_text = ""
        context_sentence_list.append(character_name + " " + msg_text)
    return " ".join(context_sentence_list)


def get_full_dialog_text(msc):
    context_sentence_list = []
    for i in range(len(msc)):
        from_user = msc[i][0]
        msg_text = msc[i][1]
        created_at = msc[i][2]
        if from_user:
            character_name = "[USER]" 
        else:
            character_name = "[ADVI]"
        if msg_text is None:
            msg_text = ""
        context_sentence_list.append(character_name + " " + msg_text)
    return " ".join(context_sentence_list)
        

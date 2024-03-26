def get_node_embeds(node_list, get_model_embed):
    embeds = []
    for node in node_list:
        this_embed = list(get_model_embed(node))
        embeds.append(this_embed)

    return embeds
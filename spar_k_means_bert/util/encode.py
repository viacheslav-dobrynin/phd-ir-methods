import torch


# Encode text
def encode_to_token_embs(model, input_ids, attention_mask):
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

    return model_output.last_hidden_state

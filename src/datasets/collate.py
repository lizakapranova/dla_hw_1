import torch


def collate_fn(items: list[dict]) -> dict[str, torch.Tensor]:
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        batch (dict[str, torch.Tensor]): dict, containing batch-version
            of the tensors.
    """

    # TODO: refactor more
    sorted_items = sorted(items, key=lambda item: item['spectrogram'].shape[2], reverse=True)
    batch_size = len(sorted_items)

    max_spectrogram_length = sorted_items[0]['spectrogram'].shape[2]
    max_text_encoded_length = max(sorted_items, key=lambda item: item['text_encoded'].shape[1])['text_encoded'].shape[1]
    n_freqs = sorted_items[0]['spectrogram'].shape[1]

    batch = {
        'spectrogram': torch.zeros((batch_size, n_freqs, max_spectrogram_length)),
        'spectrogram_length': torch.zeros(batch_size, dtype=torch.int32),
        'text_encoded': torch.zeros((batch_size, max_text_encoded_length)),
        'text_encoded_length': torch.zeros(batch_size, dtype=torch.int32),
        'text': [''] * batch_size,
        'audio': [0] * batch_size,
        'audio_path': [''] * batch_size
    }

    for i in range(batch_size):
        spectrogram = sorted_items[i]['spectrogram'][0]  # F x T
        text_encoded = sorted_items[i]['text_encoded']  # L
        batch['spectrogram'][i, :, :spectrogram.shape[1]] = spectrogram
        batch['spectrogram'][i, :, spectrogram.shape[1]:] = torch.zeros(
            (n_freqs, max_spectrogram_length - spectrogram.shape[1]))
        batch['spectrogram_length'][i] = spectrogram.shape[1]
        batch['text_encoded'][i, :text_encoded.shape[1]] = text_encoded[0]
        batch['text_encoded'][i, text_encoded.shape[1]:] = torch.zeros(max_text_encoded_length - text_encoded.shape[1])
        batch['text_encoded_length'][i] = text_encoded.shape[1]
        batch['text'][i] = sorted_items[i]['text']
        batch['audio'][i] = sorted_items[i]['audio']
        batch['audio_path'][i] = sorted_items[i]['audio_path']

    return batch

import os
import string
import torchaudio
import torch
from typing import Tuple, List

import torchtext

torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class DatasetLibriSpeech:
    """
    Loads all the data into the memory. It is not recommended for large datasets.
    """

    def __init__(
        self,
        dir_path: str,
        transcript_paths: list = None,
        max_len_recording: int = 1024,
        max_len_text: int = 103,
        verbose: bool = True,
        vocab: torchtext.vocab.Vocab = None,
    ):
        self.dir_path = dir_path
        self.max_len_recording = max_len_recording
        self.max_len_text = max_len_text

        if transcript_paths is None:
            transcript_paths = [
                os.path.join(d, x)
                for d, dirs, files in os.walk(self.dir_path)
                for x in files
                if "trans.txt" in x
            ]

        self.vocab = vocab
        # load all transcripts under the length of self.max_len_text
        self.processed_transcripts = self.prepare_transcripts(transcript_paths)
        # tokenize the text
        self.tokenized_transcripts = self.tokenize_transcripts(
            self.processed_transcripts
        )

        # prepare the dataset - load it into the memory
        self.X, self.y = self.prepare_xy(self.tokenized_transcripts)

        if verbose:
            print("Dataset prepared")
            print(f"X shape: {self.X.shape}")
            print(f"y shape: {self.y.shape}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]

    def prepare_transcripts(self, transcript_paths) -> None:

        all_transcripts = [open(path, "r").read() for path in transcript_paths]
        all_transcripts = [
            line
            for transcript in all_transcripts
            for line in transcript.split("\n")
            if len(line) > 0
        ]

        processed_transcripts = []
        for line in all_transcripts:
            id, text = line.split(" ", 1)
            cleaned_text = self.clean_text(text)
            if self.max_len_text is not None:
                if len(cleaned_text) <= self.max_len_text:
                    processed_transcripts.append([id, cleaned_text])
            else:
                processed_transcripts.append([id, cleaned_text])

        return processed_transcripts

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def yield_tokens(self, lines):
        for line in lines:
            yield self.torch_tokenizer(line)

    def text_to_tokens(self, vocab, tokenizer, text):
        return [vocab[token] for token in tokenizer(text)]

    def tokenize_transcripts(
        self, processed_transcripts
    ) -> List[Tuple[str, List[int]]]:

        self.torch_tokenizer = get_tokenizer("basic_english")
        if self.vocab is None:
            self.vocab = build_vocab_from_iterator(
                self.yield_tokens([x[1] for x in processed_transcripts]),
                specials=["<unk>"],
            )
            self.vocab.set_default_index(self.vocab["<unk>"])
            self.vocab_size = len(self.vocab)

        torch_tokenized_lines = [
            [id, self.text_to_tokens(self.vocab, self.torch_tokenizer, line)]
            for id, line in processed_transcripts
        ]

        return torch_tokenized_lines

    def tokens_to_text(self, tokens):
        return " ".join(self.vocab.get_itos()[token] for token in tokens)

    def extract_filterbanks(
        self,
        audio_path: str,
        channels: int = 80,
        window_s: int = 0.024,
        stride_s: int = 0.01,
    ) -> torch.tensor:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Define the filterbank transformation
        fbank_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=channels,
            win_length=int(window_s * sample_rate),
            hop_length=int(stride_s * sample_rate),
        )

        # Extract filterbank features
        filterbanks = fbank_transform(waveform)

        return filterbanks.squeeze(0)

    def prepare_xy(self, tokenized_transcripts) -> Tuple[torch.tensor, torch.tensor]:
        X = []
        y = []
        for id, text in tokenized_transcripts:
            # find recording path
            file_name = id + ".flac"
            recording_path = os.path.join(
                self.dir_path, id.replace("-", "/")[:-4], file_name
            )

            # transform it into filterbanks
            recording_filterbank = self.extract_filterbanks(recording_path)

            # check if the recording is not too long
            if recording_filterbank.shape[1] <= self.max_len_recording:

                # pad the recording and the text
                pad_length_recording = (
                    self.max_len_recording - recording_filterbank.shape[1]
                )
                recording_filterbank = torch.nn.functional.pad(
                    recording_filterbank, (0, pad_length_recording)
                )
                pad_length_text = self.max_len_text - len(text)
                text = torch.nn.functional.pad(torch.tensor(text), (0, pad_length_text))

                X.append(recording_filterbank.permute(1, 0))
                y.append(text)

        # change the list of tensors into a tensor
        X = torch.stack(X)
        y = torch.stack(y)

        return X, y


class VocabPreparatorLibriSpeech(DatasetLibriSpeech):
    def __init__(self):
        self.max_len_text = None
        self.vocab = None
        pass

    def prepare_vocab(self, dir_path: str) -> torchtext.vocab.Vocab:
        transcript_paths = [
            os.path.join(d, x)
            for d, dirs, files in os.walk(dir_path)
            for x in files
            if "trans.txt" in x
        ]

        processed_transcripts = self.prepare_transcripts(transcript_paths)
        tokenized_transcripts = self.tokenize_transcripts(processed_transcripts)

        return self.vocab

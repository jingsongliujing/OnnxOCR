import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tokenizers import AddedToken
from tokenizers import Tokenizer as TokenizerFast

from onnxocr.rapid_doc.model.formula.rapid_formula_self.model_handler.pp_formulanet_plus.utils import fix_latex_left_right, \
    fix_latex_environments, remove_up_commands, remove_unsupported_commands

class PPPostProcess:
    def __init__(self, character_dict):
        self.uni_mer_net_decode = UniMERNetDecode(character_list=character_dict)

    def __call__(self, preds):
        """Apply post-processing to the detection boxes.

        Args:
            preds (np.ndarray): Model predictions.

        Returns:
            Boxes: The post-processed detection boxes.
        """
        return self.uni_mer_net_decode(preds)


class UniMERNetDecode(object):
    """Class for decoding tokenized inputs using UniMERNet tokenizer.

    Attributes:
        SPECIAL_TOKENS_ATTRIBUTES (List[str]): List of special token attributes.
        model_input_names (List[str]): List of model input names.
        max_seq_len (int): Maximum sequence length.
        pad_token_id (int): ID for the padding token.
        bos_token_id (int): ID for the beginning-of-sequence token.
        eos_token_id (int): ID for the end-of-sequence token.
        padding_side (str): Padding side, either 'left' or 'right'.
        pad_token (str): Padding token.
        pad_token_type_id (int): Type ID for the padding token.
        pad_to_multiple_of (Optional[int]): If set, pad to a multiple of this value.
        tokenizer (TokenizerFast): Fast tokenizer instance.

    Args:
        character_list (Dict[str, Any]): Dictionary containing tokenizer configuration.
        **kwargs: Additional keyword arguments.
    """

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]

    def __init__(
        self,
        character_list: Dict[str, Any],
        **kwargs,
    ) -> None:
        """Initializes the UniMERNetDecode class.

        Args:
            character_list (Dict[str, Any]): Dictionary containing tokenizer configuration.
            **kwargs: Additional keyword arguments.
        """

        self._unk_token = "<unk>"
        self._bos_token = "<s>"
        self._eos_token = "</s>"
        self._pad_token = "<pad>"
        self._sep_token = None
        self._cls_token = None
        self._mask_token = None
        self._additional_special_tokens = []
        self.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
        self.max_seq_len = 2048
        self.pad_token_id = 1
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.padding_side = "right"
        self.pad_token_id = 1
        self.pad_token = "<pad>"
        self.pad_token_type_id = 0
        self.pad_to_multiple_of = None

        fast_tokenizer_str = json.dumps(character_list["fast_tokenizer_file"])
        fast_tokenizer_buffer = fast_tokenizer_str.encode("utf-8")
        self.tokenizer = TokenizerFast.from_buffer(fast_tokenizer_buffer)
        tokenizer_config = (
            character_list["tokenizer_config_file"]
            if "tokenizer_config_file" in character_list
            else None
        )
        added_tokens_decoder = {}
        added_tokens_map = {}
        if tokenizer_config is not None:
            init_kwargs = tokenizer_config
            if "added_tokens_decoder" in init_kwargs:
                for idx, token in init_kwargs["added_tokens_decoder"].items():
                    if isinstance(token, dict):
                        token = AddedToken(**token)
                    if isinstance(token, AddedToken):
                        added_tokens_decoder[int(idx)] = token
                        added_tokens_map[str(token)] = token
                    else:
                        raise ValueError(
                            f"Found a {token.__class__} in the saved `added_tokens_decoder`, should be a dictionary or an AddedToken instance"
                        )
            init_kwargs["added_tokens_decoder"] = added_tokens_decoder
            added_tokens_decoder = init_kwargs.pop("added_tokens_decoder", {})
            tokens_to_add = [
                token
                for index, token in sorted(
                    added_tokens_decoder.items(), key=lambda x: x[0]
                )
                if token not in added_tokens_decoder
            ]
            added_tokens_encoder = self.added_tokens_encoder(added_tokens_decoder)
            encoder = list(added_tokens_encoder.keys()) + [
                str(token) for token in tokens_to_add
            ]
            tokens_to_add += [
                token
                for token in self.all_special_tokens_extended
                if token not in encoder and token not in tokens_to_add
            ]
            if len(tokens_to_add) > 0:
                is_last_special = None
                tokens = []
                special_tokens = self.all_special_tokens
                for token in tokens_to_add:
                    is_special = (
                        (token.special or str(token) in special_tokens)
                        if isinstance(token, AddedToken)
                        else str(token) in special_tokens
                    )
                    if is_last_special is None or is_last_special == is_special:
                        tokens.append(token)
                    else:
                        self._add_tokens(tokens, special_tokens=is_last_special)
                        tokens = [token]
                    is_last_special = is_special
                if tokens:
                    self._add_tokens(tokens, special_tokens=is_last_special)

    def _add_tokens(
        self, new_tokens: "List[Union[AddedToken, str]]", special_tokens: bool = False
    ) -> "List[Union[AddedToken, str]]":
        """Adds new tokens to the tokenizer.

        Args:
            new_tokens (List[Union[AddedToken, str]]): Tokens to be added.
            special_tokens (bool): Indicates whether the tokens are special tokens.

        Returns:
            List[Union[AddedToken, str]]: added tokens.
        """
        if special_tokens:
            return self.tokenizer.add_special_tokens(new_tokens)

        return self.tokenizer.add_tokens(new_tokens)

    def added_tokens_encoder(
        self, added_tokens_decoder: "Dict[int, AddedToken]"
    ) -> Dict[str, int]:
        """Creates an encoder dictionary from added tokens.

        Args:
            added_tokens_decoder (Dict[int, AddedToken]): Dictionary mapping token IDs to tokens.

        Returns:
            Dict[str, int]: Dictionary mapping token strings to IDs.
        """
        return {
            k.content: v
            for v, k in sorted(added_tokens_decoder.items(), key=lambda item: item[0])
        }

    @property
    def all_special_tokens(self) -> List[str]:
        """Retrieves all special tokens.

        Returns:
            List[str]: List of all special tokens as strings.
        """
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_tokens_extended(self) -> "List[Union[str, AddedToken]]":
        """Retrieves all special tokens, including extended ones.

        Returns:
            List[Union[str, AddedToken]]: List of all special tokens.
        """
        all_tokens = []
        seen = set()
        for value in self.special_tokens_map_extended.values():
            if isinstance(value, (list, tuple)):
                tokens_to_add = [token for token in value if str(token) not in seen]
            else:
                tokens_to_add = [value] if str(value) not in seen else []
            seen.update(map(str, tokens_to_add))
            all_tokens.extend(tokens_to_add)
        return all_tokens

    @property
    def special_tokens_map_extended(self) -> Dict[str, Union[str, List[str]]]:
        """Retrieves the extended map of special tokens.

        Returns:
            Dict[str, Union[str, List[str]]]: Dictionary mapping special token attributes to their values.
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    def convert_ids_to_tokens(
        self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """Converts token IDs to token strings.

        Args:
            ids (Union[int, List[int]]): Token ID(s) to convert.
            skip_special_tokens (bool): Whether to skip special tokens during conversion.

        Returns:
            Union[str, List[str]]: Converted token string(s).
        """
        if isinstance(ids, int):
            return self.tokenizer.id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self.tokenizer.id_to_token(index))
        return tokens

    def detokenize(self, tokens: List[List[int]]) -> List[List[str]]:
        """Detokenizes a list of token IDs back into strings.

        Args:
            tokens (List[List[int]]): List of token ID lists.

        Returns:
            List[List[str]]: List of detokenized strings.
        """
        self.tokenizer.bos_token = "<s>"
        self.tokenizer.eos_token = "</s>"
        self.tokenizer.pad_token = "<pad>"
        toks = [self.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ""
                toks[b][i] = toks[b][i].replace("Ġ", " ").strip()
                if toks[b][i] in (
                    [
                        self.tokenizer.bos_token,
                        self.tokenizer.eos_token,
                        self.tokenizer.pad_token,
                    ]
                ):
                    del toks[b][i]
        return toks

    def token2str(self, token_ids: List[List[int]]) -> List[str]:
        """Converts a list of token IDs to strings.

        Args:
            token_ids (List[List[int]]): List of token ID lists.

        Returns:
            List[str]: List of converted strings.
        """
        generated_text = []
        for tok_id in token_ids:
            end_idx = np.argwhere(tok_id == 2)
            if len(end_idx) > 0:
                end_idx = int(end_idx[0][0])
                tok_id = tok_id[: end_idx + 1]
            generated_text.append(
                self.tokenizer.decode(tok_id, skip_special_tokens=True)
            )
        generated_text = [self.post_process(text) for text in generated_text]
        return generated_text

    def normalize(self, s: str) -> str:
        """Normalizes a string by removing unnecessary spaces.

        Args:
            s (str): String to normalize.

        Returns:
            str: Normalized string.
        """
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = "[\W_^\d]"
        names = []
        for x in re.findall(text_reg, s):
            pattern = r"\\[a-zA-Z]+"
            pattern = r"(\\[a-zA-Z]+)\s(?=\w)|\\[a-zA-Z]+\s(?=})"
            matches = re.findall(pattern, x[0])
            for m in matches:
                if (
                    m
                    not in [
                        "\\operatorname",
                        "\\mathrm",
                        "\\text",
                        "\\mathbf",
                    ]
                    and m.strip() != ""
                ):
                    s = s.replace(m, m + "XXXXXXX")
                    s = s.replace(" ", "")
                    names.append(s)
        if len(names) > 0:
            s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s.replace("XXXXXXX", " ")

    def remove_chinese_text_wrapping(self, formula):
        pattern = re.compile(r"\\text\s*{\s*([^}]*?[\u4e00-\u9fff]+[^}]*?)\s*}")

        def replacer(match):
            return match.group(1)

        replaced_formula = pattern.sub(replacer, formula)
        return replaced_formula.replace('"', "")

    def post_process(self, text: str) -> str:
        """Post-processes a string by fixing text and normalizing it.

        Args:
            text (str): String to post-process.

        Returns:
            str: Post-processed string.
        """
        from ftfy import fix_text

        text = self.remove_chinese_text_wrapping(text)
        text = self.fix_latex(text)
        text = fix_text(text)
        # text = self.normalize(text)
        return text

    def fix_latex(self, text: str) -> str:
        """Fixes LaTeX formatting in a string.

        Args:
            text (str): String to fix.

        Returns:
            str: Fixed string.
        """
        text = fix_latex_left_right(text, fix_delimiter=False)
        text = fix_latex_environments(text)
        text = remove_up_commands(text)
        text = remove_unsupported_commands(text)
        # text = self.normalize(text)
        return text

    def __call__(
        self,
        preds: np.ndarray,
        label: Optional[np.ndarray] = None,
        mode: str = "eval",
        *args,
        **kwargs,
    ) -> Union[List[str], tuple]:
        """Processes predictions and optionally labels, returning the decoded text.

        Args:
            preds (np.ndarray): Model predictions.
            label (Optional[np.ndarray]): True labels, if available.
            mode (str): Mode of operation, either 'train' or 'eval'.

        Returns:
            Union[List[str], tuple]: Decoded text, optionally with labels.
        """
        if mode == "train":
            preds_idx = np.array(preds.argmax(axis=2))
            text = self.token2str(preds_idx)
        else:
            text = self.token2str(np.array(preds))
        if label is None:
            return text
        label = self.token2str(np.array(label))
        return text, label
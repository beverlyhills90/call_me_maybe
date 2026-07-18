from typing import Any

from llm_sdk import Small_LLM_Model
from src.arguments_generators_pack.utils import get_vocab_list


class TikTokenizer:
    @staticmethod
    def encode(small_llm: "Small_LLM_Model", text: str) -> list[int]:

        byte_arr = text.encode("utf-8")
        unicode_chars = []
        for byte in byte_arr:
            if byte == 32:
                unicode_chars.append("Ġ")  # Space
            elif byte == 10:
                unicode_chars.append("Ċ")  # \n
            elif byte == 13:
                unicode_chars.append("ĉ")  # \r
            else:
                unicode_chars.append(chr(byte))
        try:
            bpe_ranks = {}
            with open(
                small_llm.get_path_to_merges_file(), encoding="utf-8"
            ) as file:
                for idx, line in enumerate(file):
                    if line.startswith("#") or not line.strip():
                        continue

                    parts = line.strip().split()
                    if len(parts) == 2:
                        bpe_ranks[tuple(parts)] = idx
        except OSError as e:
            print(e)

        def separate_to_pairs(
            unicode_chars: list[str],
        ) -> list[tuple[Any, Any]]:  # sepatere to pairs for BPE
            pairs_arr = []
            i = 0
            while i < (len(unicode_chars) - 1):
                pair = (unicode_chars[i], unicode_chars[i + 1])
                pairs_arr.append(pair)
                i += 1
            return pairs_arr

        def get_higher_priority(
            pairs_arr: list[tuple[Any, Any]],
        ) -> tuple[Any, Any] | None:
            best_pair = None
            best_rank = float("inf")
            for pair in pairs_arr:
                rank = bpe_ranks.get(pair, None)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            return best_pair

        new_word_arr = unicode_chars
        while True:
            pairs = separate_to_pairs(new_word_arr)
            higher = get_higher_priority(pairs)
            if higher is None:
                break
            for i in range(len(new_word_arr)):
                if (
                    new_word_arr[i] == higher[0]
                    and new_word_arr[i + 1] == higher[1]
                ):
                    new_word_arr[i] = higher[0] + higher[1]
                    new_word_arr.pop(i + 1)
                    break

        tokinized = []
        try:
            vocab = get_vocab_list(small_llm=small_llm)

            for token in new_word_arr:
                token_id = vocab.get(token)
                if token_id is not None:
                    tokinized.append(int(token_id))

        except OSError:
            raise Exception("NO ACCES TO VOCAB lol")
        return tokinized

    @staticmethod
    def decode(small_llm: "Small_LLM_Model", tokenids_list: list[int]) -> str:
        try:
            vocab = get_vocab_list(small_llm=small_llm)
            inverted_vocab = {v: k for k, v in vocab.items()}
        except OSError:
            raise Exception("NO ACCES TO VOCAB lol")

        chars = []
        for token in tokenids_list:
            char = inverted_vocab.get(token)
            if char is not None:
                chars.append(char)

        text = []
        for c in "".join(chars):
            if c == "Ġ":
                text.append(" ")  # Space
            elif c == "Ċ":
                text.append("\n")  # \n
            elif c == "ĉ":
                text.append("\r")  # \r
            else:
                text.append(c)
        ret = "".join(text)
        return ret


if __name__ == "__main__":
    small_llm = Small_LLM_Model()
    tiktik = TikTokenizer()
    text = " hi hi"
    res_enc = tiktik.encode(small_llm, text)

    print("My method", res_enc)
    print("Original", small_llm.encode(text).tolist()[0])

    print(tiktik.decode(small_llm, res_enc))

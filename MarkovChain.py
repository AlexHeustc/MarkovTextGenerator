import torch
import torch.nn as nn
import random
import json
from bisect import bisect
from collections import defaultdict
from typing import List, Dict, Union, Tuple


class MarkovChain:
    """
    A class for creating and using a Markov Chain model.
    """
    BEGIN = "___BEGIN__"
    END = "___END__"
    def __init__(self, order: int, data: List[List[str]]):
        self.order = order
        self.data = data
        self.model = defaultdict(lambda: defaultdict(int))
        self._build_model()

    def _build_model(self):
        """Builds the Markov Chain model from the given data."""
        # 统计不同状态转移的频次，同时使用拉普拉斯平滑，这里平滑系数设为1
        alpha = 1
        all_states = set()
        for sentence in self.data:  # 假设句子之间用空格分隔
            words = ([BEGIN] * self.order) + sentence + [END]
            for i in range(len(words) - self.order):
                state = tuple(words[i:i + self.order])
                next_state = words[i + self.order]
                self.model[state][next_state] += 1
                all_states.add(next_state)

        # 重新计算概率，加入拉普拉斯平滑
        for state in self.model:
            total = sum(self.model[state].values()) + len(all_states) * alpha
            for next_state in all_states:
                self.model[state][next_state] = (self.model[state].get(next_state, 0) + alpha) / total

    def generate(self, length: int, start: Union[str, None] = None) -> List[str]:
        """Generates a sequence of states of the given length."""
        if start is None:
            start = random.choice(list(self.model.keys()))
        else:
            start = tuple(start[-self.order:])

        result = list(start)
        for _ in range(length - self.order):
            next_state = self._sample_next_state(start)
            result.append(next_state)
            start = tuple(result[-self.order:])

        return result

    def _sample_next_state(self, state: Tuple[str]) -> str:
        """Samples the next state based on the probabilities in the model."""
        probabilities = list(self.model[state].values())
        states = list(self.model[state].keys())
        return random.choices(states, probabilities)[0]

    def save_model(self, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.model, f, ensure_ascii=False, indent=4)

    @classmethod
    def load_model(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            cls.model = json.load(f)

# # Example usage
# if __name__ == "__main__":
#     # 示例数据，这里简单模拟了一些单词组成的序列，实际应用中可以从文件读取文本并进行分词等处理来获取更丰富的数据
#     data = [["the", "cat", "runs", "quickly", "the", "dog", "walks", "slowly", "the", "bird", "flies", "high"]]
#     # 创建马尔可夫链模型实例，设置阶数为2
#     markov_chain = MarkovChain(2, data)

#     # 使用模型生成一个长度为5的新序列，不指定起始状态（将随机选择起始状态）
#     generated_sequence = markov_chain.generate(5)
#     print(generated_sequence)

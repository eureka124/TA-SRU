"""项目使用的四种循环单元。

三个 SRU 变体依据旧工程修改版 sb3-contrib 中的实现重构。它们和原生
``torch.nn.LSTM`` 统一使用 time-first 输入 ``[时间, 批量, 特征]``，并统一返回
``(hidden, memory)``，因此网络、PPO 和 rollout buffer 不需要关心具体循环单元。
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

RecurrentStateTuple = tuple[torch.Tensor, torch.Tensor]


class SruLstmCell(nn.Module):
    """带输入变换门的 LSTM cell。"""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        self.transform = nn.Linear(input_size, hidden_size, bias=bias)
        nn.init.orthogonal_(self.gates.weight)
        nn.init.orthogonal_(self.transform.weight)
        if self.gates.bias is not None:
            with torch.no_grad():
                self.gates.bias[hidden_size : 2 * hidden_size].copy_(
                    1.0 + torch.randn(hidden_size)
                )

    def forward(
        self, input_: torch.Tensor, hidden: torch.Tensor, memory: torch.Tensor
    ) -> RecurrentStateTuple:
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat((input_, hidden), dim=-1)
        ).chunk(4, dim=-1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(self.transform(input_) * candidate)
        next_memory = forget_gate * memory + input_gate * candidate
        next_hidden = output_gate * torch.tanh(next_memory)
        return next_hidden, next_memory


class SruGruCell(nn.Module):
    """带输入变换门的 GRU cell。"""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gates = nn.Linear(input_size + hidden_size, 2 * hidden_size, bias=bias)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.transform = nn.Linear(input_size, hidden_size, bias=bias)
        nn.init.orthogonal_(self.gates.weight)
        nn.init.orthogonal_(self.candidate.weight)
        nn.init.orthogonal_(self.transform.weight)
        if self.gates.bias is not None:
            with torch.no_grad():
                self.gates.bias[:hidden_size].copy_(1.0 + torch.randn(hidden_size))

    def forward(
        self, input_: torch.Tensor, hidden: torch.Tensor, memory: torch.Tensor
    ) -> RecurrentStateTuple:
        del memory  # GRU 没有独立记忆状态；保留参数只是为了统一接口。
        update_gate, reset_gate = self.gates(
            torch.cat((input_, hidden), dim=-1)
        ).chunk(2, dim=-1)
        update_gate = torch.sigmoid(update_gate)
        reset_gate = torch.sigmoid(reset_gate)
        candidate_input = torch.cat((input_, reset_gate * hidden), dim=-1)
        candidate = torch.tanh(self.transform(input_) * self.candidate(candidate_input))
        next_hidden = (1.0 - update_gate) * candidate + update_gate * hidden
        return next_hidden, torch.zeros_like(next_hidden)


class SruLstmGateCell(nn.Module):
    """增加 refine gate 的 SRU-LSTM cell。"""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=bias)
        self.transform = nn.Linear(input_size, hidden_size, bias=bias)
        nn.init.orthogonal_(self.gates.weight)
        nn.init.orthogonal_(self.transform.weight)
        if self.gates.bias is not None:
            with torch.no_grad():
                self.gates.bias[hidden_size : 2 * hidden_size].copy_(
                    1.0 + torch.randn(hidden_size)
                )

    def forward(
        self, input_: torch.Tensor, hidden: torch.Tensor, memory: torch.Tensor
    ) -> RecurrentStateTuple:
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat((input_, hidden), dim=-1)
        ).chunk(4, dim=-1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(self.transform(input_) * candidate)

        # Refine gate：保留旧实现的计算顺序和公式。
        forget_gate = (
            input_gate * (1.0 - (1.0 - forget_gate).square())
            + (1.0 - input_gate) * forget_gate.square()
        )
        next_memory = forget_gate * memory + (1.0 - forget_gate) * candidate
        next_hidden = output_gate * torch.tanh(next_memory)
        return next_hidden, next_memory


class _StackedSru(nn.Module):
    """三个 SRU 变体共用的清晰时间循环。"""

    def __init__(
        self,
        cell_factory: Callable[[int, int], nn.Module],
        input_size: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            cell_factory(input_size if layer == 0 else hidden_size, hidden_size)
            for layer in range(num_layers)
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> RecurrentStateTuple:
        shape = (self.num_layers, batch_size, self.hidden_size)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        state: RecurrentStateTuple | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RecurrentStateTuple]:
        if sequence.ndim != 3:
            raise ValueError(f"sequence 应为三维张量，实际形状为 {tuple(sequence.shape)}")
        time_steps, batch_size, _ = sequence.shape
        hidden, memory = state or self.initial_state(
            batch_size, device=sequence.device, dtype=sequence.dtype
        )
        outputs: list[torch.Tensor] = []
        for time_index in range(time_steps):
            if episode_starts is not None:
                keep = (~episode_starts[time_index].bool()).view(1, batch_size, 1)
                hidden = hidden * keep
                memory = memory * keep
            layer_input = sequence[time_index]
            next_hidden: list[torch.Tensor] = []
            next_memory: list[torch.Tensor] = []
            for layer_index, recurrent_cell in enumerate(self.cells):
                layer_hidden, layer_memory = recurrent_cell(
                    layer_input, hidden[layer_index], memory[layer_index]
                )
                next_hidden.append(layer_hidden)
                next_memory.append(layer_memory)
                layer_input = layer_hidden
            hidden = torch.stack(next_hidden)
            memory = torch.stack(next_memory)
            outputs.append(layer_input)
        if not outputs:
            return sequence.new_empty((0, batch_size, self.hidden_size)), (hidden, memory)
        return torch.stack(outputs), (hidden, memory)


class SruLstm(_StackedSru):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__(SruLstmCell, input_size, hidden_size, num_layers)


class SruGru(_StackedSru):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__(SruGruCell, input_size, hidden_size, num_layers)


class SruLstmGate(_StackedSru):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__(SruLstmGateCell, input_size, hidden_size, num_layers)


class TorchLstm(nn.Module):
    """让 ``torch.nn.LSTM`` 支持项目的 episode 起点清零接口。"""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> RecurrentStateTuple:
        shape = (self.num_layers, batch_size, self.hidden_size)
        return (
            torch.zeros(shape, device=device, dtype=dtype),
            torch.zeros(shape, device=device, dtype=dtype),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        state: RecurrentStateTuple | None = None,
        episode_starts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RecurrentStateTuple]:
        if sequence.ndim != 3:
            raise ValueError(f"sequence 应为三维张量，实际形状为 {tuple(sequence.shape)}")
        time_steps, batch_size, _ = sequence.shape
        hidden, memory = state or self.initial_state(
            batch_size, device=sequence.device, dtype=sequence.dtype
        )
        if time_steps == 0:
            return sequence.new_empty((0, batch_size, self.hidden_size)), (hidden, memory)

        outputs: list[torch.Tensor] = []
        for time_index in range(time_steps):
            if episode_starts is not None:
                keep = (~episode_starts[time_index].bool()).view(1, batch_size, 1)
                hidden = hidden * keep
                memory = memory * keep
            output, (hidden, memory) = self.lstm(
                sequence[time_index : time_index + 1], (hidden, memory)
            )
            outputs.append(output[0])
        return torch.stack(outputs), (hidden, memory)


def build_recurrent(
    recurrent_type: str, input_size: int, hidden_size: int, num_layers: int
) -> nn.Module:
    """根据配置创建循环单元；名称同时用于命令行和日志目录。"""

    implementations: dict[str, type[nn.Module]] = {
        "sru-lstm": SruLstm,
        "sru-gru": SruGru,
        "sru-lstm-gate": SruLstmGate,
        "lstm": TorchLstm,
    }
    try:
        implementation = implementations[recurrent_type]
    except KeyError as error:
        choices = ", ".join(implementations)
        raise ValueError(f"未知循环单元 {recurrent_type!r}，可选值：{choices}") from error
    return implementation(input_size, hidden_size, num_layers)


__all__ = [
    "RecurrentStateTuple",
    "SruGru",
    "SruGruCell",
    "SruLstm",
    "SruLstmCell",
    "SruLstmGate",
    "SruLstmGateCell",
    "TorchLstm",
    "build_recurrent",
]

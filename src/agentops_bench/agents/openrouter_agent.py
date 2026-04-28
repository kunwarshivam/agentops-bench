"""OpenRouter agent adapter — exposes open-weights frontier models.

OpenRouter (https://openrouter.ai) presents an OpenAI-compatible
endpoint for hundreds of upstream models, including the open-weights
checkpoints we care about for the broader-coverage extension of
AgentOps-Bench: Llama, Qwen, DeepSeek, and Mistral families.

This adapter is a thin shim: it reuses the OpenAI SDK (which speaks
the OpenAI Chat Completions protocol that OpenRouter mirrors) but
points the base URL at OpenRouter and reads ``OPENROUTER_API_KEY``.

The agent_id is namespaced as ``openrouter/<vendor>/<model>`` so the
cost-lookup table (see ``scoring/cost.py``) can attribute pricing to
the right open-weights checkpoint.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import openai

from agentops_bench.agents.base import AgentAdapter
from agentops_bench.schema import (
    AgentStep,
    AgentTrace,
    Condition,
    Task,
    ToolCall,
    ToolDefinition,
)
from agentops_bench.tools import InstrumentedToolServer, serialize_tool_result

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterAgent(AgentAdapter):
    """Drives an open-weights model hosted on OpenRouter.

    The model identifier follows the OpenRouter convention,
    e.g. ``meta-llama/llama-4-maverick`` or ``qwen/qwen-3-72b-instruct``.
    """

    def __init__(
        self,
        model: str = "meta-llama/llama-4-maverick",
        max_iterations: int = 25,
        timeout_seconds: float = 300.0,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._max_iterations = max_iterations
        self._timeout = timeout_seconds
        self._temperature = temperature
        self._client = openai.AsyncOpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url=OPENROUTER_BASE_URL,
        )

    @property
    def agent_id(self) -> str:
        return f"openrouter/{self._model}"

    @property
    def model_name(self) -> str:
        return self._model

    def _format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": t.parameters,
                    },
                },
            }
            for t in tools
        ]

    async def run(
        self,
        task: Task,
        tools: list[ToolDefinition],
        tool_server: InstrumentedToolServer,
    ) -> AgentTrace:
        trace = AgentTrace(
            task_id=task.id,
            agent_id=self.agent_id,
            condition=Condition.CLEAN,
        )

        formatted_tools = self._format_tools(tools)
        system_msg = (
            "You are a helpful assistant. Solve the given task using the available tools. "
            "Think step by step. When you have the final answer, provide it clearly."
        )

        user_content = task.description
        if task.context:
            user_content += f"\n\nContext:\n{task.context}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        start_time = time.monotonic()
        total_input = 0
        total_output = 0

        for iteration in range(1, self._max_iterations + 1):
            elapsed = time.monotonic() - start_time
            if elapsed > self._timeout:
                trace.error = f"Timeout after {elapsed:.1f}s"
                break

            try:
                response = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=formatted_tools if formatted_tools else openai.NOT_GIVEN,
                    max_tokens=8192,
                    temperature=self._temperature,
                )
            except Exception as exc:
                trace.error = f"API error: {exc}"
                break

            choice = response.choices[0]
            message = choice.message

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_input += input_tokens
            total_output += output_tokens

            step = AgentStep(
                step_number=iteration,
                llm_input_tokens=input_tokens,
                llm_output_tokens=output_tokens,
            )

            if message.content:
                step.reasoning = message.content

            if message.tool_calls:
                messages.append(message.model_dump())
                for tc in message.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {"raw": tc.function.arguments}

                    call = ToolCall(
                        tool_name=tc.function.name,
                        arguments=args,
                    )
                    step.tool_calls.append(call)

                    result = await tool_server.handle_call(tc.function.name, args)
                    step.tool_results.append(result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": serialize_tool_result(result.result),
                    })
            else:
                trace.completed = True
                trace.final_output = message.content or ""
                trace.steps.append(step)
                break

            trace.steps.append(step)

        trace.total_input_tokens = total_input
        trace.total_output_tokens = total_output
        trace.wall_time_seconds = time.monotonic() - start_time

        from agentops_bench.scoring.cost import compute_cost, _lookup_pricing

        inp_price, out_price = _lookup_pricing(self.agent_id)
        trace.total_cost_usd = compute_cost(total_input, total_output, inp_price, out_price)

        return trace

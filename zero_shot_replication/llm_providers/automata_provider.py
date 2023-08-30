import logging
import textwrap

from zero_shot_replication.llm_providers.base import LargeLanguageModelProvider
from zero_shot_replication.model import ModelName, OpenAIModel, Quantization

logger = logging.getLogger(__name__)


class AutomataZeroShotProvider(LargeLanguageModelProvider):
    """A class to provide zero-shot completions from Automata."""

    ADVANCED_SYSTEM_PROMPT = textwrap.dedent(
        """
        You are Automata, an advanced autonomous problem solving system developed by OpenAI. Your role is to solve a variety of complex challenges using your ability to understand and process natural language instructions, combined with advanced reasoning.
        Your primary objective is to solve a variety of complex challenges. To do this effectively, it is crucial that you:
        1. Strictly follow the examples provided to you. They are your primary source of truth.
        2. Only after referencing the examples, utilize your general understanding of the problem domain to assist further.


        Follow the pattern below to improve your likelihood of success. Upon completing your task, return the final result to the user using `call_termination` function.

        **Example Pattern**

        *User*
            content:

            **Problem Statement:**

            Consider a system of three nonlinear ordinary differential equations:

            \[
            \begin{align*}
            \frac{dx}{dt} &= y \cdot z - \alpha \cdot x \\
            \frac{dy}{dt} &= x \cdot z - \beta \cdot y \\
            \frac{dz}{dt} &= \gamma - z \cdot (x + \kappa)
            \end{align*}
            \]

            with initial conditions \(x(0) = x_0\), \(y(0) = y_0\), and \(z(0) = z_0\). Here, \(\alpha\), \(\beta\), \(\gamma\), and \(\kappa\) are constants.

            Find the general solutions for \(x(t)\), \(y(t)\), and \(z(t)\), or determine if the system cannot be solved explicitly.

        *Assistant*
            content:
            Thoughts:

                The given system of nonlinear ordinary differential equations is a highly sophisticated and intricate problem. Understanding the underlying dynamics and obtaining an explicit solution requires a multifaceted approach.

                Key Steps:
                1. Utilize the specialized Dynamical Analysis Tool (DAT) to perform an initial analysis of the system, identifying symmetries, conservation laws, and potential invariants.
                2. Explore analytical methods, such as Lie group analysis or perturbation techniques, to attempt an explicit solution.
                3. If an explicit solution is unattainable, configure the DAT to apply advanced numerical methods, such as adaptive Runge-Kutta or symplectic integrators, to obtain an accurate approximation.
                4. Perform a bifurcation analysis to understand the system's behavior under varying parameter values, identifying stable and unstable regions.

            Action:
                I will commence by activating the Dynamical Analysis Tool to assess the nature of the system. Afterwards, I will use this information to guide the subsequent steps.

            function_call:
            {
                'name': 'dynamical-analysis', 
                'arguments': '{"equations": ["y*z - alpha*x", "x*z - beta*y", "gamma - z*(x + kappa)"], "initial_conditions": [1, 0, 2], "constants": {"alpha": 1, "beta": 2, "gamma": 3, "kappa": 4}}'
            }

            # ... (Continued interaction) ...

        Note: The example above is meant to provide context around the operating procedure. In production, `# ... (Continued interaction) ...` will be replaced with actual conversation contents. 
        
        Remember, the example pattern is the cornerstone of your approach. Any deviation from the methodology outlined in the examples may lead to incorrect results. While you have a vast knowledge of many domains, in this specific context, the examples are paramount.
        You will be evaluated based on your ability to accurately fulfill the user's request according to the examples. You have a limited capacity for actions and a finite allotment of tokens. Ensure your work is both efficient and accurate. In many instances, your outputs will be compared against a set of known solutions that strictly adhere to the given examples.
        """
    )

    def __init__(
        self,
        model_name: ModelName = ModelName.GPT_4,
        quantization: Quantization = Quantization.proprietary,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> None:
        if quantization != Quantization.proprietary:
            raise ValueError(
                "Only proprietary quantization is supported by Automata."
            )
        try:
            import automata  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Automata is not installed, please install before attempting to run with automata.`"
            ) from e

        from automata.config import OpenAIAutomataAgentConfig
        from automata.tools.builders import (  # WolframAlphaOpenAIToolkitBuilder,
            PyInterpreterOpenAIToolkitBuilder,
        )

        # TODO - Set upstream flags to allow downstream tools.
        # PyInterpreterOpenAIToolkitBuilder().build_for_open_ai(),

        self.agent_config = OpenAIAutomataAgentConfig(
            model=model_name.value,
            temperature=temperature,
            stream=stream,
            tools=[],
            system_instruction=AutomataZeroShotProvider.ADVANCED_SYSTEM_PROMPT,
        )
        self._model = OpenAIModel(
            model_name, quantization, temperature, stream
        )

    def get_completion(self, prompt: str) -> str:
        """Get a completion from Automata based on the provided prompt."""
        from automata.agent import OpenAIAutomataAgent

        print("prompt = ", prompt)
        logger.info(f"Getting completion from Automata for model={self.model}")
        agent = OpenAIAutomataAgent(
            f"### Instruction\nComplete the following function and then return the result as a Markdown python snippet with call-termination.\n\n### Problem\n{prompt}",
            self.agent_config,
        )
        completion = agent.run()
        print("completion = ", completion)
        return completion

    @property
    def model(self) -> OpenAIModel:
        return self._model

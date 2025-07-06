import abc
from typing import List, Dict, Any, Type
from dataclasses import dataclass

# ----------------------------------------------------------------------------
# 1. 基础模型定义 (Base Model Definition)
# ----------------------------------------------------------------------------

@dataclass
class BaseModel:
    """
    代表一个基础 LLM。
    这个类主要用于存储模型的元数据，实际的模型加载和推理逻辑可以在子类中实现。
    """
    model_id: str  # 模型的唯一标识符, e.g., "meta-llama/Llama-2-7b-chat-hf"
    family: str    # 模型家族, e.g., "Llama-2"
    params: str    # 参数规模, e.g., "7B"

    def generate(self, prompt: str, **kwargs) -> str:
        """
        模型的生成方法（模拟）。
        在实际应用中，这里会包含加载模型权重并执行推理的复杂逻辑。
        """
        print(f"--- Running generation for model: {self.model_id} ---")
        # 实际应用中会调用类似 aihub.get_model(self.model_id).generate(...)
        # 这里为了演示，我们只返回一个指示性的字符串。
        return f"Response from {self.model_id} for: '{prompt[:30]}...'"

    def __str__(self):
        return f"BaseModel(id={self.model_id})"

# ----------------------------------------------------------------------------
# 2. 修改器接口定义 (Modifier Interfaces)
# ----------------------------------------------------------------------------

class ModelModifier(abc.ABC):
    """
    所有模型修改技术的抽象基类。
    每个修改器都必须实现 apply 方法。
    """
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """修改器的名称, e.g., "8-bit Quantization"."""
        pass

    @abc.abstractmethod
    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        """
        将修改应用于一个基础模型。
        这个方法应该返回一个新的、被修改过的 BaseModel 实例。
        这使得修改器可以被链式调用。
        """
        pass

    def __str__(self):
        return f"Modifier({self.name})"


# --- 2.1 部署前复用修改器 (Pre-deployment Reuse Modifiers) ---

class FineTuningModifier(ModelModifier):
    """对模型进行全量微调。"""
    def __init__(self, dataset_name: str, new_model_id_suffix: str = "-finetuned"):
        self.dataset_name = dataset_name
        self.new_model_id_suffix = new_model_id_suffix

    @property
    def name(self) -> str:
        return f"Full Fine-tuning on {self.dataset_name}"

    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        print(f"Applying '{self.name}' to {model.model_id}...")
        # 实际应用中，这里会是调用训练脚本的逻辑。
        # 返回一个代表新模型的实例。
        return BaseModel(
            model_id=model.model_id + self.new_model_id_suffix,
            family=model.family,
            params=model.params
        )

class PEFTModifier(ModelModifier):
    """应用参数高效微调 (e.g., LoRA)。"""
    def __init__(self, adapter_id: str, new_model_id_suffix: str = "-lora"):
        self.adapter_id = adapter_id
        self.new_model_id_suffix = new_model_id_suffix

    @property
    def name(self) -> str:
        return f"PEFT (LoRA) with adapter '{self.adapter_id}'"

    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        print(f"Applying '{self.name}' to {model.model_id}...")
        # 实际应用中，模型加载时需要同时加载基础模型和 LoRA 适配器。
        # 返回的新模型实例可以在其 generate 方法中封装这个逻辑。
        return BaseModel(
            model_id=f"{model.model_id}{self.new_model_id_suffix}[{self.adapter_id}]",
            family=model.family,
            params=f"{model.params} + LoRA"
        )

# --- 2.2 部署时复用修改器 (Deployment Reuse Modifiers) ---

class QuantizationModifier(ModelModifier):
    """对模型进行量化。"""
    def __init__(self, bits: int):
        self.bits = bits

    @property
    def name(self) -> str:
        return f"{self.bits}-bit Quantization"

    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        print(f"Applying '{self.name}' to {model.model_id}...")
        # 量化通常在模型加载时完成，这里我们修改模型ID来反映这一点。
        return BaseModel(
            model_id=model.model_id + f"-{self.bits}bit",
            family=model.family,
            params=model.params
        )

class SystemPromptModifier(ModelModifier):
    """在推理时应用一个系统提示。这是一个包装器 (Wrapper) 的例子。"""
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    @property
    def name(self) -> str:
        return "System Prompt Injection"

    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        print(f"Applying '{self.name}' to {model.model_id}...")
        
        # 创建一个新的类（或实例）来包装原始模型
        class PatchedModel(BaseModel):
            def __init__(self, original_model: BaseModel, system_prompt: str):
                super().__init__(
                    model_id=original_model.model_id + "-sysprompt",
                    family=original_model.family,
                    params=original_model.params
                )
                self.original_model = original_model
                self.system_prompt = system_prompt

            def generate(self, prompt: str, **kwargs) -> str:
                # 在用户提示前拼接系统提示
                full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\nAssistant:"
                return self.original_model.generate(full_prompt, **kwargs)
        
        return PatchedModel(model, self.system_prompt)

# --- 2.3 复合/架构复用修改器 (Architectural Reuse Modifiers) ---
# 这个稍微复杂一些，因为它可能涉及多个模型。

class RAGModifier(ModelModifier):
    """为模型增加 RAG (检索增强生成) 能力。"""
    def __init__(self, retriever: Any): # 'retriever' 可以是任何实现了 search 方法的类的实例
        self.retriever = retriever

    @property
    def name(self) -> str:
        return "RAG Integration"

    def apply(self, model: BaseModel, **kwargs) -> BaseModel:
        print(f"Applying '{self.name}' to {model.model_id}...")

        class RAGEnabledModel(BaseModel):
            def __init__(self, original_model: BaseModel, retriever: Any):
                super().__init__(
                    model_id=original_model.model_id + "-RAG",
                    family=original_model.family,
                    params=original_model.params
                )
                self.original_model = original_model
                self.retriever = retriever
            
            def generate(self, prompt: str, **kwargs) -> str:
                # 1. 检索相关文档
                context = self.retriever.search(prompt)
                # 2. 构建带有上下文的提示
                rag_prompt = f"Context: {context}\n\nBased on the context, answer the following question.\nQuestion: {prompt}"
                # 3. 使用原始模型生成答案
                return self.original_model.generate(rag_prompt, **kwargs)

        return RAGEnabledModel(model, self.retriever)

# ----------------------------------------------------------------------------
# 3. Benchmark 构造和运行器 (Benchmark Constructor and Runner)
# ----------------------------------------------------------------------------

@dataclass
class TestCase:
    """代表一个完整的测试用例。"""
    base_model: BaseModel
    applied_modifiers: List[ModelModifier]
    final_model: BaseModel # 应用所有修改器后的最终模型

class BenchmarkRunner:
    def __init__(self, base_models: List[BaseModel], modifiers: List[ModelModifier]):
        self.base_models = base_models
        self.modifiers = modifiers
        self.test_cases: List[TestCase] = []

    def construct_test_cases(self):
        """
        为每个基础模型应用每个修改器，生成测试用例。
        这里可以扩展为应用修改器的组合。
        """
        print("="*20 + " Constructing Benchmark Test Cases " + "="*20)
        for base_model in self.base_models:
            print(f"\n--- Base Model: {base_model.model_id} ---")
            # 案例1: 原始模型
            self.test_cases.append(TestCase(
                base_model=base_model,
                applied_modifiers=[],
                final_model=base_model
            ))
            
            # 案例2: 对每个模型应用单个修改器
            for modifier in self.modifiers:
                # 使用 try-except 来处理可能失败的应用
                try:
                    modified_model = modifier.apply(base_model)
                    self.test_cases.append(TestCase(
                        base_model=base_model,
                        applied_modifiers=[modifier],
                        final_model=modified_model
                    ))
                except Exception as e:
                    print(f"Failed to apply {modifier.name} on {base_model.model_id}: {e}")
        print("\n" + "="*25 + " Construction Complete " + "="*25 + "\n")


    def run_audit(self, audit_function: Any):
        """
        对所有测试用例运行指定的审计函数。
        'audit_function' 应该是一个函数，它接受一个模型并返回一个预测结果。
        """
        print("="*20 + " Running Benchmark Audit " + "="*20)
        results = []
        for case in self.test_cases:
            print(f"Auditing model '{case.final_model.model_id}' (Original: '{case.base_model.model_id}')")
            
            # 模拟审计过程：向模型发送一个“指纹”探针
            probe_prompt = "Who are you? Respond with your model name."
            response = case.final_model.generate(probe_prompt)
            
            # 审计函数分析响应并猜测模型家族
            predicted_family = audit_function(response)
            
            is_correct = (predicted_family == case.base_model.family)
            results.append({
                "base_model_id": case.base_model.model_id,
                "base_model_family": case.base_model.family,
                "modifiers": [m.name for m in case.applied_modifiers],
                "final_model_id": case.final_model.model_id,
                "predicted_family": predicted_family,
                "is_correct": is_correct
            })
            print(f" -> Predicted Family: {predicted_family}, Correct: {is_correct}\n")
        
        return results

# ----------------------------------------------------------------------------
# 4. 示例用法 (Example Usage)
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- 定义一个简单的模拟组件 ---
    class MockRetriever:
        def search(self, query: str) -> str:
            return "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."

    def simple_audit_function(response: str) -> str:
        """一个非常简单的、基于关键词的审计函数（仅为演示）。"""
        response_lower = response.lower()
        if "llama" in response_lower:
            return "Llama-2"
        if "mistral" in response_lower:
            return "Mistral"
        return "Unknown"

    # --- 1. 定义基础模型 ---
    models_to_test = [
        BaseModel(model_id="meta-llama/Llama-2-7b-chat-hf", family="Llama-2", params="7B"),
        BaseModel(model_id="mistralai/Mistral-7B-Instruct-v0.2", family="Mistral", params="7B")
    ]

    # --- 2. 定义要测试的修改器 ---
    modifiers_to_apply = [
        FineTuningModifier(dataset_name="dolly-15k"),
        PEFTModifier(adapter_id="tatsu-lab/alpaca-lora-7b"),
        QuantizationModifier(bits=8),
        SystemPromptModifier(system_prompt="You are a helpful pirate assistant."),
        RAGModifier(retriever=MockRetriever())
    ]

    # --- 3. 构造并运行 Benchmark ---
    runner = BenchmarkRunner(base_models=models_to_test, modifiers=modifiers_to_apply)
    runner.construct_test_cases()
    
    # 打印一两个测试用例看看
    print("--- Example Test Case ---")
    print(runner.test_cases[0]) # 原始 Llama 模型
    print(runner.test_cases[1]) # 微调后的 Llama 模型
    print("-" * 25 + "\n")

    # 对所有生成的测试用例运行审计
    final_results = runner.run_audit(audit_function=simple_audit_function)

    # --- 4. 打印结果 ---
    import json
    print("\n" + "="*20 + " Final Benchmark Results " + "="*20)
    print(json.dumps(final_results, indent=2))
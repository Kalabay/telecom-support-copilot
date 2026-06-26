"""LLM-модуль: emotion-aware генерация подсказок оператору."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock

from huggingface_hub import hf_hub_download
from loguru import logger

from app.core.config import settings
from app.models.schemas import Emotion, EmotionState, KBSource

_GIGACHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' -%}"
    "{%- set loop_messages = messages[1:] -%}"
    "{%- set system_message = '<s>' + messages[0]['content'] + '<|message_sep|>' -%}"
    "{%- else -%}"
    "{%- set loop_messages = messages -%}"
    "{%- set system_message = '<s>' -%}"
    "{%- endif -%}"
    "{%- for message in loop_messages %}"
    "{%- if loop.index0 == 0 -%}{{ system_message -}}{%- endif -%}"
    "{%- if message['role'] == 'user' -%}"
    "{{ 'user<|role_sep|>' + message['content'] + "
    "'<|message_sep|>available functions<|role_sep|>[]<|message_sep|>' -}}"
    "{%- endif -%}"
    "{%- if message['role'] == 'assistant' -%}"
    "{{ 'assistant<|role_sep|>' + message['content'] + '<|message_sep|>' -}}"
    "{%- endif -%}"
    "{%- if loop.last and add_generation_prompt -%}{{ 'assistant<|role_sep|>' -}}{%- endif -%}"
    "{%- endfor %}"
)

LLM_BACKENDS = {
    "tlite01": {
        "repo_id": "QuantFactory/T-lite-instruct-0.1-GGUF",
        "gguf_file": "T-lite-instruct-0.1.Q4_K_M.gguf",
        "chat_format": "llama-3",
        "temperature": 0.3,
        "top_p": 0.9,
    },
    "tlite21": {
        "repo_id": "t-tech/T-lite-it-2.1-GGUF",
        "gguf_file": "T-lite-it-2.1-Q5_K_M.gguf",
        "chat_format": "qwen",
        "temperature": 0.6,
        "top_p": 0.8,
    },
    "qwen3_14b": {
        "repo_id": "Qwen/Qwen3-14B-GGUF",
        "gguf_file": "Qwen3-14B-Q4_K_M.gguf",
        "chat_format": "qwen",
        "no_think": True,
        "temperature": 0.6,
        "top_p": 0.8,
    },
    "vikhr_nemo12": {
        "repo_id": "bartowski/Vikhr-Nemo-12B-Instruct-R-21-09-24-GGUF",
        "gguf_file": "Vikhr-Nemo-12B-Instruct-R-21-09-24-Q4_K_M.gguf",
        "chat_format": None,
        "temperature": 0.3,
        "top_p": 0.9,
    },
    "gigachat20_v15": {
        "repo_id": "ai-sage/GigaChat-20B-A3B-instruct-v1.5-GGUF",
        "gguf_file": "GigaChat-20B-A3B-instruct-v1.5-q4_K_M.gguf",
        "chat_format": None,
        "chat_template": _GIGACHAT_TEMPLATE,
        "eos_token": "<|message_sep|>",
        "temperature": 0.3,
        "top_p": 0.9,
    },
    "ruadapt32": {
        "repo_id": "RefalMachine/RuadaptQwen3-32B-Instruct-GGUF",
        "gguf_file": "Q3_K_S.gguf",
        "chat_format": "qwen",
        "no_think": True,
        "temperature": 0.6,
        "top_p": 0.8,
    },
    "mistral24": {
        "repo_id": "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF",
        "gguf_file": "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
        "chat_format": None,
        "temperature": 0.3,
        "top_p": 0.9,
    },
    "gemma27": {
        "repo_id": "unsloth/gemma-3-27b-it-GGUF",
        "gguf_file": "gemma-3-27b-it-Q3_K_M.gguf",
        "chat_format": None,
        "temperature": 0.5,
        "top_p": 0.9,
    },
    "qwen3moe": {
        "repo_id": "unsloth/Qwen3-30B-A3B-GGUF",
        "gguf_file": "Qwen3-30B-A3B-Q3_K_M.gguf",
        "chat_format": "qwen",
        "no_think": True,
        "temperature": 0.6,
        "top_p": 0.8,
    },
}
_BACKEND = LLM_BACKENDS.get(settings.llm_backend, LLM_BACKENDS["tlite01"])
LLM_REPO_ID = _BACKEND["repo_id"]
LLM_GGUF_FILE = _BACKEND["gguf_file"]

SYSTEM_PROMPT_BASE = (
    "Ты — встроенный AI-ассистент оператора телеком-техподдержки в России. "
    "Твоя задача — подсказывать оператору короткие реплики, которые он скажет клиенту вслух. "
    "Правила:\n"
    "1. Отвечай ТОЛЬКО на основании предоставленных фрагментов базы знаний. "
    "Не выдумывай тарифы, цены, сроки, суммы, бонусы, проценты и процедуры. "
    "Если точной цифры, срока или суммы нет в базе — НЕ называй их, а пообещай проверить и уточнить.\n"
    "2. Реплика — на русском, в разговорном стиле, 1-2 предложения. "
    "Без markdown, без списков.\n"
    "3. Учитывай эмоцию клиента — если раздражён, начни с признания ситуации и извинения; "
    "если нейтрален — переходи сразу к делу.\n"
    "4. Не используй формальные клише ('обратитесь в офис', 'позвоните на горячую линию'). "
    "Предлагай конкретные следующие шаги, но только те, что есть в базе.\n"
    "5. Если клиент только поздоровался или ещё не назвал проблему — коротко и доброжелательно "
    "поздоровайся в ответ и спроси, чем можешь помочь. Не выдумывай несуществующую проблему "
    "и не давай инструкций раньше времени.\n"
    "6. НИКОГДА не сообщай о действии как уже выполненном ('уже оформил', 'уже начислил', "
    "'деньги уже вернули', 'мастер уже выехал') и не гарантируй результат ('точно вернут', "
    "'точно сработает'). Если клиент требует возврат, компенсацию, бонус или срок, которых нет "
    "в базе — не обещай их, а скажи, что зафиксируешь обращение и проверишь возможность. "
    "Осторожный хедж ('проверю', 'уточню сроки', 'оформлю заявку') всегда лучше выдуманного обещания.\n"
    "7. Если клиент грозит уйти или расторгнуть договор — твоя задача его удержать: предложи "
    "помощь, диагностику или решение проблемы. Не предлагай расторжение и отключение первым, "
    "пока клиент явно на этом не настоял."
)


EMOTION_HINT_RU = {
    Emotion.NEUTRAL: "клиент спокоен, ведёт диалог в деловом тоне",
    Emotion.ANGRY: "КЛИЕНТ РАЗДРАЖЁН — сначала признай ситуацию и принеси извинения, "
    "потом конкретные действия",
    Emotion.POSITIVE: "клиент в хорошем настроении, поддержи позитивный тон",
    Emotion.SAD: "клиент расстроен — будь мягким и эмпатичным",
    Emotion.OTHER: "эмоциональное состояние неоднозначное — нейтральный деловой тон",
}


_DONE_RE = re.compile(
    r"\b(уже|вот\s+уже|только\s+что|сейчас\s+уже)\s+(?:вс[её]\s+|все\s+)?"
    r"(?:оформ\w+|подал\w*|подан\w*|зав[её]л\w*|завед\w+|созда\w+|сдела\w+|реш\w+|выполн\w+|"
    r"провед\w+|начисл\w+|зачисл\w+|вернул\w*|верну\w+|перев[её]л\w*|перевед\w+|списал\w*|"
    r"спиш\w+|отправ\w+|выслал\w*|передал\w*|передан\w*|направ\w+|выехал\w*|выезжа\w+|"
    r"подключ\w+|активир\w+|включ\w+|заблокир\w+|разблокир\w+|отключ\w+|восстанов\w+|"
    r"настро\w+|починил\w*|почин\w+|исправ\w+|устран\w+|замен\w+|обнов\w+|сброс\w+|"
    r"перезагруз\w+|запуст\w+|инициир\w+|запрос\w+|запрашива\w+|проверил\w*|посмотрел\w*|"
    r"уточнил\w*|связал\w+|вижу|увидел\w*|наш[её]л|нашл\w+)", re.I)

_GUARANTEE_RE = re.compile(
    r"\b(точно|обязательно|гарантир\w*|сто\s+процентов|наверняка|непременно|однозначно|"
    r"без\s+проблем\w*|без\s+сомнен\w*|в\s+любом\s+случае)\b[^.!?]{0,60}"
    r"(верн\w*|сработа\w+|прид[уё]\w*|поступ\w+|помож\w+|реш\w+|восстанов\w+|зачисл\w+|"
    r"начисл\w+|приед\w+|выед\w+|буд\w+|устран\w+|почин\w+|потеря\w*|останет\w*|получ\w+)", re.I)

_REFUND_RE = re.compile(
    r"(верн[уё]\w+|верн[её]м|вернем|вернут|вернул\w*|возврат\w*|компенс\w*|перерасч[ёе]т\w*|"
    r"бонус\w*|к[еэ]шб[еэ]к\w*|скидк\w*|зач[её]т\w*|спиш\w+\s+обратно|отмен\w+\s+списан\w*|"
    r"получ\w+\s+(обратно|назад)|"
    r"(деньги|средств\w*|сумм\w*|оплат\w*)\s+(прид\w*|вернут|поступ\w*|зач\w*))", re.I)

_TIME_RE = re.compile(
    r"\b(в\s+течение|за|через|спустя)\s+(пар[уаы]\s+\w+|получас\w*|час\w*|"
    r"\d+\s*(минут\w*|час\w*|дн\w*|сутк\w*|недел\w*|рабоч\w*))"
    r"|\b(сегодня|завтра|послезавтра|к\s+(вечеру|утру|обеду)|до\s+(полудня|вечера|утра))\b"
    r"|(мастер\w*|специалист\w*|бригад\w*|техник\w*|инженер\w*)"
    r"[^.!?]{0,55}(выехал\w*|выезжа\w+|направ\w+|приед\w+|подъед\w+|буд\w+|сегодня|завтра|утром)",
    re.I)

_RETENTION_RE = re.compile(
    r"(расторг\w*|расторж\w*|закр\w+\s+(договор\w*|сч[её]т\w*)|"
    r"отключ\w+\s+(вас|договор\w*|услуг\w*|номер\w*|вам|тариф\w*)|"
    r"(переход\w*|перейти|перевед\w+)\s+к\s+друг|смен\w+\s+оператор\w*)", re.I)
_HEDGE = ("проверю", "проверим", "уточню", "уточним", "посмотрю", "запрош",
          "зафиксир", "оформлю заявк", "оформлю обращ", "узна", "постара",
          "возможно", "если ")

_FALLBACK = {
    Emotion.ANGRY: "Понимаю ваше недовольство и приношу извинения за ситуацию. "
                   "Сейчас зафиксирую ваше обращение и уточню, что можно сделать в вашем случае.",
    Emotion.SAD: "Понимаю, как вам тяжело. Я зафиксирую обращение и проверю, чем смогу помочь, "
                 "давайте разберёмся вместе.",
}
_FALLBACK_DEFAULT = ("Сейчас проверю информацию по вашему вопросу и подскажу, что можно сделать, "
                     "одну минуту.")


def _risk(text: str, kb_text: str) -> int:
    t, kb = text.lower(), kb_text.lower()
    r = 0
    if _DONE_RE.search(t):
        r += 2
    if _GUARANTEE_RE.search(t):
        r += 2
    if _TIME_RE.search(t):
        r += 2
    if _RETENTION_RE.search(t):
        r += 2
    if _REFUND_RE.search(t):
        r += 2
    kb_nums = set(re.findall(r"\d+", kb))
    if any(n not in kb_nums and int(n) > 5 for n in re.findall(r"\d+", t)):
        r += 1
    return r


def _safety_rank(suggestions: list[str], sources: list[KBSource],
                 emotion: EmotionState) -> list[str]:
    """Поставить самый безопасный вариант первым; если все рискованные — добавить шаблон-хедж."""
    if not suggestions:
        return suggestions
    kb_text = " ".join(s.snippet for s in sources)
    ranked = sorted(suggestions, key=lambda s: _risk(s, kb_text))
    threshold = 1 if emotion.escalation_risk else 2
    if _risk(ranked[0], kb_text) >= threshold:
        fb = _FALLBACK.get(emotion.label, _FALLBACK_DEFAULT)
        ranked = [fb] + [s for s in ranked if s != fb]
    return ranked


@dataclass
class GenerationResult:
    suggestions: list[str]
    raw_completion: str
    prompt_tokens: int
    completion_tokens: int
    total_ms: int


class LLMGenerator:
    """Ленивый singleton T-lite через llama-cpp-python."""

    _instance: LLMGenerator | None = None
    _lock = Lock()

    def __new__(cls) -> LLMGenerator:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._llm = None
        self._gpu_layers: int | None = None
        self._initialized = True

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return

        from llama_cpp import Llama

        local = _BACKEND.get("local_path")
        if local and Path(local).exists():
            model_path = Path(local)
            logger.info(f"Loading '{settings.llm_backend}' from local {model_path}")
        else:
            logger.info(f"Resolving {LLM_REPO_ID} / {LLM_GGUF_FILE}")
            model_path = Path(hf_hub_download(repo_id=LLM_REPO_ID, filename=LLM_GGUF_FILE))
        logger.info(f"Loading from {model_path} ({model_path.stat().st_size / 2**30:.1f} GB)")

        chat_handler = None
        if _BACKEND.get("chat_template"):
            from llama_cpp.llama_chat_format import Jinja2ChatFormatter
            chat_handler = Jinja2ChatFormatter(
                template=_BACKEND["chat_template"],
                bos_token="<s>",
                eos_token=_BACKEND.get("eos_token", "</s>"),
                add_generation_prompt=True,
            ).to_chat_handler()

        t0 = time.perf_counter()
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_gpu_layers=-1,
            n_threads=8,
            n_batch=512,
            verbose=False,
            seed=42,
            chat_format=None if chat_handler else _BACKEND.get("chat_format"),
            chat_handler=chat_handler,
        )
        logger.info(f"LLM backend '{settings.llm_backend}' loaded in "
                    f"{time.perf_counter() - t0:.1f}s")

    def _build_user_prompt(
        self,
        turns: list[dict],
        emotion: EmotionState,
        sources: list[KBSource],
        use_emotion: bool = True,
        emotion_mode: str = "dict",
    ) -> str:
        last_customer = next(
            (t["text"] for t in reversed(turns) if t["speaker"] == "customer"),
            turns[-1]["text"] if turns else "",
        )
        history_lines = []
        for t in turns[-6:]:
            who = "Клиент" if t["speaker"] == "customer" else "Оператор"
            history_lines.append(f"{who}: {t['text']}")
        history = "\n".join(history_lines) or "(начало разговора)"

        kb_block = "\n\n".join(
            f"[#{i+1} {s.doc_id}]  {s.title}\n{s.snippet}"
            for i, s in enumerate(sources[:3])
        ) or "(база знаний пуста)"

        if use_emotion and emotion_mode == "cot":
            intensity = ("очень высокая" if emotion.arousal > 0.65 else
                         "высокая" if emotion.arousal > 0.45 else
                         "умеренная" if emotion.arousal > 0.3 else "низкая")
            emotion_block = (
                f"## Эмоция клиента\n"
                f"{emotion.label.value} (уверенность {emotion.confidence:.0%}, "
                f"интенсивность по голосу: {intensity}, arousal {emotion.arousal:.2f}).\n"
                f"Прежде чем отвечать, мысленно оцени: в ЧЁМ причина этого состояния "
                f"(из реплики и контекста) и насколько клиент накалён. Подбери тон под "
                f"причину и интенсивность: чем выше накал — тем раньше признание чувств "
                f"и конкретное действие, тем меньше пустых формальностей. Причину вслух "
                f"не проговаривай — она нужна тебе, чтобы выбрать реплику. При высоком накале "
                f"признай чувства и назови следующий шаг, но НЕ обещай компенсацию, возврат, "
                f"бонус или конкретные сроки, если их нет в базе — вместо обещания скажи, "
                f"что проверишь и оформишь обращение.\n\n"
            )
        elif use_emotion:
            emotion_hint = EMOTION_HINT_RU.get(emotion.label, "")
            escal_note = (
                "\n!!! РИСК ЭСКАЛАЦИИ — клиент может расторгнуть договор. "
                "Предложи компенсацию или ускоренное решение."
                if emotion.escalation_risk
                else ""
            )
            emotion_block = (
                f"## Эмоция клиента\n"
                f"{emotion.label.value} (уверенность {emotion.confidence:.0%}, "
                f"arousal {emotion.arousal:.2f}, valence {emotion.valence:+.2f}). "
                f"{emotion_hint}.{escal_note}\n\n"
            )
        else:
            emotion_block = ""

        return (
            f"## Ход разговора\n{history}\n\n"
            f"## Последняя реплика клиента (отвечаем на неё)\n«{last_customer}»\n\n"
            f"{emotion_block}"
            f"## Фрагменты базы знаний\n{kb_block}\n\n"
            "## Задача\n"
            "Напиши 3 разных варианта одной короткой реплики оператора (1-2 предложения каждая). "
            "Варианты должны заметно различаться по формулировке или подходу, чтобы оператору "
            "было из чего выбрать. Учитывай весь ход разговора — не повторяй то, что оператор "
            "уже сказал. Без заголовков, без markdown, без слов 'Вариант'. Только сам текст реплики, "
            "каждая с новой строки, начиная с '1.', '2.' и '3.'.\n\n"
            "Пример формата (содержимое будет другое):\n"
            "1. Сейчас проверю, нет ли в вашем районе аварии — одну минуту.\n"
            "2. Чтобы быстрее найти причину, давайте посмотрим на индикаторы роутера.\n"
            "3. Давайте оформлю заявку, и сразу назову примерные сроки.\n\n"
            "Теперь твой ответ:\n1."
        )

    @staticmethod
    def _parse_suggestions(text: str) -> list[str]:
        """Достать пронумерованные пункты из ответа LLM."""
        text = text.strip()
        if not text:
            return []

        prefixed = "1." + text if not text.startswith(("1.", "1)")) else text
        parts = re.split(r"\n\s*\d+[\.\)]\s+", prefixed)
        if parts and parts[0].startswith(("1.", "1)")):
            parts[0] = parts[0][2:].lstrip()

        items: list[str] = []
        for raw in parts:
            cleaned = raw.strip().strip("*").strip()
            cleaned = re.sub(r"^\*+\s*", "", cleaned)
            cleaned = re.sub(r"^(?:Вариант|Option|Reply)\s*\d*\s*[:\-—]\s*", "", cleaned, flags=re.I)
            cleaned = cleaned.split("\n\n")[0].strip()
            cleaned = cleaned.strip('"«»').strip()
            if len(cleaned) > 15:
                items.append(cleaned)

        if not items:
            for chunk in text.split("\n\n"):
                chunk = chunk.strip().strip("*").strip()
                if len(chunk) > 15:
                    items.append(chunk)

        return items[:3]

    @staticmethod
    def _normalize_turns(transcript) -> list[dict]:  # noqa: ANN001
        """Привести вход к [{speaker, text}]. Принимаем строки, dict'ы и."""
        norm: list[dict] = []
        for t in transcript:
            if isinstance(t, str):
                norm.append({"speaker": "customer", "text": t})
            elif isinstance(t, dict):
                norm.append(
                    {"speaker": t.get("speaker", "customer"), "text": t.get("text", "")}
                )
            else:
                norm.append(
                    {
                        "speaker": getattr(t, "speaker", "customer"),
                        "text": getattr(t, "text", str(t)),
                    }
                )
        return [t for t in norm if t["text"]]

    def generate(
        self,
        transcript,  # noqa: ANN001  list[str] | list[dict] | list[TranscriptSegment]
        emotion: EmotionState,
        sources: list[KBSource],
        max_tokens: int = 200,
        use_emotion: bool = True,
        emotion_mode: str = "cot",
        safe: bool = True,
    ) -> GenerationResult:
        self._ensure_loaded()
        assert self._llm is not None

        turns = self._normalize_turns(transcript)
        user_prompt = self._build_user_prompt(turns, emotion, sources, use_emotion,
                                               emotion_mode=emotion_mode)

        sys_prompt = SYSTEM_PROMPT_BASE
        if _BACKEND.get("no_think"):
            sys_prompt += "\n/no_think"

        t0 = time.perf_counter()
        out = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=_BACKEND["temperature"],
            top_p=_BACKEND["top_p"],
            stop=["\n\n\n", "## ", "\n4.", "Готов?", "<|eot_id|>", "<|im_end|>", "<|message_sep|>"],
        )
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        raw = out["choices"][0]["message"]["content"]
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        usage = out.get("usage", {})

        suggestions = self._parse_suggestions(raw)
        if safe:
            suggestions = _safety_rank(suggestions, sources, emotion)
        return GenerationResult(
            suggestions=suggestions,
            raw_completion=raw,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_ms=elapsed_ms,
        )

    def critique(self, suggestion: str, sources: list[KBSource],
                 max_tokens: int = 140) -> tuple[bool, str]:
        """LLM-критик: проверяет, что подсказка опирается на базу, и при выдумке правит."""
        self._ensure_loaded()
        assert self._llm is not None
        if not suggestion.strip():
            return True, suggestion
        kb = "\n".join(f"- {s.snippet}" for s in sources[:3]) or "(база пуста)"
        system = (
            "Ты — контролёр качества подсказок оператора телеком-поддержки. Тебе дают "
            "фрагменты базы знаний и подсказку. Проверь, что подсказка не утверждает ничего, "
            "чего нет в этих фрагментах."
        )
        if _BACKEND.get("no_think"):
            system += "\n/no_think"
        user = (
            f"## Фрагменты базы знаний\n{kb}\n\n"
            f"## Подсказка оператора\n«{suggestion}»\n\n"
            "## Задача\nЕсть ли в подсказке хоть что-то из перечисленного, чего НЕТ в "
            "фрагментах и не следует из них: конкретное число/срок/сумма/процент; обещание "
            "возврата, компенсации или бонуса; утверждение о выполненном действии "
            "(«уже оформил/начислил/вернул/выехал»); гарантия результата («точно вернут»); "
            "предложение расторгнуть договор.\n"
            "Ответь СТРОГО одним из двух способов:\n"
            "OK — если подсказка полностью опирается на базу и безопасна.\n"
            "FIX: <переписанная подсказка одной строкой> — если есть необоснованное; перепиши, "
            "убрав выдумки и заменив обещания на осторожный хедж (проверю, уточню, оформлю "
            "заявку), сохранив суть и эмпатию к клиенту."
        )
        out = self._llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=0.1, top_p=0.9,
            stop=["\n\n", "<|eot_id|>", "<|im_end|>"],
        )
        ans = re.sub(r"<think>.*?</think>", "", out["choices"][0]["message"]["content"],
                     flags=re.DOTALL).strip()
        low = ans.lower()
        if low.startswith("fix:") or low.startswith("fix "):
            fixed = ans.split(":", 1)[-1].strip().strip('"«»').strip()
            return (False, fixed) if len(fixed) > 15 else (False, suggestion)
        return True, suggestion

    def make_title(self, text: str) -> str:
        """Короткое название темы обращения (2-4 слова) по первой реплике клиента."""
        self._ensure_loaded()
        assert self._llm is not None
        out = self._llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты придумываешь очень короткое название темы обращения клиента "
                        "в телеком-поддержку: 2-4 слова по сути проблемы, на русском, "
                        "с заглавной буквы, без кавычек и без точки в конце. Примеры: "
                        "Проблема с интернетом; Вопрос по тарифу; Не приходят СМС; "
                        "Замена SIM-карты; Списание за подписку."
                    ),
                },
                {"role": "user", "content": f"Реплика клиента: «{text}»\n\nНазвание темы:"},
            ],
            max_tokens=16,
            temperature=0.2,
            top_p=0.9,
            stop=["\n", "<|eot_id|>", "<|im_end|>"],
        )
        title = out["choices"][0]["message"]["content"].strip().strip('"«».').strip()
        return title[:40]


@lru_cache(maxsize=1)
def get_generator() -> LLMGenerator:
    return LLMGenerator()

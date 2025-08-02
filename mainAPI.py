import asyncio
import logging
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import os
import json
import pandas as pd
import pdfplumber
from collections import defaultdict
from flask import Flask, request, jsonify
import threading
import re

def read_two_texts(path1: str, path2: str) -> dict:
    contents = {}
    for name, path in (("file1", path1), ("file2", path2)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                contents[name] = f.read()
        except FileNotFoundError:
            contents[name] = ""
            print(f"Warning: ไม่พบไฟล์ {path}")
        except Exception as e:
            contents[name] = ""
            print(f"Error อ่านไฟล์ {path}: {e}")
    return contents

# ตัวอย่างเรียกใช้งาน
texts = read_two_texts("txt_outputs/doc2_full.txt", "txt_outputs/doc3_full.txt")
fileA_text = texts["file1"][:2000] if texts["file1"] else ""  # จำกัดขนาดเพื่อความเร็ว
fileB_text = texts["file2"][:2000] if texts["file2"] else ""

# --- MCP / LLM imports ---
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:
    print("Warning: MCP not available, using demo tools only")
    ClientSession = None

try:
    import aiohttp
except ImportError as e:
    print(f"❌ HTTP import error: {e}")
    print("Install with: pip install aiohttp")
    sys.exit(1)

# --- Config dataclasses ---
@dataclass
class LLMConfig:
    base_url: str = "http://172.16.30.121:8000/v1"
    api_key: str = "not-needed"
    model: str = "scb10x/typhoon2.1-gemma3-12b"
    max_tokens: int = 500  # Increased to allow for longer responses
    temperature: float = 0.7  # ลดความสุ่มเพื่อความเร็ว

@dataclass
class MCPConfig:
    server_url: str = "https://mcp-hackathon.cmkl.ai/mcp"

# --- Tool & Agent classes ---
class SimpleTool:
    def __init__(self, name: str, description: str, input_schema: dict, mcp_session):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.mcp_session = mcp_session

    async def call(self, **kwargs) -> str:
        try:
            result = await self.mcp_session.call_tool(self.name, kwargs)
            if hasattr(result, 'content') and result.content:
                text_parts = []
                for content in result.content:
                    if hasattr(content, 'text'):
                        text_parts.append(content.text)
                    elif isinstance(content, dict) and 'text' in content:
                        text_parts.append(content['text'])
                    elif isinstance(content, str):
                        text_parts.append(content)
                return '\n'.join(text_parts)[:500] if text_parts else str(result)[:500]  # จำกัดขนาด
            return str(result)[:500]
        except Exception as e:
            return f"Error: {str(e)}"

class SimpleAgent:
    def __init__(self, llm_client, tools: List[SimpleTool]):
        self.llm_client = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.conversation_history = []
        self.max_retries = 2  # ลดจาก 5
        self.retry_delay = 1  # ลดจาก 5 วินาที
        self.history_lock = asyncio.Lock()

    def _create_system_prompt(self) -> str:
        tools_desc = ""
        if self.tools:
            tools_list = [f"- {t.name}: {t.description}" for t in self.tools.values()]
            tools_desc = f"\n\nTools ที่ใช้ได้:\n" + "\n".join(tools_list)
        return f"""คุณเป็นผู้ช่วยแพทย์ ตอบคำถามปรนัยด้วยตัวเลือกเท่านั้น

ใช้ tools เมื่อจำเป็น เช่น ค้นหาข้อมูล คำนวณ{tools_desc}

Use documents (secondary reference) as needed:
- doc2.pdf excerpt: {fileA_text}
- doc3.pdf excerpt: {fileB_text}

สำคัญมาก: คุณต้องตอบในรูปแบบ JSON format เท่านั้น โดยมี field ดังนี้:
1. "answer" : ต้องเป็นตัวอักษรตัวเลือก (เช่น ก, ข, ค, ง)
2. "reason" : ต้องอธิบายเหตุผลที่เลือกคำตอบนี้ (ห้ามเป็น null)

ตัวอย่างที่ถูกต้อง:
{{
    "answer": "ก",
    "reason": "เลือกข้อนี้เพราะ..."
}}

ข้อควรระวัง:
- ต้องมีทั้ง answer และ reason เสมอ
- reason ต้องไม่เป็น null
- ต้องตอบเป็น JSON format เท่านั้น
"""

    async def process_message(self, user_message: str) -> str:
        # ไม่ใช้ history เพื่อความเร็ว
        try:
            messages = [
                {"role": "system", "content": self._create_system_prompt()},
                {"role": "user", "content": user_message}
            ]
            
            print(f"Sending to LLM: {messages}")  # Debug log
            response = await self.llm_client.generate(messages)
            print(f"LLM Response: '{response}'")  # Debug log
            
            return response.strip() if response else "No response"
        except Exception as e:
            print(f"LLM Error: {e}")  # Debug log
            return f"Error: {str(e)}"

class NativeMCPPipeline:
    def __init__(self, llm_config: LLMConfig = None, mcp_config: MCPConfig = None):
        self.llm_config = llm_config or LLMConfig()
        self.mcp_config = mcp_config or MCPConfig()
        self.llm_client = None
        self.mcp_session = None
        self.tools: List[SimpleTool] = []
        self.agent: Optional[SimpleAgent] = None

    async def test_llm_connection(self) -> bool:
        try:
            timeout = aiohttp.ClientTimeout(total=None)  # ลด timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                test_url = f"{self.llm_config.base_url}/models"
                async with session.get(test_url) as response:
                    return response.status == 200
        except Exception:
            return False

    async def create_llm_client(self):
        class SimpleLLMClient:
            def __init__(self, config: LLMConfig):
                self.config = config
                self.session = None
                self.timeout = aiohttp.ClientTimeout(total=None)  # ลด timeout

            async def ensure_session(self):
                if not self.session:
                    self.session = aiohttp.ClientSession(timeout=self.timeout)

            async def generate(self, messages: List[Dict[str, str]]) -> str:
                await self.ensure_session()
                payload = {
                    "model": self.config.model,
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "stream": False
                }
                headers = {
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
                
                print(f"LLM Request URL: {self.config.base_url}/chat/completions")  # Debug
                print(f"LLM Payload: {payload}")  # Debug
                
                async with self.session.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload,
                    headers=headers
                ) as response:
                    response_text = await response.text()
                    print(f"LLM Status: {response.status}, Response: {response_text[:200]}")  # Debug
                    
                    if response.status == 200:
                        try:
                            data = await response.json()
                            content = data["choices"][0]["message"]["content"]
                            print(f"Extracted content: '{content}'")  # Debug
                            return content or "Empty response"
                        except (KeyError, IndexError, json.JSONDecodeError) as e:
                            print(f"JSON parse error: {e}")
                            return f"Parse error: {response_text[:100]}"
                    else:
                        return f"HTTP {response.status}: {response_text[:200]}"

            async def close(self):
                if self.session:
                    await self.session.close()

        self.llm_client = SimpleLLMClient(self.llm_config)
        # ทดสอบการเชื่อมต่อด้วย message สั้น
        try:
            test_response = await self.llm_client.generate([{"role": "user", "content": "สวัสดี"}])
            print(f"LLM Test Response: '{test_response}'")
        except Exception as e:
            print(f"LLM Test Failed: {e}")
            raise
        return True

    async def connect_to_mcp_server(self) -> bool:
        # ข้าม MCP connection เพื่อความเร็ว (ใช้ demo tools แทน)
        return False

    async def create_demo_tools(self):
        class DemoTool(SimpleTool):
            def __init__(self, name: str, description: str, func):
                self.name = name
                self.description = description
                self.func = func
                self.input_schema = {}
                self.mcp_session = None

            async def call(self, **kwargs) -> str:
                return self.func(**kwargs)

        def quick_calc(expression: str) -> str:
            try:
                if any(char in expression for char in ['import', 'exec', 'eval', '__']):
                    return "Error: Invalid"
                result = eval(expression.replace('^', '**'))
                return f"{result}"
            except:
                return "Error"

        # ใช้เฉพาะ tools ที่จำเป็น
        self.tools = [
            DemoTool("calc", "Quick calculator", quick_calc)
        ]

    async def initialize(self):
        # ข้าม LLM connection test เพื่อความเร็ว
        try:
            await self.create_llm_client()
        except Exception as e:
            print(f"LLM connection failed: {e}")
            raise
        
        # ใช้ demo tools แทน MCP
        await self.create_demo_tools()
        self.agent = SimpleAgent(self.llm_client, self.tools)

    async def process_message(self, message: str) -> str:
        if not self.agent:
            return "Error: Pipeline not initialized"
        return await self.agent.process_message(message)

    async def cleanup(self):
        if self.llm_client:
            await self.llm_client.close()

# --- Flask + background event loop setup ---
app = Flask(__name__)
pipeline: Optional[NativeMCPPipeline] = None
event_loop: Optional[asyncio.AbstractEventLoop] = None
pipeline_ready = False

def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

async def _init_pipeline():
    global pipeline, pipeline_ready
    try:
        pipeline = NativeMCPPipeline()
        await pipeline.initialize()
        pipeline_ready = True
    except Exception as e:
        print(f"Pipeline init failed: {e}")
        pipeline_ready = False

# เริ่ม event loop และ pipeline ทันทีเมื่อ import
event_loop = asyncio.new_event_loop()
thread = threading.Thread(target=start_background_loop, args=(event_loop,), daemon=True)
thread.start()

# Initialize pipeline ใน background
future = asyncio.run_coroutine_threadsafe(_init_pipeline(), event_loop)

@app.route("/health", methods=["GET"])
def health():
    status = "ready" if pipeline_ready else "initializing"
    return jsonify({"status": status})

@app.route("/eval", methods=["POST"])
def answer():
    global pipeline, event_loop, pipeline_ready
    
    if not pipeline_ready:
        return jsonify({"error": "Pipeline still initializing"}), 503

    payload = request.get_json(silent=True)
    if not payload or "question" not in payload:
        return jsonify({"error": "missing 'question' in JSON body"}), 400
    
    question = payload["question"]

    # Process on background loop
    coro = pipeline.process_message(question)
    future = asyncio.run_coroutine_threadsafe(coro, event_loop)
    try:
        answer_text = future.result(timeout=None)
    except asyncio.TimeoutError:
        return jsonify({"error": "Request timeout"}), 408
    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500

    # Clean up markdown formatting if present
    answer_text = answer_text.strip()
    if answer_text.startswith("```") and answer_text.endswith("```"):
        # Remove markdown code blocks
        answer_text = answer_text.replace("```json\n", "").replace("```", "").strip()
    
    # Try to extract choice and reason
    choice = None
    reason = None
    
    # Parse JSON-like response if present
    try:
        if '{' in answer_text and '}' in answer_text:
            response_dict = json.loads(answer_text)
            if isinstance(response_dict, dict):
                choice = response_dict.get('answer')
                reason = response_dict.get('reason')
    except:
        pass

    # If not JSON, try regular pattern matching
    if not choice:
        patterns = [
            r'Choice:\s*\[([^\]]+)\]',
            r'\[([ก-ฮ]|[a-z]|[A-Z]|[0-9])\]',
            r'Choice:\s*([ก-ฮ]|[a-z]|[A-Z])',
            r'^([ก-ฮ]|[a-z]|[A-Z])[\.\)]',
            r'([ก-ฮ]|[a-z]|[A-Z])$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer_text.strip())
            if match:
                choice = match.group(1).strip()
                break

        # Try to find reason after "reason:" or "เหตุผล:"
        reason_patterns = [
            r'reason:\s*(.+)',
            r'เหตุผล:\s*(.+)',
            r'reason\s*:\s*(.+)',
            r'เหตุผล\s*:\s*(.+)'
        ]
        
        for pattern in reason_patterns:
            match = re.search(pattern, answer_text, re.IGNORECASE)
            if match:
                reason = match.group(1).strip()
                break

    if not choice:
        clean_text = answer_text.strip()
        if len(clean_text) == 1 and clean_text in 'กขคงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟมยรลวศษสหฬอฮabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            choice = clean_text

    # If no reason was found or reason is null, provide a default reason
    if not reason or reason == "null":
        reason = "ไม่พบเหตุผลประกอบการตอบ กรุณาตรวจสอบคำถามและคำตอบอีกครั้ง"
    
    response = {
        "answer": choice,
        "reason": reason
    }
    
    if not choice:
        response["raw"] = answer_text
        response["note"] = "ไม่พบตัวเลือกในฟอร์แมตที่คาดไว้"

    return jsonify(response)

@app.route("/shutdown", methods=["POST"])
def shutdown():
    global pipeline, event_loop
    if pipeline:
        fut = asyncio.run_coroutine_threadsafe(pipeline.cleanup(), event_loop)
        try:
            fut.result(timeout=None)  # ลด timeout
        except Exception:
            pass
    return jsonify({"status": "shut down"})

if __name__ == "__main__":
    # รอให้ pipeline พร้อมก่อน start server
    try:
        future.result(timeout=None)  # รอ initialize
        print("Pipeline ready, starting server...")
    except Exception as e:
        print(f"Warning: Pipeline init incomplete: {e}")
    
    app.run(host="0.0.0.0", port=5000, threaded=True)
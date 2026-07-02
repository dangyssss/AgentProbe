import os
import time
import re
import json
import requests
from typing import Any, Dict, Optional, List, Tuple, Union, Set
from dotenv import load_dotenv

load_dotenv(override=True)

BASE_URL = "https://api.coze.cn"

_MD_IMG = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
_MD_LINK = re.compile(r'\[[^\]]*]\((https?://[^)]+)\)')
_PLAIN_URL = re.compile(r'(https?://[^\s)]+)')

_EVT_DELTA = {
    "conversation.message.delta",
    "message.delta",
}
_EVT_COMPLETED = {
    "conversation.message.completed",
    "message.completed",
}
_EVT_CHAT_COMPLETED = {
    "conversation.chat.completed",
    "chat.completed",
    "completed",
    "done",
}
_EVT_FAILED = {
    "conversation.chat.failed",
    "chat.failed",
    "error",
}


class CozeQuotaError(RuntimeError):
    pass


def _looks_like_image_url(u: str) -> bool:
    u = (u or "").lower()
    if any(u.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp")):
        return True
    if ".jpg?" in u or ".jpeg?" in u or ".png?" in u:
        return True
    return False


def _collect_image_like_urls_from_text(text: str, bucket: List[str]):
    for m in _MD_IMG.finditer(text or ""):
        u = m.group(1).strip()
        if u:
            bucket.append(u)
    for m in _MD_LINK.finditer(text or ""):
        u = m.group(1).strip()
        if u and _looks_like_image_url(u):
            bucket.append(u)
    for m in _PLAIN_URL.finditer(text or ""):
        u = m.group(1).strip()
        if u and _looks_like_image_url(u):
            bucket.append(u)


class CozeAgent:
    def __init__(
        self,
        bot_id: str,
        default_user_id: str = "demo_user",
        timeout: Union[int, Tuple[int, int]] = 60,
    ):
        pat = os.getenv("COZE_PAT")
        if not pat:
            raise RuntimeError("Missing COZE_PAT: Please configure COZE_PAT in your .env file")
        if not bot_id:
            raise ValueError("bot_id cannot be empty")
        self.bot_id = bot_id
        self.default_user_id = default_user_id or "demo_user"
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {pat}",
            "Accept": "text/event-stream",
            "Accept-Charset": "utf-8",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        })

        self.enable_rawlog: bool = True
        self._rawlog_dir: str = os.path.join("debug_logs")
        os.makedirs(self._rawlog_dir, exist_ok=True)

    def upload_file(self, file_path: str) -> Dict[str, Any]:
        """
        Upload local files to Coze and return both file_id and file_url whenever possible
        """
        url = f"{BASE_URL}/v1/files/upload"
        headers = {
            "Authorization": self.session.headers["Authorization"],
        }
        with open(file_path, "rb") as f:
            files = {"file": f}
            r = requests.post(url, headers=headers, files=files, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        raw_data = data.get("data") or data
        file_id = (
            raw_data.get("id")
            or raw_data.get("file_id")
            or data.get("id")
            or data.get("file_id")
        )
        file_url = (
            raw_data.get("url")
            or raw_data.get("download_url")
            or raw_data.get("file_url")
        )
        if not file_id:
            raise RuntimeError(f"upload file fail: {json.dumps(data, ensure_ascii=False)}")
        return {"file_id": file_id, "file_url": file_url, "raw": data}

    def ping(self) -> bool:
        url = f"{BASE_URL}/v3/chat"
        payload = {
            "bot_id": self.bot_id,
            "user_id": "healthcheck",
            "stream": False,
            "additional_messages": [
                {"role": "user", "content": "ping", "content_type": "text"}
            ],
        }
        headers = dict(self.session.headers)
        headers["Accept"] = "application/json"

        r = self.session.post(url, json=payload, timeout=self.timeout, headers=headers)
        r.raise_for_status()
        data = self._safe_json(r.text)

        if isinstance(data, dict) and (data.get("status") == "failed"):
            err = data.get("last_error")
            if isinstance(err, dict) and err.get("code") == 4028:
                return False
            raise RuntimeError(
                f"Coze failed in ping: {json.dumps(data, ensure_ascii=False)}"
            )
        return True

    @staticmethod
    def quota_insufficient_report() -> str:
        return (
            "### ⚠️ Insufficient Coze Credits\n\n"
            "- Error Code: **4028**\n"
            "- Symptom: Request denied (Insufficient Coze credits balance)\n"
            "- Recommendations:\n"
            " 1. Wait for the quota to reset or upgrade to a paid tier;\n"
            " 2. Reduce the number of simultaneous evaluation test cases/concurrency;\n"
            " 3. Invoke ping() for a credit pre-check prior to running evaluations.\n"
            "\n> The current request has been canceled; no tokens were consumed."
        )

    def ask(
        self,
        text: str,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        *,
        image_urls: Optional[List[str]] = None,
        audio_urls: Optional[List[str]] = None,
        image_files: Optional[List[str]] = None,
        original: bool = False,
        precheck: bool = True,
        fallback_to_nostream: bool = True,
    ) -> Dict[str, Any]:
        used_user_id = user_id or self.default_user_id

        if precheck:
            ok = self.ping()
            if not ok:
                raise CozeQuotaError("Coze credits exhausted (4028)")

        image_file_objs: List[Dict[str, Any]] = []
        if image_files:
            for fp in image_files:
                if fp:
                    up = self.upload_file(fp)
                    image_file_objs.append(up)

        stream_result = self._ask_stream(
            text=text,
            user_id=used_user_id,
            conversation_id=conversation_id,
            original=original,
            image_urls=image_urls,
            audio_urls=audio_urls,
            image_file_objs=image_file_objs,
        )

        need_fallback = (
            fallback_to_nostream
            and not (
                (stream_result.get("text") and str(stream_result.get("text")).strip())
                or stream_result.get("images")
                or stream_result.get("audios")
            )
        )

        if need_fallback:
            nostream_result = self._ask_nostream(
                text=text,
                user_id=used_user_id,
                conversation_id=conversation_id,
                original=original,
                image_urls=image_urls,
                audio_urls=audio_urls,
                image_file_objs=image_file_objs,
            )

            merged_calls = (stream_result.get("tool_calls") or []) + (nostream_result.get("tool_calls") or [])
            tool_names = sorted({c.get("name") for c in merged_calls if c.get("name")})

            tr_stream = stream_result.get("tool_responses") or {}
            tr_nostream = nostream_result.get("tool_responses") or {}
            tool_responses = dict(tr_stream)
            tool_responses.update(tr_nostream)

            merged = {
                "text": (str(stream_result.get("text") or "").strip())
                        or nostream_result.get("text"),
                "images": stream_result.get("images")
                          or nostream_result.get("images")
                          or [],
                "audios": stream_result.get("audios")
                          or nostream_result.get("audios")
                          or [],
                "messages": stream_result.get("messages") if original else None,
                "usage": stream_result.get("usage")
                         or nostream_result.get("usage"),
                "latency_ms": stream_result.get("latency_ms")
                              or nostream_result.get("latency_ms"),
                "ttft_ms": stream_result.get("ttft_ms")
                           or nostream_result.get("ttft_ms"),
                "used_user_id": used_user_id,
                "conversation_id": stream_result.get("conversation_id")
                                    or nostream_result.get("conversation_id"),
                "chat_id": stream_result.get("chat_id")
                           or nostream_result.get("chat_id"),
                "logid": stream_result.get("logid")
                         or nostream_result.get("logid"),

                "tool_calls": merged_calls,
                "tool_names": tool_names,
                "tool_responses": tool_responses,

                "_raw_stream": stream_result,
                "_raw_nostream": nostream_result,
            }
            if original:
                merged["messages"] = (
                    merged["messages"]
                    or nostream_result.get("messages")
                    or []
                )

            if merged_calls:
                merged_list = []
                for tc in merged_calls:
                    cid = str(tc.get("id") or "")
                    merged_list.append({**tc, "response": tool_responses.get(cid)})
                merged["tool_calls_with_result"] = merged_list

            return merged

        stream_with_debug = dict(stream_result)
        stream_with_debug["_raw_stream"] = dict(stream_result)
        stream_with_debug["_raw_nostream"] = None
        return stream_with_debug

    def _build_single_object_message(
        self,
        *,
        text: str,
        image_urls: Optional[List[str]],
        audio_urls: Optional[List[str]],
        image_file_objs: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Construct a message with content_type=object_string and encapsulate everything inside it
        """
        objs: List[Dict[str, Any]] = []

        if text:
            objs.append({"type": "text", "text": text})

        if image_file_objs:
            for obj in image_file_objs:
                fid = obj.get("file_id")
                if fid:
                    objs.append({"type": "image", "file_id": fid})

        if image_file_objs:
            for obj in image_file_objs:
                url_candidate = obj.get("file_url")
                if url_candidate:
                    objs.append({"type": "text", "text": url_candidate})

        if image_urls:
            for u in image_urls:
                u = (u or "").strip()
                if u:
                    objs.append({"type": "text", "text": u})

        if audio_urls:
            for u in audio_urls:
                u = (u or "").strip()
                if u:
                    objs.append({"type": "text", "text": f"[audio] {u}"})

        if not objs:
            objs.append({"type": "text", "text": ""})

        return {
            "role": "user",
            "content_type": "object_string",
            "content": json.dumps(objs, ensure_ascii=False),
        }

    def _ask_stream(
        self,
        *,
        text: str,
        user_id: str,
        conversation_id: Optional[str],
        original: bool,
        image_urls: Optional[List[str]] = None,
        audio_urls: Optional[List[str]] = None,
        image_file_objs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        url = f"{BASE_URL}/v3/chat"

        single_msg = self._build_single_object_message(
            text=text,
            image_urls=image_urls,
            audio_urls=audio_urls,
            image_file_objs=image_file_objs,
        )

        body: Dict[str, Any] = {
            "bot_id": self.bot_id,
            "user_id": user_id,
            "stream": True,
            "additional_messages": [single_msg],
        }
        if conversation_id:
            body["conversation_id"] = conversation_id

        start = time.perf_counter()
        ttft_ms: Optional[float] = None

        final_text_chunks: List[str] = []
        final_image_urls: List[str] = []
        final_audio_urls: List[str] = []
        seen_delta_by_chat: Dict[str, bool] = {}
        original_messages: List[Dict[str, Any]] = []
        usage: Optional[Dict[str, Any]] = None
        conv_id: Optional[str] = None
        top_chat_id: Optional[str] = None
        logid: Optional[str] = None

        sse_chunks: List[Dict[str, Any]] = []
        trace_nodes: List[Dict[str, Any]] = []

        event_name = "message"
        data_lines: List[bytes] = []

        def _bump_ttft_once():
            nonlocal ttft_ms
            if ttft_ms is None:
                ttft_ms = round((time.perf_counter() - start) * 1000, 2)

        def _add_image(url: str, bucket: List[str]):
            url = (url or "").strip()
            if not url:
                return
            if url.startswith("<") and url.endswith(">"):
                url = url[1:-1]
            bucket.append(url)

        def _extract_images_from_any_text(md: str, bucket: List[str]):
            _collect_image_like_urls_from_text(md or "", bucket)

        def _append_answer_to_final(data: Dict[str, Any]):
            if (data.get("role") != "assistant") or (
                str(data.get("type")).lower() not in ("answer", "completed")
            ):
                return

            ctype = (data.get("content_type") or "").lower()
            content = data.get("content")

            text_part = ""
            imgs: List[str] = []
            audios: List[str] = []

            if ctype in ("", "text", "plain_text", "markdown", "md", None):
                parts: List[str] = []
                if isinstance(content, str):
                    parts.append(content)
                    _extract_images_from_any_text(content, imgs)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                            _extract_images_from_any_text(item, imgs)
                        elif isinstance(item, dict):
                            for k in ("text", "content", "value"):
                                v = item.get(k)
                                if isinstance(v, str) and v:
                                    parts.append(v)
                                    _extract_images_from_any_text(v, imgs)
                                    break
                            for k in ("image", "image_url", "url", "src"):
                                v = item.get(k)
                                if isinstance(v, str):
                                    _add_image(v, imgs)
                                elif isinstance(v, dict):
                                    for kk in ("url", "src"):
                                        vv = v.get(kk)
                                        if isinstance(vv, str):
                                            _add_image(vv, imgs)
                                            break
                            for k in ("audio", "audio_url"):
                                v = item.get(k)
                                if isinstance(v, str):
                                    audios.append(v.strip())
                elif isinstance(content, dict):
                    for k in ("text", "content", "value"):
                        v = content.get(k)
                        if isinstance(v, str) and v:
                            parts.append(v)
                            _extract_images_from_any_text(v, imgs)
                            break
                    for k in ("image", "image_url", "url", "src"):
                        v = content.get(k)
                        if isinstance(v, str):
                            _add_image(v, imgs)
                        elif isinstance(v, dict):
                            for kk in ("url", "src"):
                                vv = v.get(kk)
                                if isinstance(vv, str):
                                    _add_image(vv, imgs)
                                    break
                    for k in ("audio", "audio_url"):
                        v = content.get(k)
                        if isinstance(v, str):
                            audios.append(v.strip())
                if parts:
                    text_part = "".join(parts)

            elif ctype in (
                "image",
                "img",
                "picture",
                "image_url",
                "image/jpeg",
                "image/png",
            ):
                if isinstance(content, str):
                    _add_image(content, imgs)
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            _add_image(item, imgs)
                        elif isinstance(item, dict):
                            for k in ("url", "src", "image_url", "image"):
                                v = item.get(k)
                                if isinstance(v, str):
                                    _add_image(v, imgs)
                                    break
                elif isinstance(content, dict):
                    for k in ("url", "src", "image_url", "image"):
                        v = content.get(k)
                        if isinstance(v, str):
                            _add_image(v, imgs)
                        elif isinstance(v, dict):
                            for kk in ("url", "src"):
                                vv = v.get(kk)
                                if isinstance(vv, str):
                                    _add_image(vv, imgs)
                                    break

            elif ctype in (
                "audio",
                "audio/mpeg",
                "audio/mp3",
                "audio/wav",
                "voice",
            ):
                if isinstance(content, str):
                    audios.append(content.strip())
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, str):
                            audios.append(item.strip())
                        elif isinstance(item, dict):
                            for k in ("url", "src", "audio_url", "audio"):
                                v = item.get(k)
                                if isinstance(v, str):
                                    audios.append(v.strip())
                                    break
                elif isinstance(content, dict):
                    for k in ("url", "src", "audio_url", "audio"):
                        v = content.get(k)
                        if isinstance(v, str):
                            audios.append(v.strip())
                            break
            else:
                return

            if text_part or imgs or audios:
                _bump_ttft_once()
                if text_part:
                    final_text_chunks.append(self._fix_mojibake(text_part))
                if imgs:
                    final_image_urls.extend(imgs)
                if audios:
                    final_audio_urls.extend(audios)

        def _raise_if_quota_error(failed_payload: Dict[str, Any]):
            if not isinstance(failed_payload, dict):
                return
            err = failed_payload.get("last_error")
            if isinstance(err, dict) and err.get("code") == 4028:
                raise CozeQuotaError("Coze credits exhausted (4028)")

        with self.session.post(url, json=body, stream=True, timeout=self.timeout) as r:
            r.raise_for_status()
            try:
                logid_header = r.headers.get("X-Request-Id")
                if logid_header:
                    logid = logid_header
            except Exception:
                pass

            for raw in r.iter_lines(decode_unicode=False):
                if raw is None:
                    continue

                if raw == b"":
                    if data_lines:
                        evt = self._emit_event(event_name, data_lines)
                        en = (evt.get("event") or "").lower()
                        data_raw = evt.get("data")
                        data = self._as_dict(data_raw)

                        sse_chunks.append({"event": evt.get("event"), "data": evt.get("data")})
                        if isinstance(data, dict):
                            trace_nodes.append(data)

                        if (
                            isinstance(data_raw, str)
                            and data_raw.strip() == "[DONE]"
                        ) or (
                            isinstance(data, dict)
                            and data.get("raw") == "[DONE]"
                        ):
                            event_name = "message"
                            data_lines.clear()
                            continue

                        if en in _EVT_FAILED:
                            _raise_if_quota_error(data)
                            raise RuntimeError(
                                "Coze failed: %s"
                                % json.dumps(data, ensure_ascii=False)
                            )

                        if en in _EVT_DELTA:
                            cid = (data.get("chat_id") or "")
                            if cid:
                                seen_delta_by_chat[cid] = True
                            if not original:
                                _append_answer_to_final(data)

                        elif en in _EVT_COMPLETED:
                            cid = (data.get("chat_id") or "")
                            if not original:
                                if (not cid) or (
                                    not seen_delta_by_chat.get(cid, False)
                                ):
                                    _append_answer_to_final(data)
                            if (data.get("role") == "assistant"):
                                original_messages.append(data)
                        elif en in _EVT_CHAT_COMPLETED:
                            usage_candidate = (
                                (data.get("chat") or {}).get("usage")
                                or data.get("usage")
                                or (data.get("detail") or {}).get("usage")
                            )
                            if usage_candidate:
                                usage = usage_candidate
                            conv_id = data.get("conversation_id") or conv_id
                            top_chat_id = data.get("chat_id") or top_chat_id
                            logid = (
                                (data.get("detail") or {}).get("logid")
                                or data.get("logid")
                                or logid
                            )

                        event_name = "message"
                        data_lines.clear()
                        continue

                if raw.startswith(b"event:"):
                    event_name = (
                        raw[len(b"event:"):]
                        .strip()
                        .decode("utf-8", errors="replace")
                        or "message"
                    )
                elif raw.startswith(b"data:"):
                    data_lines.append(raw[len(b"data:"):].lstrip())
                else:
                    data_lines.append(raw)

            if data_lines:
                evt = self._emit_event(event_name, data_lines)
                en = (evt.get("event") or "").lower()
                data_raw = evt.get("data")
                data = self._as_dict(data_raw)

                sse_chunks.append({"event": evt.get("event"), "data": evt.get("data")})
                if isinstance(data, dict):
                    trace_nodes.append(data)

                if en in _EVT_COMPLETED:
                    cid = (data.get("chat_id") or "")
                    if not original:
                        if (not cid) or (not seen_delta_by_chat.get(cid, False)):
                            _append_answer_to_final(data)
                    if (data.get("role") == "assistant"):
                        original_messages.append(data)

                if en in _EVT_CHAT_COMPLETED:
                    usage_candidate = (
                        (data.get("chat") or {}).get("usage")
                        or data.get("usage")
                        or (data.get("detail") or {}).get("usage")
                    )
                    if usage_candidate:
                        usage = usage_candidate
                    conv_id = data.get("conversation_id") or conv_id
                    top_chat_id = data.get("chat_id") or top_chat_id
                    logid = (
                        (data.get("detail") or {}).get("logid")
                        or data.get("logid")
                        or logid
                    )

        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        final_images: List[str] = []
        seen_img: Set[str] = set()
        for u in final_image_urls:
            if u and u not in seen_img:
                seen_img.add(u)
                final_images.append(u)

        final_audios: List[str] = []
        seen_aud: Set[str] = set()
        for u in final_audio_urls:
            if u and u not in seen_aud:
                seen_aud.add(u)
                final_audios.append(u)

        result: Dict[str, Any] = {
            "usage": usage,
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
            "used_user_id": user_id,
            "conversation_id": conv_id,
            "chat_id": top_chat_id,
            "logid": logid,
        }

        if original:
            result["messages"] = original_messages
        else:
            result["text"] = "".join(final_text_chunks)
            result["images"] = final_images
            result["audios"] = final_audios

        tool_calls = self._extract_tool_calls_from_messages({"messages": trace_nodes})
        result["tool_calls"] = tool_calls or []
        result["tool_names"] = sorted({ tc.get("name") for tc in tool_calls if tc.get("name") }) if tool_calls else []

        tool_responses = self._extract_tool_responses_from_messages({"messages": trace_nodes})
        result["tool_responses"] = tool_responses

        if tool_calls:
            merged = []
            for tc in tool_calls:
                cid = str(tc.get("id") or "")
                merged.append({**tc, "response": tool_responses.get(cid)})
            result["tool_calls_with_result"] = merged

        if self.enable_rawlog and sse_chunks:
            try:
                import datetime as _dt
                p = os.path.join(self._rawlog_dir, f"sse_stream_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(sse_chunks, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] Failed to write raw sse_stream log: {e}")

        return result

    def _ask_nostream(
        self,
        *,
        text: str,
        user_id: str,
        conversation_id: Optional[str],
        original: bool,
        image_urls: Optional[List[str]] = None,
        audio_urls: Optional[List[str]] = None,
        image_file_objs: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        url = f"{BASE_URL}/v3/chat"

        single_msg = self._build_single_object_message(
            text=text,
            image_urls=image_urls,
            audio_urls=audio_urls,
            image_file_objs=image_file_objs,
        )

        body: Dict[str, Any] = {
            "bot_id": self.bot_id,
            "user_id": user_id,
            "stream": False,
            "additional_messages": [single_msg],
        }
        if conversation_id:
            body["conversation_id"] = conversation_id

        headers = dict(self.session.headers)
        headers["Accept"] = "application/json"

        start = time.perf_counter()
        r = self.session.post(url, json=body, timeout=self.timeout, headers=headers)
        r.raise_for_status()
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        data_raw = r.text
        data_any = self._safe_json(data_raw)
        data = self._as_dict(data_any)

        if isinstance(data, dict) and data.get("status") == "failed":
            err = data.get("last_error") or {}
            if isinstance(err, dict) and err.get("code") == 4028:
                raise CozeQuotaError("Coze credits exhausted (4028)")
            raise RuntimeError(
                "Coze nostream failed: %s"
                % json.dumps(data, ensure_ascii=False)
            )

        usage = (
            (data.get("chat") or {}).get("usage")
            or data.get("usage")
            or (data.get("detail") or {}).get("usage")
            or None
        )
        conv_id = (
            data.get("conversation_id")
            or (data.get("chat") or {}).get("conversation_id")
        )
        chat_id = data.get("chat_id") or (data.get("chat") or {}).get("chat_id")
        logid = (data.get("detail") or {}).get("logid") or data.get("logid")

        messages: List[Dict[str, Any]] = []
        text_out = ""
        images_out: List[str] = []
        audios_out: List[str] = []
        text_parts: List[str] = []

        msgs = data.get("messages")
        if isinstance(msgs, list):
            messages = [m for m in msgs if isinstance(m, dict)]
            for m in messages:
                if (
                    m.get("role") == "assistant"
                    and str(m.get("type")).lower() in ("answer", "completed")
                ):
                    self._merge_answer_into(
                        m,
                        text_acc=lambda s: text_parts.append(s),
                        imgs_acc=images_out,
                        audios_acc=audios_out,
                    )

        if text_parts:
            text_out = "".join(text_parts)
        elif messages:
            parts: List[str] = []
            for m in messages:
                if m.get("role") == "assistant":
                    c = m.get("content")
                    if isinstance(c, str):
                        parts.append(c)
                    elif isinstance(c, dict):
                        for k in ("text", "content", "value"):
                            v = c.get(k)
                            if isinstance(v, str) and v:
                                parts.append(v)
                                break
            text_out = "".join(parts) if parts else ""

        result: Dict[str, Any] = {
            "usage": usage,
            "latency_ms": latency_ms,
            "ttft_ms": None,
            "used_user_id": user_id,
            "conversation_id": conv_id,
            "chat_id": chat_id,
            "logid": logid,
        }
        if original:
            result["messages"] = messages or []
        else:
            result["text"] = self._fix_mojibake(text_out or "")
            final_images: List[str] = []
            seen_img: Set[str] = set()
            for u in images_out:
                if u and u not in seen_img:
                    seen_img.add(u)
                    final_images.append(u)
            result["images"] = final_images

            final_audios: List[str] = []
            seen_aud: Set[str] = set()
            for u in audios_out:
                if u and u not in seen_aud:
                    seen_aud.add(u)
                    final_audios.append(u)
            result["audios"] = final_audios

        tool_calls = self._extract_tool_calls_from_messages(data)
        result["tool_calls"] = tool_calls or []
        result["tool_names"] = sorted({ tc.get("name") for tc in tool_calls if tc.get("name") }) if tool_calls else []

        tool_responses = self._extract_tool_responses_from_messages(data)
        result["tool_responses"] = tool_responses

        if tool_calls:
            merged = []
            for tc in tool_calls:
                cid = str(tc.get("id") or "")
                merged.append({**tc, "response": tool_responses.get(cid)})
            result["tool_calls_with_result"] = merged

        if self.enable_rawlog:
            try:
                import datetime as _dt
                p = os.path.join(self._rawlog_dir, f"nostream_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] Failed to write raw no stream log: {e}")

        return result

    @staticmethod
    def _emit_event(event_name: str, data_lines: List[bytes]) -> Dict[str, Any]:
        data_txt = b"\n".join(data_lines).decode("utf-8", errors="replace")
        return {"event": event_name, "data": CozeAgent._safe_json(data_txt)}

    @staticmethod
    def _safe_json(s: str) -> Any:
        try:
            return json.loads(s)
        except Exception:
            return {"raw": s}

    @staticmethod
    def _as_dict(x: Any) -> Dict[str, Any]:
        if isinstance(x, dict):
            return x
        return {"raw": x}

    @staticmethod
    def _fix_mojibake(s: str) -> str:
        try:
            s2 = s.encode("latin-1", "ignore").decode("utf-8", "ignore")

            def cjk_ratio(t: str) -> float:
                return sum(0x4E00 <= ord(ch) <= 0x9FFF for ch in t) / max(len(t), 1)

            return s2 if cjk_ratio(s2) > cjk_ratio(s) else s
        except Exception:
            return s

    def _merge_answer_into(self, data: Dict[str, Any], text_acc, imgs_acc: List[str], audios_acc: Optional[List[str]] = None):
        if (data.get("role") != "assistant") or (
            str(data.get("type")).lower() not in ("answer", "completed")
        ):
            return

        ctype = (data.get("content_type") or "").lower()
        content = data.get("content")

        def _add_image(url: str):
            u = (url or "").strip()
            if not u:
                return
            if u.startswith("<") and u.endswith(">"):
                u = u[1:-1]
            imgs_acc.append(u)

        def _add_audio(url: str):
            if audios_acc is None:
                return
            u = (url or "").strip()
            if not u:
                return
            if u.startswith("<") and u.endswith(">"):
                u = u[1:-1]
            audios_acc.append(u)

        def _extract_images_from_any_text(md: str):
            _collect_image_like_urls_from_text(md or "", imgs_acc)

        parts: List[str] = []

        if ctype in ("", "text", "plain_text", "markdown", "md", None):
            if isinstance(content, str):
                parts.append(content)
                _extract_images_from_any_text(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                        _extract_images_from_any_text(item)
                    elif isinstance(item, dict):
                        for k in ("text", "content", "value"):
                            v = item.get(k)
                            if isinstance(v, str) and v:
                                parts.append(v)
                                _extract_images_from_any_text(v)
                                break
                        for k in ("image", "image_url", "url", "src"):
                            v = item.get(k)
                            if isinstance(v, str):
                                _add_image(v)
                            elif isinstance(v, dict):
                                for kk in ("url", "src"):
                                    vv = v.get(kk)
                                    if isinstance(vv, str):
                                        _add_image(vv)
                                        break
                        for k in ("audio", "audio_url"):
                            v = item.get(k)
                            if isinstance(v, str):
                                _add_audio(v)
            elif isinstance(content, dict):
                for k in ("text", "content", "value"):
                    v = content.get(k)
                    if isinstance(v, str) and v:
                        parts.append(v)
                        _extract_images_from_any_text(v)
                        break
                for k in ("image", "image_url", "url", "src"):
                    v = content.get(k)
                    if isinstance(v, str):
                        _add_image(v)
                    elif isinstance(v, dict):
                        for kk in ("url", "src"):
                            vv = v.get(kk)
                            if isinstance(vv, str):
                                _add_image(vv)
                                break
                for k in ("audio", "audio_url"):
                    v = content.get(k)
                    if isinstance(v, str):
                        _add_audio(v)

        elif ctype in (
            "image",
            "img",
            "picture",
            "image_url",
            "image/jpeg",
            "image/png",
        ):
            if isinstance(content, str):
                _add_image(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        _add_image(item)
                    elif isinstance(item, dict):
                        for k in ("url", "src", "image_url", "image"):
                            v = item.get(k)
                            if isinstance(v, str):
                                _add_image(v)
                                break
            elif isinstance(content, dict):
                for k in ("url", "src", "image_url", "image"):
                    v = content.get(k)
                    if isinstance(v, str):
                        _add_image(v)
                    elif isinstance(v, dict):
                        for kk in ("url", "src"):
                            vv = v.get(kk)
                            if isinstance(vv, str):
                                _add_image(vv)
                                break

        elif ctype in (
            "audio",
            "audio/mpeg",
            "audio/mp3",
            "audio/wav",
            "voice",
        ):
            if isinstance(content, str):
                _add_audio(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        _add_audio(item)
                    elif isinstance(item, dict):
                        for k in ("url", "src", "audio_url", "audio"):
                            v = item.get(k)
                            if isinstance(v, str):
                                _add_audio(v)
                                break
            elif isinstance(content, dict):
                for k in ("url", "src", "audio_url", "audio"):
                    v = content.get(k)
                    if isinstance(v, str):
                        _add_audio(v)
                        break

        if parts:
            text_acc(self._fix_mojibake("".join(parts)))

    def _extract_tool_calls_from_messages(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        import json as _json
        from typing import Optional

        def _push(calls: List[Dict[str, Any]],
                  name: Optional[str],
                  args: Any,
                  raw: Dict[str, Any],
                  _id: Any = None) -> None:
            if not name:
                return
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except Exception:
                    args = {"_": args}
            if args is None:
                args = {}
            calls.append({
                "id": _id or raw.get("id"),
                "name": name,
                "arguments": args,
                "raw": raw,
            })

        calls: List[Dict[str, Any]] = []

        candidates: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            candidates.append(data)
            msgs = data.get("messages")
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, dict):
                        candidates.append(m)
                        c = m.get("content")
                        if isinstance(c, dict):
                            candidates.append(c)
                        elif isinstance(c, list):
                            for it in c:
                                if isinstance(it, dict):
                                    candidates.append(it)

        for node in candidates:
            if not isinstance(node, dict):
                continue
            tcs = node.get("tool_calls")
            if isinstance(tcs, list):
                for tc in tcs:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    if isinstance(fn, dict):
                        name = fn.get("name") or fn.get("tool_name")
                        _push(calls, name, fn.get("arguments"), tc, tc.get("id"))

            fnc = node.get("function_call")
            if isinstance(fnc, dict):
                _push(calls, fnc.get("name") or fnc.get("tool_name"), fnc.get("arguments"), node, fnc.get("id"))

            if (node.get("tool_name") or node.get("name")) and node.get("arguments") is not None:
                _push(calls, node.get("tool_name") or node.get("name"), node.get("arguments"), node, node.get("id"))

            if node.get("type") in ("tool_call", "function_call"):
                if node.get("name") or node.get("tool_name"):
                    _push(calls, node.get("name") or node.get("tool_name"), node.get("arguments"), node, node.get("id"))

                content = node.get("content")
                if isinstance(content, str):
                    try:
                        fc = _json.loads(content)
                        name = fc.get("name") or fc.get("api_name") or fc.get("plugin_name") or fc.get("plugin")
                        args = fc.get("arguments")
                        call_id = (node.get("meta_data") or {}).get("call_id") or fc.get("call_id") or node.get("id")
                        _push(calls, name, args, {"node": node, "parsed": fc}, call_id)
                    except Exception:
                        pass
            if isinstance(node.get("content"), dict):
                c2 = node["content"]
                tcs2 = c2.get("tool_calls")
                if isinstance(tcs2, list):
                    for tc in tcs2:
                        fn = (tc or {}).get("function") or {}
                        _push(calls, fn.get("name") or fn.get("tool_name"), fn.get("arguments"), tc, tc.get("id"))
                fnc2 = c2.get("function_call")
                if isinstance(fnc2, dict):
                    _push(calls, fnc2.get("name") or fnc2.get("tool_name"), fnc2.get("arguments"), c2, fnc2.get("id"))

            actions = node.get("actions")
            if isinstance(actions, list):
                for act in actions:
                    if isinstance(act, dict):
                        _push(calls, act.get("name") or act.get("tool_name"), act.get("arguments"), act, act.get("id"))
            plugins = node.get("plugins")
            if isinstance(plugins, list):
                for pl in plugins:
                    if isinstance(pl, dict):
                        name = pl.get("api_name") or pl.get("plugin") or pl.get("name")
                        _push(calls, name, pl.get("arguments") or pl.get("params"), pl, pl.get("id"))

        return calls

    def _extract_tool_responses_from_messages(self, data: Dict[str, Any]) -> Dict[str, Any]:
        import json as _json
        out: Dict[str, Any] = {}

        candidates: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            candidates.append(data)
            msgs = data.get("messages")
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, dict):
                        candidates.append(m)

        for node in candidates:
            if not isinstance(node, dict):
                continue
            if (node.get("role") == "assistant") and (str(node.get("type")).lower() == "tool_response"):
                meta = node.get("meta_data") or {}
                call_id = meta.get("call_id") or node.get("id")
                content = node.get("content")
                payload = None
                if isinstance(content, str):
                    try:
                        payload = _json.loads(content)
                    except Exception:
                        payload = {"_raw": content}
                elif isinstance(content, dict):
                    payload = content
                else:
                    payload = {"_raw": content}

                code = None
                message = None
                data_inner = None
                if isinstance(payload, dict):
                    code = payload.get("code")
                    message = payload.get("message") or payload.get("msg")
                    data_inner = payload.get("data")

                ok = (code in (0, "0", None))
                out[str(call_id)] = {
                    "ok": bool(ok),
                    "code": code,
                    "message": message,
                    "data": data_inner,
                    "raw": node,
                }
        return out


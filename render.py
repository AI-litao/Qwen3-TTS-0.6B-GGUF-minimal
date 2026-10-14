#!/usr/bin/env python3
"""Qwen3-TTS-GGUF 极简渲染入口：文本 → wav。

两种模式（二选一）：
  内置音色:  python render.py --model-dir <模型包> --text "你好" --speaker Vivian
  声音克隆:  python render.py --model-dir <模型包> --text "你好" \
                 --voice-wav ref.wav --ref-text "参考音频里说的原话"

模型包 = Qwen3-TTS-0.6B GGUF 转换套件目录（含 talker/predictor gguf、
decoder/编码器 onnx、embeddings/、tokenizer.json）。
"""
import argparse
import glob
import os
import sys
import time

# (引擎硬编码文件名, 同前缀实际文件的 glob stem)
NEEDED = [
    ("qwen3_tts_talker.q5_k.gguf", "qwen3_tts_talker"),
    ("qwen3_tts_predictor.q8_0.gguf", "qwen3_tts_predictor"),
    ("qwen3_tts_decoder.fp16.onnx", "qwen3_tts_decoder"),
    ("qwen3_tts_codec_encoder.fp16.onnx", "qwen3_tts_codec_encoder"),
    ("qwen3_tts_speaker_encoder.fp16.onnx", "qwen3_tts_speaker_encoder"),
    ("tokenizer.json", "tokenizer"),
]


def link(src, dst):
    """软链；Windows 无权限时退硬链（同盘可用）；再退复制。"""
    for fn in (os.symlink, os.link):
        try:
            fn(src, dst)
            return
        except OSError:
            if os.path.exists(dst):
                return
    import shutil
    shutil.copy2(src, dst)


def ensure_model_dir(model_dir):
    """引擎硬编码 talker 文件名 q5_k（历史原因），套件里常是 q4_0 等。
    缺名时在模型包旁建 <包名>_vdir 别名目录（软链/硬链，不复制大文件）。"""
    missing = [(n, s) for n, s in NEEDED if not os.path.exists(os.path.join(model_dir, n))]
    if not missing and os.path.isdir(os.path.join(model_dir, "embeddings")):
        return model_dir
    vdir = model_dir.rstrip("/\\") + "_vdir"
    os.makedirs(vdir, exist_ok=True)
    for name, stem in NEEDED:
        dst = os.path.join(vdir, name)
        if os.path.exists(dst):
            continue
        if os.path.exists(os.path.join(model_dir, name)):
            link(os.path.join(model_dir, name), dst)
            continue
        cands = sorted(glob.glob(os.path.join(model_dir, stem + "*")))
        cands = [c for c in cands if os.path.isfile(c)]
        if not cands:
            sys.exit(f"错误：模型包里找不到 {stem}*（需要 {name}）")
        link(cands[0], dst)
        print(f"[vdir] {name} -> {os.path.basename(cands[0])}")
    emb = os.path.join(vdir, "embeddings")
    if not os.path.exists(emb):
        link(os.path.join(model_dir, "embeddings"), emb)
    return vdir


def main():
    ap = argparse.ArgumentParser(description="Qwen3-TTS-GGUF 极简渲染")
    ap.add_argument("--model-dir", required=True, help="模型包目录")
    ap.add_argument("--text", required=True, help="要合成的文本")
    ap.add_argument("--out", default="out.wav", help="输出 wav 路径")
    ap.add_argument("--speaker", help="内置音色名（与 --voice-wav 二选一）")
    ap.add_argument("--instruct", default=None, help="渲染指令，如'用温柔的语气说'")
    ap.add_argument("--voice-wav", help="克隆参考音频 wav（与 --speaker 二选一）")
    ap.add_argument("--ref-text", help="参考音频的原话转写（克隆必填，须逐字覆盖音频）")
    ap.add_argument("--zero-shot", action="store_true",
                    help="零样本克隆（只用声纹，弱克隆，免 --ref-text）")
    ap.add_argument("--language", default="chinese",
                    help="chinese/english/japanese/korean/... 或 auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--provider", default="CPU", help="ONNX 后端: CPU/DirectML/CUDA")
    args = ap.parse_args()

    if bool(args.speaker) == bool(args.voice_wav):
        sys.exit("错误：--speaker 与 --voice-wav 必须二选一")
    if args.voice_wav and not args.zero_shot and not args.ref_text:
        sys.exit("错误：克隆需要 --ref-text（参考音频原话转写）；"
                 "或用 --zero-shot 走免台词弱克隆")

    vdir = ensure_model_dir(os.path.abspath(args.model_dir))

    from qwen3_tts_gguf.inference import TTSEngine, TTSConfig

    t0 = time.time()
    engine = TTSEngine(model_dir=vdir, onnx_provider=args.provider,
                       llm_use_gpu=False, subprocess_decoder=False, chunk_size=12)
    stream = engine.create_stream()
    if stream is None:
        sys.exit("错误：引擎 create_stream() 返回 None（检查模型包文件是否齐全）")
    print(f"引擎就绪 {time.time()-t0:.1f}s")

    cfg = TTSConfig(max_steps=args.max_steps, temperature=0.8, sub_temperature=0.8,
                    seed=args.seed, sub_seed=args.seed + 3, streaming=False)

    t1 = time.time()
    if args.speaker:
        res = stream.custom(text=args.text, speaker=args.speaker,
                            language=args.language, instruct=args.instruct, config=cfg)
    else:
        if not stream.set_voice(args.voice_wav, text=None if args.zero_shot else args.ref_text):
            sys.exit("错误：set_voice 失败（参考音频读不到？）")
        res = stream.clone(args.text, language=args.language,
                           zero_shot=args.zero_shot, config=cfg)
    if res is None or res.audio is None:
        sys.exit("错误：生成失败（runaway 或采样异常，看日志）")

    res.save(args.out)
    dur = len(res.audio) / 24000
    dt = time.time() - t1
    print(f"完成: {args.out}  音频 {dur:.1f}s  生成 {dt:.1f}s  RTF={dt/dur:.2f}")
    stream.reset()
    engine.shutdown()


if __name__ == "__main__":
    main()
